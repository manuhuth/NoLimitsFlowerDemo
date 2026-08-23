"""Model catalog, data loaders, site partitioning and theta glue. No Flower imports.

Four models federate through the same primitive, `objective_and_gradient(method, ctx,
theta)`, which is a sum over subjects, so with disjoint subjects per site the summed
site contributions ARE the pooled-data value and gradient (the exact federated-learning
property). The `CATALOG` pairs, per model, a model string, a real-data loader + column
map, an estimator default, a site count and an acceptance kind.

Theta on the wire is the TRANSFORMED (optimization) scale, which keeps positive
parameters positive without server-side bounds. The server (a worker thread that can
never boot Julia) reconstructs natural-scale numbers from a per-coordinate LOG MASK the
sites report in the prepare round: `inverse_transform(0)` is 1 on a log-scaled
coordinate and 0 on an identity one, so `to_natural`/`precondition_scale` need no
model-specific bookkeeping and work the same for the 87-weight neural model and the
5-parameter growth model. `objective_and_gradient(method, ctx, theta; scale=:transformed)`
wants theta on the NATURAL scale and returns the gradient on the transformed axes, so the
client maps the wire vector back with the model's own inverse transform (all in one Julia
helper, `nlf_objgrad`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"

# ----------------------------------------------------------------------------------
# Model 1 - warfarin (regression baseline). One-compartment oral absorption with
# multiplicative log-normal random effects on ka, cl, v. Linear ODE -> closed form.
# ----------------------------------------------------------------------------------
WARFARIN_MODEL = """
@fixedEffects begin
    ka       = RealNumber(1.0)
    cl       = RealNumber(0.13)
    v        = RealNumber(8.0)
    omega_ka = RealNumber(0.4, scale=:log)
    omega_cl = RealNumber(0.3, scale=:log)
    omega_v  = RealNumber(0.2, scale=:log)
    sigma    = RealNumber(0.5, scale=:log)
end

@covariates begin
    t    = Covariate()
    Dose = ConstantCovariate(constant_on=:ID)
end

@randomEffects begin
    eta_ka = RandomEffect(LogNormal(0.0, omega_ka); column=:ID)
    eta_cl = RandomEffect(LogNormal(0.0, omega_cl); column=:ID)
    eta_v  = RandomEffect(LogNormal(0.0, omega_v);  column=:ID)
end

@preDifferentialEquation begin
    kai = ka * eta_ka
    cli = cl * eta_cl
    vi  = v * eta_v
end

@DifferentialEquation begin
    D(depot)   ~ -kai * depot
    D(central) ~ kai * depot - (cli / vi) * central
end

@initialDE begin
    depot   = Dose
    central = 0.0
end

@formulas begin
    cp = central(t) / vi
    conc ~ Normal(cp, sigma)
end
"""

# ----------------------------------------------------------------------------------
# Model 2 - theophylline (real). Same 1-compartment oral PK family as warfarin, the
# proven model from NoLimitsPy examples/theophylline.py with :Subject renamed to :id.
# ----------------------------------------------------------------------------------
THEOPH_MODEL = """
@fixedEffects begin
    ka       = RealNumber(1.5)
    cl       = RealNumber(0.04)
    v        = RealNumber(0.5)
    omega_ka = RealNumber(0.5, scale=:log)
    omega_cl = RealNumber(0.3, scale=:log)
    omega_v  = RealNumber(0.3, scale=:log)
    sigma    = RealNumber(0.7, scale=:log)
end

@covariates begin
    t    = Covariate()
    Dose = ConstantCovariate(constant_on=:id)
end

@randomEffects begin
    eta_ka = RandomEffect(LogNormal(0.0, omega_ka); column=:id)
    eta_cl = RandomEffect(LogNormal(0.0, omega_cl); column=:id)
    eta_v  = RandomEffect(LogNormal(0.0, omega_v);  column=:id)
end

@preDifferentialEquation begin
    kai = ka * eta_ka
    cli = cl * eta_cl
    vi  = v * eta_v
end

@DifferentialEquation begin
    D(depot)   ~ -kai * depot
    D(central) ~ kai * depot - (cli / vi) * central
end

@initialDE begin
    depot   = Dose
    central = 0.0
end

@formulas begin
    cp = central(t) / vi
    conc ~ Normal(cp, sigma)
end
"""

# ----------------------------------------------------------------------------------
# Model 3 - warfarin-nn (neural mixed effects on the SAME real warfarin data). The FFNN
# `seed` is PINNED so every site builds identical initial weights: without it theta0
# disagrees across sites and the prepare-round agreement check fails. 87 parameters
# (sigma + 86 network weights).
# ----------------------------------------------------------------------------------
WARFARIN_NN_MODEL = """
@Model begin
    @covariates begin
        t = Covariate()
        d = ConstantCovariate()
    end
    @fixedEffects begin
        sigma = RealNumber(1.0, scale=:log)
        nn_params = FFNNParameters((3, 5, 5, 5, 1); activation=:tanh, output_activation=:identity, function_name=:NN, calculate_se=false, seed=1234)
    end
    @randomEffects begin
        eta = RandomEffect(Normal(0.0, 1.0); column=:id)
    end
    @formulas begin
        mean_func = NN([d, t, eta], nn_params)[1]
        C ~ Normal(mean_func, sigma)
    end
end
"""

# ----------------------------------------------------------------------------------
# Model 4 - orange (real growth curve, non-PK, non-ODE, closed-form logistic growth on
# the classic nlme Orange dataset). Algebraic formula: the age covariate is referenced
# by name (the reserved-`t` gotcha only applies to DE state access).
# ----------------------------------------------------------------------------------
ORANGE_MODEL = """
@Model begin
    @covariates begin
        age = Covariate()
    end
    @fixedEffects begin
        Asym  = RealNumber(200.0)
        xmid  = RealNumber(700.0)
        scal  = RealNumber(350.0)
        omega = RealNumber(50.0, scale=:log)
        sigma = RealNumber(10.0, scale=:log)
    end
    @randomEffects begin
        eta = RandomEffect(Normal(0.0, omega); column=:Tree)
    end
    @formulas begin
        circ = (Asym + eta) / (1.0 + exp((xmid - age) / scal))
        circumference ~ Normal(circ, sigma)
    end
end
"""

# --- warfarin simulation (data-source="simulated"), kept exactly as the baseline ---
TRUE_THETA = {
    "ka": 1.0, "cl": 0.13, "v": 8.0,
    "omega_ka": 0.4, "omega_cl": 0.3, "omega_v": 0.2, "sigma": 0.5,
}
N_SUBJECTS = 24
DOSE = 100.0
TIMES = (0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 36.0, 48.0, 72.0, 96.0, 120.0)
DEFAULT_SEED = 20260818
DEFAULT_SOURCE = "warfarin"
DEFAULT_MODEL = "warfarin"

# The GPL-3 nlmixr2data warfarin frame committed verbatim to data/warfarin.csv (same
# O'Reilly PK/PD study; see data/README.md). Long format: id,time,amt,dv,dvid(cp/pca),
# evid,wt,age,sex. The demo's models are PK only, so the loaders keep the concentration
# (dvid=="cp") observations and carry each subject's dose as a constant covariate.
WARFARIN_CACHE = DATA / "warfarin.csv"

# Defined once per Julia session; the client only ever calls `nlf_objgrad`.
JULIA_HELPERS = """
(isdefined(NoLimits, :objective_and_gradient) && isdefined(NoLimits, :build_fit_context)) ||
    error("this NoLimits build has no objective_and_gradient/build_fit_context; they " *
    "shipped in v0.2.6 - point PYTHON_JULIAPKG_PROJECT at a Julia project with that " *
    "release or newer (see the README dev section)")

const NLF_CTX = IdDict()
nlf_ctx(dm) = get!(() -> NoLimits.build_fit_context(dm), NLF_CTX, dm)

# Flat parameter labels (sigma, nn_params[1], ...) so names align 1:1 with theta coords
# and with the log mask for every model, block-vector parameters included.
nlf_names(dm) = string.(NoLimits.ComponentArrays.labels(NoLimits.get_params(dm, scale = :untransformed)))

nlf_theta0(dm) = Vector{Float64}(NoLimits.get_params(dm, scale = :transformed))
nlf_fit_theta(fit) = Vector{Float64}(NoLimits.get_params(fit, scale = :transformed))

# Per-coordinate log mask: inverse_transform(0) is 1.0 on a log-scaled coordinate
# (exp(0)) and 0.0 on an identity one, so the server needs no model-specific rule.
function nlf_logmask(dm)
    ax = NoLimits.ComponentArrays.getaxes(NoLimits.get_params(dm, scale = :transformed))
    n = length(NoLimits.get_params(dm, scale = :transformed))
    z = NoLimits.ComponentArrays.ComponentArray(zeros(n), ax)
    Vector{Float64}(dm.model.fixed.inverse_transform(z))
end

function nlf_natural(dm, v)
    ax = NoLimits.ComponentArrays.getaxes(NoLimits.get_params(dm, scale = :transformed))
    dm.model.fixed.inverse_transform(NoLimits.ComponentArrays.ComponentArray(collect(Float64, v), ax))
end
nlf_natural_vec(dm, v) = Vector{Float64}(nlf_natural(dm, v))

# Natural-scale vector -> transformed-scale wire vector (the model's own transform).
function nlf_transform(dm, v)
    ax = NoLimits.ComponentArrays.getaxes(NoLimits.get_params(dm, scale = :untransformed))
    ca = NoLimits.ComponentArrays.ComponentArray(collect(Float64, v), ax)
    Vector{Float64}(dm.model.fixed.transform(ca))
end

function nlf_objgrad(dm, v, method)
    val, grad = NoLimits.objective_and_gradient(
        method, nlf_ctx(dm), nlf_natural(dm, v), scale = "transformed")
    (Float64(val), Vector{Float64}(grad))
end
"""


def simulate(seed: int = DEFAULT_SEED, n_subjects: int = N_SUBJECTS) -> pd.DataFrame:
    """Seeded synthetic warfarin data at TRUE_THETA (closed-form ODE, no solver)."""
    rng = np.random.default_rng(seed)
    p = TRUE_THETA
    kai = p["ka"] * rng.lognormal(0.0, p["omega_ka"], size=n_subjects)
    cli = p["cl"] * rng.lognormal(0.0, p["omega_cl"], size=n_subjects)
    vi = p["v"] * rng.lognormal(0.0, p["omega_v"], size=n_subjects)
    ke = cli / vi
    ids = np.repeat(np.arange(n_subjects), len(TIMES))
    t = np.tile(np.asarray(TIMES, dtype=float), n_subjects)
    cp = (
        DOSE / vi[ids] * kai[ids] / (kai[ids] - ke[ids])
        * (np.exp(-ke[ids] * t) - np.exp(-kai[ids] * t))
    )
    conc = cp + rng.normal(0.0, p["sigma"], size=cp.size)
    return pd.DataFrame({
        "ID": [f"S{i:02d}" for i in ids], "t": t, "Dose": DOSE, "conc": conc,
    })


def _warfarin_raw(nl=None) -> pd.DataFrame:
    """PK frame from the committed nlmixr2data warfarin CSV, as id/t/d/C.

    nlmixr2data warfarin is long PK/PD: keep the concentration observations
    (dvid=="cp", evid==0) and carry each subject's single dose (amt on its evid==1 row)
    as a constant covariate on every row.
    """
    raw = pd.read_csv(WARFARIN_CACHE)
    dose = raw.loc[raw["evid"] == 1].set_index("id")["amt"]
    cp = raw[(raw["dvid"] == "cp") & (raw["evid"] == 0)]
    return pd.DataFrame({
        "id": cp["id"].astype(str).to_numpy(),
        "t": cp["time"].to_numpy(dtype=float),
        "d": dose.reindex(cp["id"]).to_numpy(dtype=float),
        "C": cp["dv"].to_numpy(dtype=float),
    })


def load_warfarin(source: str = DEFAULT_SOURCE, seed: int = DEFAULT_SEED, nl=None) -> pd.DataFrame:
    """Warfarin PK data as ID/t/Dose/conc, real (default) or the seeded simulation."""
    if source == "simulated":
        return simulate(seed=seed)
    if source != "warfarin":
        raise ValueError(f"unknown data-source {source!r} (expected 'warfarin' or 'simulated')")
    pk = _warfarin_raw(nl)
    return pd.DataFrame({
        "ID": pk["id"].to_numpy(),
        "t": pk["t"].to_numpy(dtype=float),
        "Dose": pk["d"].to_numpy(dtype=float),
        "conc": pk["C"].to_numpy(dtype=float),
    })


def load_warfarin_nn(source: str = DEFAULT_SOURCE, seed: int = DEFAULT_SEED, nl=None) -> pd.DataFrame:
    """The SAME real warfarin PK rows in the neural model's raw columns id/t/d/C."""
    return _warfarin_raw(nl)


def load_theoph(source: str = DEFAULT_SOURCE, seed: int = DEFAULT_SEED, nl=None) -> pd.DataFrame:
    """Real Theoph data (12 subjects) as id/t/Dose/conc, from the vendored data/theoph.csv."""
    raw = pd.read_csv(DATA / "theoph.csv")
    return pd.DataFrame({
        "id": raw["Subject"].astype(str).to_numpy(),
        "t": raw["Time"].to_numpy(dtype=float),
        "Dose": raw["Dose"].to_numpy(dtype=float),
        "conc": raw["conc"].to_numpy(dtype=float),
    })


def load_orange(source: str = DEFAULT_SOURCE, seed: int = DEFAULT_SEED, nl=None) -> pd.DataFrame:
    """Real nlme Orange data (5 trees) as Tree/age/circumference, from data/orange.csv."""
    raw = pd.read_csv(DATA / "orange.csv")
    return pd.DataFrame({
        "Tree": raw["Tree"].astype(str).to_numpy(),
        "age": raw["age"].to_numpy(dtype=float),
        "circumference": raw["circumference"].to_numpy(dtype=float),
    })


@dataclass(frozen=True)
class ModelSpec:
    model: str                 # the NoLimits @Model string
    loader: Callable           # (source, seed, nl) -> DataFrame in the model's columns
    primary_id: str            # subject-id column (partitioning + DataModel grouping)
    time_col: str              # time / independent-variable column
    columns: dict              # source -> model column map, for the README/tests
    num_sites: int             # site partition
    estimator: str = "laplace"
    acceptance: str = "strict"  # "strict" (obj + params) | "nn" (additivity gate only)
    param_tol: float = 1.0e-3   # worst natural-scale parameter tolerance for "strict"
    pooled_init: bool = False   # pass pooled_init=true to the pooled reference fit
    fit_seed: int = 0           # Random.seed!(fit_seed) before the pooled fit, 0 = none


CATALOG = {
    "warfarin": ModelSpec(
        model=WARFARIN_MODEL, loader=load_warfarin, primary_id="ID", time_col="t",
        columns={"id": "ID", "time": "t", "amt": "Dose", "dv": "conc"}, num_sites=3,
    ),
    "theophylline": ModelSpec(
        model=THEOPH_MODEL, loader=load_theoph, primary_id="id", time_col="t",
        columns={"Subject": "id", "Time": "t", "Dose": "Dose", "conc": "conc"},
        num_sites=3, pooled_init=True,
    ),
    "warfarin-nn": ModelSpec(
        model=WARFARIN_NN_MODEL, loader=load_warfarin_nn, primary_id="id", time_col="t",
        columns={"id": "id", "time": "t", "amt": "d", "dv": "C"}, num_sites=3,
        acceptance="nn", pooled_init=True, fit_seed=1234,
    ),
    "orange": ModelSpec(
        model=ORANGE_MODEL, loader=load_orange, primary_id="Tree", time_col="age",
        columns={"Tree": "Tree", "age": "age", "circumference": "circumference"},
        num_sites=3, pooled_init=True, param_tol=1.0e-3,
    ),
}


def spec(model: str = DEFAULT_MODEL) -> ModelSpec:
    if model not in CATALOG:
        raise ValueError(f"unknown model {model!r} (expected one of {tuple(CATALOG)})")
    return CATALOG[model]


def dataset(model: str = DEFAULT_MODEL, source: str = DEFAULT_SOURCE,
            seed: int = DEFAULT_SEED, nl=None) -> pd.DataFrame:
    """The data to federate for `model` (data-source only matters for warfarin)."""
    return spec(model).loader(source, seed, nl)


def partition(df: pd.DataFrame, num_sites: int, primary_id: str = "ID") -> list[pd.DataFrame]:
    """Split into contiguous blocks of whole subjects (disjoint subjects per site)."""
    subjects = list(dict.fromkeys(df[primary_id]))
    blocks = np.array_split(np.asarray(subjects, dtype=object), num_sites)
    if any(len(b) == 0 for b in blocks):
        raise ValueError(f"{len(subjects)} subjects cannot fill {num_sites} sites")
    return [df[df[primary_id].isin(set(b))].reset_index(drop=True) for b in blocks]


_helpers_loaded = False


def build_data_model(nl, model: str, df: pd.DataFrame):
    """Build the DataModel for `model` (minutes of Julia codegen on first call per model)."""
    global _helpers_loaded
    if not _helpers_loaded:
        nl.seval(JULIA_HELPERS)
        _helpers_loaded = True
    sp = spec(model)
    return nl.DataModel(nl.model(sp.model), df, primary_id=sp.primary_id, time_col=sp.time_col)


def true_theta_transformed(nl, dm) -> np.ndarray:
    """TRUE_THETA (warfarin) as a transformed-scale wire vector, in the model's axes order."""
    names = [str(s) for s in nl.seval("nlf_names")(dm)]
    if set(names) != set(TRUE_THETA):
        raise ValueError(f"model parameters {names} do not match TRUE_THETA")
    natural = np.array([TRUE_THETA[n] for n in names], dtype=float)
    return np.asarray(nl.seval("nlf_transform")(dm, natural), dtype=float)


def to_natural(theta_transformed, log_mask) -> np.ndarray:
    """Transformed -> natural: exp on log-scaled coordinates, identity elsewhere.

    `log_mask` is the per-coordinate mask the sites report (Julia's own
    inverse_transform(0)), so the server needs no model-specific rule and no Julia.
    """
    theta = np.asarray(theta_transformed, dtype=float)
    mask = np.asarray(log_mask, dtype=bool)
    return np.where(mask, np.exp(theta), theta)


def precondition_scale(theta0_transformed, log_mask) -> np.ndarray:
    """Diagonal preconditioning scale, NoLimits' own rule (`_precondition_scale`).

    s_i = 1 on a log-scaled coordinate, else max(|theta0_i|, 1). Reimplemented rather than
    called because it runs on the server, which has no Julia.
    """
    theta0 = np.asarray(theta0_transformed, dtype=float)
    mask = np.asarray(log_mask, dtype=bool)
    return np.where(mask, 1.0, np.maximum(np.abs(theta0), 1.0))


def _check_theta(model: str, source: str) -> str:
    """'true' for the warfarin simulation (known theta), 'theta0' otherwise."""
    return "true" if (model == "warfarin" and source == "simulated") else "theta0"


def _theta_for(nl, model, dm, source):
    if _check_theta(model, source) == "true":
        return true_theta_transformed(nl, dm)
    return np.asarray(nl.seval("nlf_theta0")(dm), dtype=float)


ESTIMATORS = ("laplace", "focei", "ghq", "pooled")


def _method(nl, estimator: str, ghq_level: int):
    if estimator == "laplace":
        return nl.Laplace()
    if estimator == "focei":
        return nl.FOCEI()
    if estimator == "ghq":
        return nl.GHQuadrature(level=ghq_level)
    if estimator == "pooled":
        return nl.Pooled()
    raise ValueError(f"unknown estimator {estimator!r} (expected one of {ESTIMATORS})")


def objective_and_gradient(nl, dm, theta_transformed, estimator: str, ghq_level: int):
    """(value, gradient-on-transformed-axes) for one DataModel at a wire vector."""
    method = _method(nl, estimator, ghq_level)
    value, grad = nl.seval("nlf_objgrad")(dm, np.asarray(theta_transformed, dtype=float), method)
    value = float(value)
    grad = np.asarray(grad, dtype=float)
    if not np.isfinite(value) or not np.all(np.isfinite(grad)):
        raise RuntimeError(f"non-finite objective/gradient at theta={theta_transformed}")
    return value, grad


def pooled_reference(model: str = DEFAULT_MODEL, estimator: str = "laplace", ghq_level: int = 5,
                     seed: int = DEFAULT_SEED, source: str = DEFAULT_SOURCE):
    """(theta, value, gradient) at the additivity-check theta, unpartitioned.

    Sum over sites of the same call must equal this (the additivity property).
    """
    import NoLimitsPy as nl
    dm = build_data_model(nl, model, dataset(model, source, seed, nl))
    theta = _theta_for(nl, model, dm, source)
    value, gradient = objective_and_gradient(nl, dm, theta, estimator, ghq_level)
    return {"theta": theta.tolist(), "value": value, "gradient": gradient.tolist()}


def pooled_fit(model: str = DEFAULT_MODEL, estimator: str = "laplace", ghq_level: int = 5,
               seed: int = DEFAULT_SEED, source: str = DEFAULT_SOURCE):
    """The pooled `fit_model` reference plus the shared start point, axes and log mask.

    Boots Julia in the calling process; run it where the main thread is free:
    `python -m nolimits_flower.task fit <model> <estimator> <ghq> <seed> <source>` prints it
    as JSON, which is how the ServerApp (a worker thread) gets it.
    """
    import NoLimitsPy as nl
    sp = spec(model)
    dm = build_data_model(nl, model, dataset(model, source, seed, nl))
    if sp.fit_seed:
        nl.seval("import Random")
        nl.seval("Random.seed!")(sp.fit_seed)
    method = _method(nl, estimator, ghq_level)
    fit = nl.fit_model(dm, method, pooled_init=True) if sp.pooled_init else nl.fit_model(dm, method)
    theta = np.asarray(nl.seval("nlf_fit_theta")(fit), dtype=float)
    value, _ = objective_and_gradient(nl, dm, theta, estimator, ghq_level)
    names = [str(s) for s in nl.seval("nlf_names")(dm)]
    log_mask = np.asarray(nl.seval("nlf_logmask")(dm), dtype=float)
    natural = np.asarray(nl.seval("nlf_natural_vec")(dm, theta), dtype=float)
    # The Julia-free to_natural must reproduce Julia's own inverse transform, or the
    # server would report the wrong numbers.
    if not np.allclose(natural, to_natural(theta, log_mask), rtol=1e-12, atol=0.0):
        raise RuntimeError(f"to_natural does not match the model transform for {model}")
    return {
        "names": names,
        "theta0": [float(v) for v in nl.seval("nlf_theta0")(dm)],
        "log_mask": log_mask.tolist(),
        "theta_transformed": theta.tolist(),
        "theta_natural": natural.tolist(),
        "value": value,
    }


def additivity_probe(model: str = DEFAULT_MODEL, num_sites: int = 0, ghq_level: int = 5,
                     seed: int = DEFAULT_SEED, source: str = DEFAULT_SOURCE):
    """Per-estimator additivity: sum over sites of (value, gradient) vs the pooled-data call.

    One Julia boot, evaluated at the additivity-check theta (identical for every DataModel).
    The neural model is checked for `laplace` only; the PK/growth models for all four.
    """
    import NoLimitsPy as nl
    sp = spec(model)
    if num_sites <= 0:
        num_sites = sp.num_sites
    df = dataset(model, source, seed, nl)
    pooled_dm = build_data_model(nl, model, df)
    site_dms = [build_data_model(nl, model, s) for s in partition(df, num_sites, sp.primary_id)]
    theta = _theta_for(nl, model, pooled_dm, source)
    estimators = ("laplace",) if sp.acceptance == "nn" else ESTIMATORS
    out = {}
    for estimator in estimators:
        pooled_value, pooled_grad = objective_and_gradient(nl, pooled_dm, theta, estimator, ghq_level)
        pairs = [objective_and_gradient(nl, dm, theta, estimator, ghq_level) for dm in site_dms]
        fed_value = sum(v for v, _ in pairs)
        fed_grad = np.sum([g for _, g in pairs], axis=0)
        out[estimator] = {
            "pooled_value": pooled_value,
            "federated_value": fed_value,
            "value_rel": abs(fed_value - pooled_value) / abs(pooled_value),
            "gradient_rel": float(np.linalg.norm(fed_grad - pooled_grad) / np.linalg.norm(pooled_grad)),
            "worst_gradient_coord_rel": float(
                np.max(np.abs(fed_grad - pooled_grad) / np.maximum(np.abs(pooled_grad), 1e-12))
            ),
        }
    return {"theta": theta.tolist(), "model": model, "sites": num_sites, "source": source, "probes": out}


if __name__ == "__main__":
    # python -m nolimits_flower.task {fit|ref|probe} <model> [estimator] [ghq] [seed] [source]
    # -> one "POOLED_JSON {...}" line on stdout.
    import json
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    rest = (
        sys.argv[3] if len(sys.argv) > 3 else "laplace",
        int(sys.argv[4]) if len(sys.argv) > 4 else 5,
        int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_SEED,
        sys.argv[6] if len(sys.argv) > 6 else DEFAULT_SOURCE,
    )
    if mode == "probe":
        result = additivity_probe(model, 0, *rest[1:])
    elif mode == "fit":
        result = pooled_fit(model, *rest)
    else:
        result = pooled_reference(model, *rest)
    print("POOLED_JSON " + json.dumps(result))
