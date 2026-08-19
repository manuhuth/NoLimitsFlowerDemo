"""Model, synthetic data, site partitioning and theta glue. No Flower imports.

Theta on the wire is the TRANSFORMED (optimization) scale, which keeps the variance
parameters positive without server-side bounds. The model mixes scales (the structural
PK parameters are plain, the omegas and sigma are `scale=:log`), so `LOG_SCALED` is the
map the server uses to report natural-scale numbers without booting Julia; `pooled_fit`
checks it against Julia's own inverse transform on every run.

`objective_and_gradient(method, dm, theta; scale=...)` always wants theta on the
NATURAL scale and only uses `scale` to pick the coordinates of the returned
gradient (docstring: "Value AND theta-gradient ... at natural-scale theta. The
gradient is a ComponentArray on theta's axes (`scale = :untransformed`) or on the
transformed axes (`:transformed`)"). So the client rebuilds a transformed
ComponentArray from the wire vector using a cached axes template, maps it back
with `get_inverse_transform`, and asks for the gradient on transformed axes -
exactly the coordinates the server optimizes in. All of that lives in one Julia
helper defined once per process (`JULIA_HELPERS`).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Warfarin population PK: one-compartment oral absorption (depot -> central) with
# multiplicative log-normal random effects on ka, cl and v. The ODE is linear, so
# NoLimits takes its closed-form fast path instead of a numerical solver.
MODEL = """
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

# Truth used by the simulation; also the model's own initial values (see MODEL) and the
# theta the additivity check uses. Warfarin-typical: 100 mg oral dose, conc in mg/L.
TRUE_THETA = {
    "ka": 1.0, "cl": 0.13, "v": 8.0,
    "omega_ka": 0.4, "omega_cl": 0.3, "omega_v": 0.2, "sigma": 0.5,
}
PARAM_NAMES = tuple(TRUE_THETA)
# The fixed effects declared `scale=:log` in MODEL; everything else is plain.
LOG_SCALED = frozenset({"omega_ka", "omega_cl", "omega_v", "sigma"})

N_SUBJECTS = 24
DOSE = 100.0
TIMES = (0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 36.0, 48.0, 72.0, 96.0, 120.0)
DEFAULT_SEED = 20260818

DEFAULT_SOURCE = "warfarin"
# Raw Monolix warfarin frame, written on the first download and read from then on, so
# repeat runs and the tests need no network. Gitignored: data is not package content.
WARFARIN_CACHE = Path(__file__).resolve().parents[2] / "data" / "warfarin.csv"

# Defined once per Julia session; `nlf_objgrad` is the only thing the client calls.
JULIA_HELPERS = """
(isdefined(NoLimits, :objective_and_gradient) && isdefined(NoLimits, :build_fit_context)) ||
    error("this NoLimits build has no objective_and_gradient/build_fit_context; they " *
    "shipped in v0.2.6 - point PYTHON_JULIAPKG_PROJECT at a Julia project with that " *
    "release or newer (see the README dev section)")

const NLF_CACHE = IdDict()
const NLF_CTX = IdDict()

# One FitContext per site DataModel, built in the prepare round. It carries the batch
# infos, the constants cache and the evaluation cache, so a round no longer rebuilds
# them. The `dm` form of objective_and_gradient remains valid and gives the same
# numbers; it just redoes that setup on every call.
nlf_ctx(dm) = get!(() -> NoLimits.build_fit_context(dm), NLF_CTX, dm)

function nlf_axes(dm)
    get!(NLF_CACHE, dm) do
        (NoLimits.ComponentArrays.getaxes(NoLimits.get_params(dm, scale = :transformed)),
         dm.model.fixed.inverse_transform)
    end
end

nlf_names(dm) = string.(keys(NoLimits.get_params(dm, scale = :untransformed)))

# The model's own default theta, already on the transformed scale: the federated
# optimizer's start point and (by construction) fit_model's own start point.
nlf_theta0(dm) = Vector{Float64}(NoLimits.get_params(dm, scale = :transformed))

# A fit's theta* as a plain vector (ComponentArrays do not cross the wrappers).
nlf_fit_theta(fit) = Vector{Float64}(NoLimits.get_params(fit, scale = :transformed))

function nlf_natural(dm, v)
    ax, inv = nlf_axes(dm)
    inv(NoLimits.ComponentArrays.ComponentArray(collect(Float64, v), ax))
end

nlf_natural_vec(dm, v) = Vector{Float64}(nlf_natural(dm, v))

# Natural-scale vector -> transformed-scale wire vector (the model's own transform, so
# mixed scales need no bookkeeping on the Python side).
function nlf_transform(dm, v)
    ax = NoLimits.ComponentArrays.getaxes(NoLimits.get_params(dm, scale = :untransformed))
    ca = NoLimits.ComponentArrays.ComponentArray(collect(Float64, v), ax)
    Vector{Float64}(dm.model.fixed.transform(ca))
end

# (value, gradient-on-transformed-axes) at the transformed-scale wire vector.
function nlf_objgrad(dm, v, method)
    val, grad = NoLimits.objective_and_gradient(
        method, nlf_ctx(dm), nlf_natural(dm, v), scale = "transformed")
    (Float64(val), Vector{Float64}(grad))
end
"""

# Downloads the Monolix warfarin data through NoLimits' own loader and writes the raw
# frame to the cache. Only ever runs when the cache is missing.
WARFARIN_JULIA = """
import CSV
function nlf_warfarin_cache(path)
    mkpath(dirname(path))
    CSV.write(path, NoLimits.load_warfarin_from_monolix())
    return path
end
"""


def simulate(seed: int = DEFAULT_SEED, n_subjects: int = N_SUBJECTS) -> pd.DataFrame:
    """Seeded synthetic data at TRUE_THETA: n_subjects x len(TIMES) concentrations.

    The concentration is the closed-form solution of the model's own linear ODE
    (single bolus into depot at t=0), so the simulation needs no solver.
    """
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


def warfarin(nl=None) -> pd.DataFrame:
    """The Monolix warfarin PK data in this model's columns (ID, t, Dose, conc).

    NoLimits' own `load_warfarin_from_monolix()` downloads the joint PK/PD frame (32
    dosed subjects, 30 of which have the baseline INR record the loader requires, hence
    30 in the returned frame). The PK rows are the ones with a non-missing `C`; `d` is
    the per-subject dose the loader already carried forward to every row, which is what
    the model's `ConstantCovariate(constant_on=:ID)` wants.

    The download happens once: the raw frame is cached at `data/warfarin.csv` and every
    later call (and every test) reads the cache, so repeat runs are offline.
    """
    if not WARFARIN_CACHE.exists():
        if nl is None:
            import NoLimitsPy as nl
        nl.seval(WARFARIN_JULIA)
        nl.seval("nlf_warfarin_cache")(str(WARFARIN_CACHE))
    raw = pd.read_csv(WARFARIN_CACHE)
    pk = raw[raw["C"].notna()]
    return pd.DataFrame({
        "ID": pk["id"].astype(str).to_numpy(),
        "t": pk["t"].to_numpy(dtype=float),
        "Dose": pk["d"].to_numpy(dtype=float),
        "conc": pk["C"].to_numpy(dtype=float),
    })


def dataset(source: str = DEFAULT_SOURCE, seed: int = DEFAULT_SEED, nl=None) -> pd.DataFrame:
    """The data to federate: the real warfarin PK data, or the seeded simulation."""
    if source == "warfarin":
        return warfarin(nl)
    if source == "simulated":
        return simulate(seed=seed)
    raise ValueError(f"unknown data-source {source!r} (expected 'warfarin' or 'simulated')")


def partition(df: pd.DataFrame, num_sites: int) -> list[pd.DataFrame]:
    """Split into contiguous blocks of whole subjects (disjoint subjects per site)."""
    subjects = list(dict.fromkeys(df["ID"]))
    blocks = np.array_split(np.asarray(subjects, dtype=object), num_sites)
    if any(len(b) == 0 for b in blocks):
        raise ValueError(f"{len(subjects)} subjects cannot fill {num_sites} sites")
    return [df[df["ID"].isin(set(b))].reset_index(drop=True) for b in blocks]


_helpers_loaded = False


def build_data_model(nl, df: pd.DataFrame):
    """Build the DataModel (minutes of Julia codegen on first call per process)."""
    global _helpers_loaded
    if not _helpers_loaded:
        nl.seval(JULIA_HELPERS)
        _helpers_loaded = True
    return nl.DataModel(nl.model(MODEL), df, primary_id="ID", time_col="t")


def true_theta_transformed(nl, dm) -> np.ndarray:
    """TRUE_THETA as a transformed-scale wire vector, in the model's own axes order."""
    names = [str(s) for s in nl.seval("nlf_names")(dm)]
    if set(names) != set(TRUE_THETA):
        raise ValueError(f"model parameters {names} do not match TRUE_THETA")
    natural = np.array([TRUE_THETA[n] for n in names], dtype=float)
    return np.asarray(nl.seval("nlf_transform")(dm, natural), dtype=float)


def to_natural(theta_transformed, names) -> np.ndarray:
    """Transformed -> natural scale: exp for the `scale=:log` parameters, else identity.

    The Julia-free twin of the model's inverse transform, so the ServerApp (a worker
    thread that can never boot Julia) can report natural-scale numbers. `pooled_fit`
    asserts it against Julia's own inverse transform.
    """
    theta = np.asarray(theta_transformed, dtype=float)
    return np.where([n in LOG_SCALED for n in names], np.exp(theta), theta)


def precondition_scale(theta0_transformed, names) -> np.ndarray:
    """Diagonal preconditioning scale for the server optimizer, NoLimits' own rule.

    Mirrors `_precondition_scale` / `_precondition_maps` in NoLimits.jl
    src/estimation/common.jl (nlmixr2's scaleC): the optimizer works in z with
    theta_t = theta0_t + s .* z, and s_i = max(|theta0_t_i|, 1) for a coordinate that is
    on the identity scale, else 1. NoLimits also treats a parameter that the model uses
    inside an `exp` as log-scaled; MODEL here has no `exp`, so LOG_SCALED is the whole
    rule. Reimplemented rather than called: this runs on the server, which has no Julia.
    """
    theta0 = np.asarray(theta0_transformed, dtype=float)
    return np.where([n in LOG_SCALED for n in names], 1.0, np.maximum(np.abs(theta0), 1.0))


def pooled_reference(estimator: str = "laplace", ghq_level: int = 5, seed: int = DEFAULT_SEED,
                     source: str = DEFAULT_SOURCE):
    """(theta_transformed, value, gradient) at the additivity-check theta, unpartitioned.

    Sum over sites of the same call must equal this. The check theta is TRUE_THETA for
    the simulation and the model's own default theta0 for the real warfarin data, where
    no true theta exists.
    """
    import NoLimitsPy as nl

    dm = build_data_model(nl, dataset(source, seed, nl))
    theta = (
        true_theta_transformed(nl, dm) if source == "simulated"
        else np.asarray(nl.seval("nlf_theta0")(dm), dtype=float)
    )
    value, gradient = objective_and_gradient(nl, dm, theta, estimator, ghq_level)
    return {"theta": theta.tolist(), "value": value, "gradient": gradient.tolist()}


def pooled_fit(estimator: str = "laplace", ghq_level: int = 5, seed: int = DEFAULT_SEED,
               source: str = DEFAULT_SOURCE):
    """The pooled `fit_model` reference plus the shared start point and axes order.

    Boots Julia in the calling process, so run it where the main thread is free:
    `python -m nolimits_flower.task fit <estimator> <ghq_level> <seed> <source>` prints this
    as JSON, which is how the ServerApp (a worker thread) gets it.
    """
    import NoLimitsPy as nl

    dm = build_data_model(nl, dataset(source, seed, nl))
    fit = nl.fit_model(dm, _method(nl, estimator, ghq_level))
    theta = np.asarray(nl.seval("nlf_fit_theta")(fit), dtype=float)
    # Re-evaluate the objective at theta* through the same primitive the sites use,
    # so the federated/pooled comparison cannot differ by bookkeeping.
    value, _ = objective_and_gradient(nl, dm, theta, estimator, ghq_level)
    names = [str(s) for s in nl.seval("nlf_names")(dm)]
    natural = np.asarray(nl.seval("nlf_natural_vec")(dm, theta), dtype=float)
    # LOG_SCALED must agree with the model string, or the server would report and
    # compare the wrong numbers.
    if not np.allclose(natural, to_natural(theta, names), rtol=1e-12, atol=0.0):
        raise RuntimeError(f"LOG_SCALED does not match the model transform for {names}")
    return {
        "names": names,
        "theta0": [float(v) for v in nl.seval("nlf_theta0")(dm)],
        "theta_transformed": theta.tolist(),
        "theta_natural": natural.tolist(),
        "value": value,
    }


ESTIMATORS = ("laplace", "focei", "ghq", "pooled")


def additivity_probe(num_sites: int = 3, ghq_level: int = 5, seed: int = DEFAULT_SEED,
                     source: str = DEFAULT_SOURCE):
    """Per-estimator additivity: sum over sites of (value, gradient) vs the pooled-data call.

    One Julia boot, one model codegen, 1 + num_sites DataModels, evaluated at the model's
    own default theta0 (identical for every DataModel, since it comes from MODEL). All four
    estimators are per-subject sums on this model and come out additive to ~1e-16; `pooled`
    is the one whose exactness is model-conditional (see `server_app.POOLED_CAVEAT`), which
    is why this probe measures rather than assumes.
    """
    import NoLimitsPy as nl

    df = dataset(source, seed, nl)
    pooled_dm = build_data_model(nl, df)
    site_dms = [build_data_model(nl, s) for s in partition(df, num_sites)]
    theta = np.asarray(nl.seval("nlf_theta0")(pooled_dm), dtype=float)
    out = {}
    for estimator in ESTIMATORS:
        pooled_value, pooled_grad = objective_and_gradient(
            nl, pooled_dm, theta, estimator, ghq_level)
        pairs = [objective_and_gradient(nl, dm, theta, estimator, ghq_level) for dm in site_dms]
        fed_value = sum(v for v, _ in pairs)
        fed_grad = np.sum([g for _, g in pairs], axis=0)
        out[estimator] = {
            "pooled_value": pooled_value,
            "federated_value": fed_value,
            "value_rel": abs(fed_value - pooled_value) / abs(pooled_value),
            "gradient_rel": float(
                np.linalg.norm(fed_grad - pooled_grad) / np.linalg.norm(pooled_grad)
            ),
            "worst_gradient_coord_rel": float(
                np.max(np.abs(fed_grad - pooled_grad) / np.maximum(np.abs(pooled_grad), 1e-12))
            ),
        }
    return {"theta": theta.tolist(), "sites": num_sites, "source": source, "probes": out}


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


def objective_and_gradient(nl, dm, theta_transformed: np.ndarray, estimator: str, ghq_level: int):
    """(value, gradient) for one DataModel at a transformed-scale wire vector."""
    method = _method(nl, estimator, ghq_level)
    value, grad = nl.seval("nlf_objgrad")(dm, np.asarray(theta_transformed, dtype=float), method)
    value = float(value)
    grad = np.asarray(grad, dtype=float)
    if not np.isfinite(value) or not np.all(np.isfinite(grad)):
        raise RuntimeError(f"non-finite objective/gradient at theta={theta_transformed}")
    return value, grad


if __name__ == "__main__":
    # `python -m nolimits_flower.task {fit|objgrad|probe} [estimator] [ghq_level] [seed] [source]`
    # -> one "POOLED_JSON {...}" line on stdout. `probe` reads the estimator slot as the
    # site count (it covers all estimators in one boot).
    import json
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    args = (
        sys.argv[2] if len(sys.argv) > 2 else "laplace",
        int(sys.argv[3]) if len(sys.argv) > 3 else 5,
        int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_SEED,
        sys.argv[5] if len(sys.argv) > 5 else DEFAULT_SOURCE,
    )
    if mode == "probe":
        result = additivity_probe(int(args[0]) if args[0].isdigit() else 3, *args[1:])
    else:
        result = pooled_fit(*args) if mode == "fit" else pooled_reference(*args)
    print("POOLED_JSON " + json.dumps(result))
