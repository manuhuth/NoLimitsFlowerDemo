"""Model, synthetic data, site partitioning and theta glue. No Flower imports.

Theta on the wire is the TRANSFORMED (optimization) scale: all four fixed effects
are `scale=:log`, so the wire vector is unconstrained and the server optimizer in
Phase 3 keeps positivity implicitly.

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

import numpy as np
import pandas as pd

# The tiny quickstart model: exponential decay, one Normal RE on ID.
MODEL = """
@fixedEffects begin
    A0    = RealNumber(10.0, scale=:log)
    k     = RealNumber(0.3, scale=:log)
    omega = RealNumber(0.3, scale=:log)
    sigma = RealNumber(0.5, scale=:log)
end

@covariates begin
    time = Covariate()
end

@randomEffects begin
    eta = RandomEffect(Normal(0.0, omega); column=:ID)
end

@formulas begin
    pred = A0 * exp(eta) * exp(-k * time)
    y ~ Normal(pred, sigma)
end
"""

# Truth used by the simulation; also the theta the Phase-2 verification round uses.
TRUE_THETA = {"A0": 10.0, "k": 0.3, "omega": 0.3, "sigma": 0.5}
PARAM_NAMES = tuple(TRUE_THETA)

N_SUBJECTS = 24
TIMES = (0.5, 1.0, 2.0, 4.0)

# Defined once per Julia session; `nlf_objgrad` is the only thing the client calls.
JULIA_HELPERS = """
isdefined(NoLimits, :objective_and_gradient) || error(
    "this NoLimits build has no objective_and_gradient; point PYTHON_JULIAPKG_PROJECT " *
    "at a Julia project tracking NoLimits main (see the README dev section)")

const NLF_CACHE = IdDict()

function nlf_axes(dm)
    get!(NLF_CACHE, dm) do
        (NoLimits.ComponentArrays.getaxes(NoLimits.get_params(dm, scale = :transformed)),
         dm.model.fixed.inverse_transform)
    end
end

nlf_names(dm) = string.(keys(NoLimits.get_params(dm, scale = :untransformed)))

function nlf_natural(dm, v)
    ax, inv = nlf_axes(dm)
    inv(NoLimits.ComponentArrays.ComponentArray(collect(Float64, v), ax))
end

# (value, gradient-on-transformed-axes) at the transformed-scale wire vector.
function nlf_objgrad(dm, v, method)
    val, grad = NoLimits.objective_and_gradient(method, dm, nlf_natural(dm, v), scale = "transformed")
    (Float64(val), Vector{Float64}(grad))
end
"""


def simulate(seed: int = 20260818, n_subjects: int = N_SUBJECTS) -> pd.DataFrame:
    """Seeded synthetic data at TRUE_THETA: n_subjects x len(TIMES) observations."""
    rng = np.random.default_rng(seed)
    p = TRUE_THETA
    eta = rng.normal(0.0, p["omega"], size=n_subjects)
    ids = np.repeat(np.arange(n_subjects), len(TIMES))
    time = np.tile(np.asarray(TIMES, dtype=float), n_subjects)
    pred = p["A0"] * np.exp(eta)[ids] * np.exp(-p["k"] * time)
    y = pred + rng.normal(0.0, p["sigma"], size=pred.size)
    return pd.DataFrame({"ID": [f"S{i:02d}" for i in ids], "time": time, "y": y})


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
    return nl.DataModel(nl.model(MODEL), df, primary_id="ID", time_col="time")


def true_theta_transformed(nl, dm) -> np.ndarray:
    """TRUE_THETA as a transformed-scale wire vector, in the model's own axes order."""
    names = [str(s) for s in nl.seval("nlf_names")(dm)]
    if set(names) != set(TRUE_THETA):
        raise ValueError(f"model parameters {names} do not match TRUE_THETA")
    return np.log([TRUE_THETA[n] for n in names])  # every fixed effect is scale=:log


def pooled_reference(estimator: str = "laplace", ghq_level: int = 5, seed: int = 20260818):
    """(theta_transformed, value, gradient) for the UNPARTITIONED data set.

    Boots Julia in the calling process, so call it where the main thread is free.
    `python -m nolimits_flower.task <estimator> <ghq_level>` runs it as its own
    process and prints the result as JSON; that is how server_app gets it.
    """
    import NoLimitsPy as nl

    dm = build_data_model(nl, simulate(seed=seed))
    theta = true_theta_transformed(nl, dm)
    value, gradient = objective_and_gradient(nl, dm, theta, estimator, ghq_level)
    return theta, value, gradient


def objective_and_gradient(nl, dm, theta_transformed: np.ndarray, estimator: str, ghq_level: int):
    """(value, gradient) for one DataModel at a transformed-scale wire vector."""
    method = nl.Laplace() if estimator == "laplace" else nl.GHQuadrature(level=ghq_level)
    value, grad = nl.seval("nlf_objgrad")(dm, np.asarray(theta_transformed, dtype=float), method)
    value = float(value)
    grad = np.asarray(grad, dtype=float)
    if not np.isfinite(value) or not np.all(np.isfinite(grad)):
        raise RuntimeError(f"non-finite objective/gradient at theta={theta_transformed}")
    return value, grad


if __name__ == "__main__":  # `python -m nolimits_flower.task laplace 5` -> JSON on stdout
    import json
    import sys

    theta, value, gradient = pooled_reference(
        sys.argv[1] if len(sys.argv) > 1 else "laplace",
        int(sys.argv[2]) if len(sys.argv) > 2 else 5,
    )
    print("POOLED_JSON " + json.dumps(
        {"theta": theta.tolist(), "value": value, "gradient": gradient.tolist()}
    ))
