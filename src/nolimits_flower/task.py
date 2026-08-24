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

import math
import re
import secrets
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
# Model 5 - theophylline naive-pooled (NO random effects). The SAME 1-compartment oral
# PK on the SAME real theoph data, but population fixed effects only, so it is the model
# class the fixed-effects estimators MLE and MAP require. Weakly-informative LogNormal
# priors on every fixed effect (evaluated on the natural scale) let the SAME model serve
# both: MLE ignores the priors, MAP uses them. log-scaled coordinates keep ka/cl/v/sigma
# positive without random effects to do it.
# ----------------------------------------------------------------------------------
THEOPH_POOLED_MODEL = """
@fixedEffects begin
    ka    = RealNumber(1.5, scale=:log, prior=LogNormal(log(1.5), 1.0))
    cl    = RealNumber(0.04, scale=:log, prior=LogNormal(log(0.04), 1.0))
    v     = RealNumber(0.5, scale=:log, prior=LogNormal(log(0.5), 1.0))
    sigma = RealNumber(0.7, scale=:log, prior=LogNormal(log(0.7), 1.0))
end

@covariates begin
    t    = Covariate()
    Dose = ConstantCovariate(constant_on=:id)
end

@preDifferentialEquation begin
    ke = cl / v
end

@DifferentialEquation begin
    D(depot)   ~ -ka * depot
    D(central) ~ ka * depot - ke * central
end

@initialDE begin
    depot   = Dose
    central = 0.0
end

@formulas begin
    cp = central(t) / v
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

# --- differential privacy: the per-batch pairs the clipping needs --------------------
#
# A random-effect BATCH is NoLimits' independence unit (`build_re_batch_infos`): every
# individual sharing a random-effect level lands in the same batch. With one ID-grouped
# random effect - this demo's model class - a batch IS one subject and the returned
# `maxids` (largest individuals in any batch) is 1. Laplace/FOCEI need the empirical-Bayes
# mode once for all batches; GHQ integrates per batch and needs none; Pooled has no batch
# form (rejected before reaching here). Gradients are on the transformed axes: the caller
# scales each row by the preconditioning s and clips in that (optimizer) coordinate.
nlf_bstars(method::Union{NoLimits.Laplace, NoLimits.FOCEI}, ctx, theta) =
    NoLimits.empirical_bayes(ctx, theta)
nlf_bstars(method, ctx, theta) = nothing

nlf_batch_og(method::Union{NoLimits.Laplace, NoLimits.FOCEI}, ctx, theta, bi, bstars) =
    NoLimits.objective_and_gradient(
        method, ctx.dm, theta, NoLimits.get_batch_infos(ctx)[bi], bstars[bi];
        const_cache = ctx.const_cache, cache = ctx.cache, scale = "transformed")

nlf_batch_og(method::NoLimits.GHQuadrature, ctx, theta, bi, bstars) =
    NoLimits.objective_and_gradient(
        method, ctx.dm, theta, NoLimits.get_batch_infos(ctx)[bi];
        const_cache = ctx.const_cache, cache = ctx.cache, scale = "transformed")

# (per-batch values, per-batch transformed-axes gradients as rows, largest batch size).
# Summing the rows reproduces nlf_objgrad's gradient exactly (the transform is linear in
# the gradient, so it commutes with the sum over batches).
function nlf_dp_batches(dm, v, method)
    theta = nlf_natural(dm, v)
    ctx = nlf_ctx(dm)
    infos = NoLimits.get_batch_infos(ctx)
    bstars = nlf_bstars(method, ctx, theta)
    p = length(NoLimits.get_params(dm, scale = :transformed))
    grads = Matrix{Float64}(undef, length(infos), p)
    vals = Vector{Float64}(undef, length(infos))
    maxids = 0
    for bi in eachindex(infos)
        val, g = nlf_batch_og(method, ctx, theta, bi, bstars)
        grads[bi, :] .= Vector{Float64}(g)
        vals[bi] = Float64(val)
        maxids = max(maxids, length(infos[bi].inds))
    end
    (vals, grads, maxids)
end

# --- MLE / MAP (no random effects): per-INDIVIDUAL clipping unit ----------------------
#
# A no-RE model has no random-effect batch, so the DP clipping unit is the individual and
# the per-subject term is the conditional log-likelihood (`objective_and_gradient(MLE(),
# ctx, θ, idx)`). Summing the rows reproduces the population MLE gradient exactly. MAP's
# extra term is the PUBLIC prior, added as an offset by the carrier (see map_prior); it
# never enters this per-subject material, so MAP and MLE share this data path (maxids == 1).
function nlf_mle_individuals(dm, v)
    theta = nlf_natural(dm, v)
    ctx = nlf_ctx(dm)
    n = length(NoLimits.get_individuals(dm))
    p = length(NoLimits.get_params(dm, scale = :transformed))
    grads = Matrix{Float64}(undef, n, p)
    vals = Vector{Float64}(undef, n)
    for i in 1:n
        val, g = NoLimits.objective_and_gradient(
            NoLimits.MLE(), ctx, theta, i; scale = "transformed")
        grads[i, :] .= Vector{Float64}(g)
        vals[i] = Float64(val)
    end
    (vals, grads, 1)
end

# --- MCEM: nested federated EM (local E-step, federated M-step) -----------------------
#
# The M-step Q(θ) = Σ_batch (1/M) Σ_m log f(y_b, η_b^m | θ) at FIXED posterior draws is a
# per-subject sum, exactly like the single-shot objectives, so its value AND gradient are
# federated-summable and per-subject-clippable. The E-step (`mcem_e_step`) is LOCAL: each
# site samples its own subjects' posteriors and keeps the draws + warm-start state across the
# M-step rounds of one outer iteration. `mcem_q_partition` splits the free fixed effects into
# q1 (observation-side, needs the ODE) and q2 (RE-distribution only); the server optimizes
# each with its own L-BFGS-B, summing the sites' per-part (value, gradient).
import Random

# q1/q2 names as Strings, in the model's parameter order (== nlf_names order for the demo's
# all-scalar fixed effects), so the server can map each to a coordinate of the wire vector.
function nlf_mcem_parts(dm)
    p = NoLimits.mcem_q_partition(dm)
    (q1 = String.(p.q1), q2 = String.(p.q2))
end

# One LOCAL E-step at the wire θ. `state === nothing` on outer iter 1 (prior-mean seeding);
# thread `new_state` forward so warm-start + per-batch RNGs persist. The rng is seeded per
# (site, run) so the fit is reproducible; draws are held FIXED for this iteration's M-step.
function nlf_mcem_estep(dm, v, sample_schedule, maxiters, seed, state)
    theta = nlf_natural(dm, v)
    method = NoLimits.MCEM(sample_schedule = Int(sample_schedule), maxiters = Int(maxiters))
    rng = Random.Xoshiro(UInt64(seed))
    draws, new_state = NoLimits.mcem_e_step(dm, theta, method, state; rng = rng)
    (draws, new_state)
end

# M-step Q value + transformed-axes gradient over the `part`'s free names at FIXED `draws`.
# The gradient is on the `free_names` axes (frozen complement), so the server optimizes just
# that sub-vector. Summing over sites reproduces the pooled Q (per-subject additive).
function nlf_mcem_q(dm, v, draws, part, free_names)
    theta = nlf_natural(dm, v)
    Q, g = NoLimits.mcem_q_objective_and_gradient(
        dm, theta, draws; part = Symbol(part),
        free_names = Symbol.(collect(free_names)), scale = "transformed")
    (Float64(Q), Vector{Float64}(g))
end

# Per-subject (batch idx) form for the additivity proof and DP clipping: summing over idx
# equals the population `nlf_mcem_q` to machine precision.
function nlf_mcem_q_idx(dm, v, draws, idx, part, free_names)
    theta = nlf_natural(dm, v)
    Q, g = NoLimits.mcem_q_objective_and_gradient(
        dm, theta, draws, Int(idx); part = Symbol(part),
        free_names = Symbol.(collect(free_names)), scale = "transformed")
    (Float64(Q), Vector{Float64}(g))
end

# Per-subject rows for one M-step part (DP clipping unit == subject): (per-subject values,
# per-subject transformed-axes gradients over the part's free axes). Summing the rows == the
# population part; the caller clips + noises before any release.
function nlf_mcem_dp_part(dm, v, draws, part, free_names)
    theta = nlf_natural(dm, v)
    fnames = Symbol.(collect(free_names))
    n = length(draws)
    p = length(fnames)
    grads = Matrix{Float64}(undef, n, p)
    vals = Vector{Float64}(undef, n)
    for i in 1:n
        Q, g = NoLimits.mcem_q_objective_and_gradient(
            dm, theta, draws, i; part = Symbol(part), free_names = fnames, scale = "transformed")
        grads[i, :] .= Vector{Float64}(g)
        vals[i] = Float64(Q)
    end
    (vals, grads)
end

# --- SAEM: nested federated EM (local E-step, federated closed-form + numerical M-step) -----
#
# SAEM reuses the MCEM E-step (`nlf_mcem_estep`) and the MCEM Q kernels (`nlf_mcem_q`) for the
# NUMERICAL M-step. What is new is the CLOSED-FORM half: per iteration each site emits
# per-subject-additive SAEM sufficient statistics (`saem_sufficient_statistics`), the server
# SUMS them, and a COORDINATOR site runs the stateful closed-form update
# (`saem_closed_form_mstep`, bit-identical to the fit). Sites emit the statistics DE-NORMALIZED
# (RE moments as Σx=mean*n and Σxx'=second*n; outcome/HMM fields are already plain sums) so the
# server aggregates with a single element-wise numpy sum, exactly like the MCEM gradient sum.

# closed-form-eligible vs numerical free names PLUS the mcem q1/q2 split (so the driver knows
# which Q part each numerical name lives in). All Strings, in the model's parameter order.
function nlf_saem_parts(dm)
    e = NoLimits.saem_closed_form_eligibility(dm)
    p = NoLimits.mcem_q_partition(dm)
    (closed_form = String.(e.closed_form), numerical = String.(e.numerical),
        q1 = String.(p.q1), q2 = String.(p.q2))
end

_nlf_flat_push!(out, x::Number) = push!(out, Float64(x))
_nlf_flat_push!(out, x::AbstractArray) = append!(out, Float64.(vec(x)))

# Flatten the (re, outcome, hmm) stats to a Float64 vector of DE-NORMALIZED additive
# quantities, in the deterministic key order. re: [Σx (d), vec(Σxx') (d*d), n]; outcome:
# [s1, s2, ss, n]; hmm: [sum_w..., sum_wy...]. Identical layout on every site (same model),
# so the server sums coordinate-wise; the coordinator re-normalizes on the way back in.
function _nlf_saem_flatten(stats)
    out = Float64[]
    for re in keys(stats.re)
        s = getfield(stats.re, re)
        _nlf_flat_push!(out, s.mean .* s.n)
        _nlf_flat_push!(out, s.second .* s.n)
        _nlf_flat_push!(out, Float64(s.n))
    end
    for col in keys(stats.outcome)
        s = getfield(stats.outcome, col)
        _nlf_flat_push!(out, s.s1)
        _nlf_flat_push!(out, s.s2)
        _nlf_flat_push!(out, s.ss)
        _nlf_flat_push!(out, Float64(s.n))
    end
    for col in keys(stats.hmm)
        s = getfield(stats.hmm, col)
        _nlf_flat_push!(out, s.sum_w)
        _nlf_flat_push!(out, s.sum_wy)
    end
    out
end

# Rebuild aggregated stats from the server's summed flat vector, re-normalizing the RE moments
# (mean = Σx/n, second = Σxx'/n). `template` gives the structure (families/keys/dims), which is
# model-fixed and identical on every site.
function _nlf_saem_unflatten(template, flat)
    i = 0
    re_pairs = Pair{Symbol, Any}[]
    for re in keys(template.re)
        s = getfield(template.re, re)
        d = length(s.mean)
        sx = flat[(i + 1):(i + d)]; i += d
        sxx = reshape(flat[(i + 1):(i + d * d)], d, d); i += d * d
        n = flat[i + 1]; i += 1
        push!(re_pairs, re => (family = s.family, mean = sx ./ n, second = sxx ./ n, n = n))
    end
    out_pairs = Pair{Symbol, Any}[]
    for col in keys(template.outcome)
        s = getfield(template.outcome, col)
        st = (family = s.family, s1 = flat[i + 1], s2 = flat[i + 2], ss = flat[i + 3], n = flat[i + 4])
        i += 4
        push!(out_pairs, col => st)
    end
    hmm_pairs = Pair{Symbol, Any}[]
    for col in keys(template.hmm)
        s = getfield(template.hmm, col)
        lw = length(s.sum_w); lwy = length(s.sum_wy)
        sw = lw == 1 ? flat[i + 1] : flat[(i + 1):(i + lw)]; i += lw
        swy = lwy == 1 ? flat[i + 1] : flat[(i + 1):(i + lwy)]; i += lwy
        push!(hmm_pairs, col => (family = s.family, target = s.target, sum_w = sw, sum_wy = swy))
    end
    (re = NamedTuple(re_pairs), outcome = NamedTuple(out_pairs), hmm = NamedTuple(hmm_pairs))
end

# This site's DE-NORMALIZED additive sufficient statistics over the FIXED draws (stats round).
nlf_saem_stats_flat(dm, v, draws) =
    _nlf_saem_flatten(NoLimits.saem_sufficient_statistics(dm, nlf_natural(dm, v), draws))

# Per-subject (batch idx) form for the additivity proof: summing over idx == the population.
nlf_saem_stats_flat_idx(dm, v, draws, idx) =
    _nlf_saem_flatten(NoLimits.saem_sufficient_statistics(dm, nlf_natural(dm, v), draws, Int(idx)))

# The demo SAEM method, shared by the pooled reference fit, the eligibility split and the
# coordinator's γ schedule so all three agree. mstep_sa_on_params=false makes the pooled fit's
# numerical M-step a plain maximization, matching the federated L-BFGS-B; convergence_window >
# maxiters disables early stopping so the pooled fit runs the same fixed outer budget.
nlf_saem_method(maxiters) = NoLimits.SAEM(
    maxiters = Int(maxiters), sa_burnin_iters = 0, convergence_window = 50,
    mstep_sa_on_params = false)

# One COORDINATOR-side closed-form M-step from the server's summed flat stats. Reconstructs the
# aggregated stats (template structure from the coordinator's own draws), computes γ from the
# SAEM SA schedule at outer iteration `k`, runs the STATEFUL closed-form update, and returns the
# eligible params on the TRANSFORMED (wire) scale + the smoothed_state to carry to iter k+1.
# `smoothed_state === nothing` on k == 1. All demo closed-form params are scalar.
function nlf_saem_mstep(dm, v, draws, summed_flat, smoothed_state, k, maxiters)
    theta = nlf_natural(dm, v)
    template = NoLimits.saem_sufficient_statistics(dm, theta, draws)
    agg = _nlf_saem_unflatten(template, collect(Float64, summed_flat))
    method = nlf_saem_method(maxiters)
    γ = NoLimits._saem_gamma_schedule(Int(k), method.saem)
    updates, new_state = NoLimits.saem_closed_form_mstep(
        dm, agg, smoothed_state, theta, Float64(γ); method = method)
    θ_nat = deepcopy(theta)
    for name in keys(updates)
        setproperty!(θ_nat, name, getproperty(updates, name))
    end
    θ_t = dm.model.fixed.transform(θ_nat)
    names = String[String(n) for n in keys(updates)]
    vals = Float64[Float64(getproperty(θ_t, n)) for n in keys(updates)]
    (names, vals, new_state)
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
    probe_estimators: tuple = ()  # estimators the additivity probe checks; () = the default set


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
    # Naive-pooled theoph (no random effects): the model class MLE/MAP require. Reuses the
    # theoph loader and column map. estimator="mle" default; the additivity probe checks
    # mle and map (the RE estimators do not apply to a fixed-effects-only model).
    "theoph-pooled": ModelSpec(
        model=THEOPH_POOLED_MODEL, loader=load_theoph, primary_id="id", time_col="t",
        columns={"Subject": "id", "Time": "t", "Dose": "Dose", "conc": "conc"},
        num_sites=3, estimator="mle", probe_estimators=("mle", "map"),
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


# Random-effect estimators the SINGLE-SHOT (value, gradient) additivity probe checks.
ESTIMATORS = ("laplace", "focei", "ghq", "pooled")
# Fixed-effects-only estimators: they REQUIRE a model with no random effects (theoph-pooled).
FE_ESTIMATORS = ("mle", "map")
# MCEM and SAEM are NESTED estimators (local E-step, federated M-step): they do not fit the
# single-shot probe, so they are not in ESTIMATORS but are valid run-config values on any RE
# model.
MCEM_ESTIMATORS = ("mcem",)
SAEM_ESTIMATORS = ("saem",)
ALL_ESTIMATORS = ESTIMATORS + FE_ESTIMATORS + MCEM_ESTIMATORS + SAEM_ESTIMATORS

# MCEM demo settings, fixed (documented in docs/estimators.md). Kept in ONE place so the
# federated outer loop, the sites' E-step, and the pooled reference fit all agree: the pooled
# fit's `maxiters` == the federated outer-iteration budget, and both seed the same way.
MCEM_SAMPLE_SCHEDULE = 100   # SaemixMH posterior draws per subject per E-step
MCEM_OUTER_ITERS = 15        # fixed outer EM iterations (no convergence test; see docs)
MCEM_MSTEP_MAXFUN = 12       # inner L-BFGS-B evals per M-step part (approximate M-step is fine)
MCEM_SEED = 20260824         # base E-step seed; each site uses MCEM_SEED + site_id

# SAEM demo settings, fixed. The E-step reuses the MCEM sampler (SaemixMH), so it travels the
# mcem-* E-step keys on the wire. maxiters IS the fixed outer budget and also the pooled fit's
# maxiters (see nlf_saem_method); the closed-form path converges fast, so fewer draws than MCEM
# suffice. The pooled reference matches only up to Monte-Carlo noise (SAEM is stochastic and the
# per-site RNG partition differs), so acceptance is parameter-wise at SAEM_PARAM_TOL.
SAEM_SAMPLE_SCHEDULE = 50    # SaemixMH posterior draws per subject per E-step
SAEM_OUTER_ITERS = 20        # fixed outer EM iterations == the pooled fit's maxiters
SAEM_MSTEP_MAXFUN = 12       # inner L-BFGS-B evals per numerical M-step part
SAEM_SEED = 20260824         # base E-step seed; each site uses SAEM_SEED + site_id


def _method(nl, estimator: str, ghq_level: int):
    if estimator == "laplace":
        return nl.Laplace()
    if estimator == "focei":
        return nl.FOCEI()
    if estimator == "ghq":
        return nl.GHQuadrature(level=ghq_level)
    if estimator == "pooled":
        return nl.Pooled()
    if estimator == "mle":
        return nl.MLE()
    if estimator == "map":
        return nl.MAP()
    if estimator == "mcem":
        return nl.MCEM(sample_schedule=MCEM_SAMPLE_SCHEDULE, maxiters=MCEM_OUTER_ITERS)
    if estimator == "saem":
        return nl.seval("nlf_saem_method")(SAEM_OUTER_ITERS)
    raise ValueError(f"unknown estimator {estimator!r} (expected one of {ALL_ESTIMATORS})")


# --- federated MAP prior-carrier rule -------------------------------------------------
#
# MAP's objective is [sum over subjects of loglik] + ONE shared log-prior. The server just
# sums site payloads, so exactly ONE site must contribute the prior or it is counted S
# times. Site index 0 is the deterministic prior carrier: it computes MAP (its sweep
# includes the public prior), every other site computes MLE (loglik only). The naive server
# sum is then the pooled MAP objective. MLE and every RE estimator are unaffected.
def site_estimator(estimator: str, partition_id: int) -> str:
    """The estimator this site actually runs: MAP on the carrier (site 0), else MLE."""
    if estimator == "map" and int(partition_id) != 0:
        return "mle"
    return estimator


def map_prior(nl, dm, theta_transformed):
    """The PUBLIC MAP log-prior as (value, transformed-axes gradient).

    prior = population MAP - population MLE at the same theta and scale; the log-likelihood
    is computed identically in both sweeps and cancels, leaving exactly the log-prior. It is
    data-independent, so under DP the carrier adds it as an un-clipped, un-noised offset and
    it never enters the privacy accountant.
    """
    map_v, map_g = objective_and_gradient(nl, dm, theta_transformed, "map", 1, require_finite=False)
    mle_v, mle_g = objective_and_gradient(nl, dm, theta_transformed, "mle", 1, require_finite=False)
    return map_v - mle_v, map_g - mle_g


# --- differential privacy --------------------------------------------------------------
#
# The mechanism: each site clips every SUBJECT's (preconditioned) gradient to L2 norm
# <= `dp-clip`, so add/remove-one-subject moves the site sum by at most `dp-clip`; then
# each of the S sites adds N(0, (sigma*clip)^2 / S) per coordinate, so the noise on the
# federated sum is exactly N(0, (sigma*clip)^2) - the Gaussian mechanism at multiplier sigma.
# The noise is DISTRIBUTED (each site adds its 1/S share) so it would compose with SecAgg.

DP_ADJACENCY = "add/remove one subject"

# NOT seeded: reproducible privacy noise is no privacy (holding the seed recovers the exact
# clipped sum). One OS-entropy generator per process, so the per-round draws are one stream.
DP_RNG = np.random.default_rng(secrets.randbits(128))

# Substrings that mark a parameter as an RE SD/variance/covariance or the residual. Matched
# token-wise on the lowercased name, so `omega_cl`, `sigma`, `cov_ka_cl` land in the variance
# group and `cl`, `ka`, `v` in location. A false positive is fixed by a dp-groups override.
DP_VARIANCE_MARKERS = ("omega", "sigma", "tau", "corr", "cov", "sd", "var", "rho")

# Renyi-DP grid: fine below 10 (where the optimum sits for usable sigmas) and integral above.
DP_ALPHAS = np.unique(np.concatenate([np.linspace(1.01, 10.0, 900), np.arange(10, 513)]))


def dp_clip_sum(gradients, clip: float) -> np.ndarray:
    """Per-subject L2 clipping, then the site sum. Sensitivity of the result is `clip`."""
    gradients = np.atleast_2d(np.asarray(gradients, dtype=float))
    norms = np.linalg.norm(gradients, axis=1)
    factors = np.where(norms > clip, clip / np.maximum(norms, 1e-300), 1.0)
    return (gradients * factors[:, None]).sum(axis=0)


def dp_noise(shape, clip: float, sigma: float, num_sites: int) -> np.ndarray:
    """This site's share of the distributed Gaussian noise: variance (sigma*clip)^2 / S."""
    return DP_RNG.normal(0.0, sigma * clip / math.sqrt(num_sites), shape)


# Per-group clipping bounds each subject's per-group sub-vector on its own, so a subject
# atypical in the location coordinates does not eat the variance group's budget (the omega
# collapse). It is EXACTLY as private as joint clipping at C_total = sqrt(sum_g C_g^2):
# clipping subject i's group-g sub-vector to C_g bounds its whole L2 norm by C_total, so
# add/remove-one moves the site sum by at most C_total. We add ISOTROPIC noise sigma*C_total
# on every coordinate, so the release is one Gaussian mechanism at multiplier sigma - the
# accountant is UNCHANGED whichever clip mode is in force.


def dp_param_group(name: str, override: dict | None = None) -> str:
    """The DP group of one parameter: an explicit override, else the name heuristic."""
    override = override or {}
    if name in override:
        return str(override[name])
    tokens = re.split(r"[^a-z0-9]+", name.lower())
    is_var = any(t.startswith(m) for t in tokens if t for m in DP_VARIANCE_MARKERS)
    return "variance" if is_var else "location"


def dp_resolve_groups(names, override: dict | None = None):
    """(group_ids, group_names): coordinate -> group index, and the ordered group names."""
    labels = [dp_param_group(str(n), override) for n in names]
    ordered = list(dict.fromkeys(labels))
    index = {g: i for i, g in enumerate(ordered)}
    return [index[l] for l in labels], ordered


def dp_unmatched_group_names(names, override: dict | None = None):
    """Names the heuristic did not match to a variance marker and that carry no override,
    so they defaulted to 'location'. Membership never affects (eps, delta); this only flags
    a possible misclassification the operator may want to correct with dp-groups."""
    override = override or {}
    out = []
    for n in names:
        n = str(n)
        if n in override:
            continue
        tokens = re.split(r"[^a-z0-9]+", n.lower())
        if not any(t.startswith(m) for t in tokens if t for m in DP_VARIANCE_MARKERS):
            out.append(n)
    return out


def dp_group_clips(group_names, default_clip: float, per_group: dict | None = None):
    """Per-group clip C_g aligned to `group_names`; `per_group` overrides the default."""
    per_group = per_group or {}
    return [float(per_group.get(g, default_clip)) for g in group_names]


def dp_clip_total(group_clips) -> float:
    """C_total = sqrt(sum_g C_g^2): the L2 sensitivity of the per-group clipped site sum."""
    return float(math.sqrt(sum(c * c for c in group_clips)))


def dp_clip_sum_grouped(gradients, group_ids, group_clips) -> np.ndarray:
    """Per-subject, per-group L2 clipping, then the site sum. Each subject's sub-vector on
    group g's coordinates is clipped to group_clips[g], so its whole contribution has L2
    norm <= sqrt(sum_g C_g^2) = C_total."""
    gradients = np.atleast_2d(np.asarray(gradients, dtype=float))
    group_ids = np.asarray(group_ids, dtype=int)
    out = gradients.copy()
    for g, clip in enumerate(group_clips):
        cols = group_ids == g
        if not cols.any():
            continue
        block = gradients[:, cols]
        norms = np.linalg.norm(block, axis=1)
        factors = np.where(norms > clip, clip / np.maximum(norms, 1e-300), 1.0)
        out[:, cols] = block * factors[:, None]
    return out.sum(axis=0)


def parse_group_mapping(text) -> dict[str, str]:
    """Parse `a:x,b:y` run-config strings into {a: x, b: y}. Empty text -> {}."""
    out: dict[str, str] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition(":")
        if not sep:
            raise ValueError(f"bad group mapping entry {item!r}: expected 'name:value'")
        out[key.strip()] = value.strip()
    return out


def dp_epsilon(rounds: int, sigma: float, delta: float) -> float:
    """(eps, delta) for `rounds` full-batch Gaussian releases at noise multiplier `sigma`.

    One release is (alpha, alpha/(2 sigma^2))-RDP for every alpha > 1; RDP composes by
    addition, so T rounds are (alpha, T*alpha/(2 sigma^2))-RDP; convert to (eps, delta) with
    the standard tail bound and minimize over alpha.
    """
    eps = rounds * DP_ALPHAS / (2.0 * sigma**2) + math.log(1.0 / delta) / (DP_ALPHAS - 1.0)
    return float(eps.min())


def dp_batch_contributions(nl, dm, theta_transformed, estimator: str, ghq_level: int):
    """(per-batch values, per-batch transformed-axes gradients, largest batch size).

    Never leaves the site: the caller scales, clips and noises these before any release.
    """
    if estimator == "pooled":
        raise ValueError(
            "dp=true cannot use estimator='pooled': the naive-pooled objective calibrates "
            "its plug-in random effects on the whole data set and has no per-subject form, "
            "so a per-subject clipping bound does not exist for it. Use laplace, focei or ghq."
        )
    theta = np.asarray(theta_transformed, dtype=float)
    if estimator in FE_ESTIMATORS:
        # No random effects: the clipping unit is the individual. MAP's data part is the
        # per-individual MLE too (the prior is a separate public offset the carrier adds).
        vals, grads, maxids = nl.seval("nlf_mle_individuals")(dm, theta)
    else:
        vals, grads, maxids = nl.seval("nlf_dp_batches")(
            dm, theta, _method(nl, estimator, ghq_level)
        )
    vals = np.asarray(vals, dtype=float)
    grads = np.atleast_2d(np.asarray(grads, dtype=float))
    if not np.all(np.isfinite(vals)) or not np.all(np.isfinite(grads)):
        raise RuntimeError("non-finite per-subject objective/gradient in a dp round")
    return vals, grads, int(maxids)


def objective_and_gradient(nl, dm, theta_transformed, estimator: str, ghq_level: int,
                           require_finite: bool = True):
    """(value, gradient-on-transformed-axes) for one DataModel at a wire vector.

    `require_finite` is the fail-fast default for the reference/probe/fit paths, where a
    non-finite result at a fixed well-defined theta means a real bug. The FEDERATED round
    handler passes `require_finite=False`: a non-finite marginal at an optimizer's rough
    line-search probe is a legitimate estimator result, not a site failure, so the site must
    reply successfully with it and let the server backtrack on a finite penalty. Raising here
    would turn it into an error reply that the server's genuine-site-failure guard aborts on.
    """
    method = _method(nl, estimator, ghq_level)
    value, grad = nl.seval("nlf_objgrad")(dm, np.asarray(theta_transformed, dtype=float), method)
    value = float(value)
    grad = np.asarray(grad, dtype=float)
    if require_finite and (not np.isfinite(value) or not np.all(np.isfinite(grad))):
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
    # MCEM/SAEM are stochastic: seed the global RNG so the pooled reference is deterministic
    # (and comparable to the seeded federated fit). Otherwise honor the model's fit_seed.
    fit_seed = MCEM_SEED if estimator == "mcem" else (
        SAEM_SEED if estimator == "saem" else sp.fit_seed)
    if fit_seed:
        nl.seval("import Random")
        nl.seval("Random.seed!")(fit_seed)
    method = _method(nl, estimator, ghq_level)
    fit = nl.fit_model(dm, method, pooled_init=True) if sp.pooled_init else nl.fit_model(dm, method)
    theta = np.asarray(nl.seval("nlf_fit_theta")(fit), dtype=float)
    # MCEM/SAEM have no deterministic theta-objective (objective_and_gradient rejects them), so
    # report the Laplace marginal loglik at the fitted theta as a deterministic quality yardstick.
    value_estimator = "laplace" if estimator in ("mcem", "saem") else estimator
    value, _ = objective_and_gradient(nl, dm, theta, value_estimator, ghq_level)
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
    The neural model is checked for `laplace` only; the PK/growth models for all four; the
    naive-pooled model for `mle` and `map` (map through the site-0 prior-carrier rule).
    """
    import NoLimitsPy as nl
    sp = spec(model)
    if num_sites <= 0:
        num_sites = sp.num_sites
    df = dataset(model, source, seed, nl)
    pooled_dm = build_data_model(nl, model, df)
    site_dms = [build_data_model(nl, model, s) for s in partition(df, num_sites, sp.primary_id)]
    theta = _theta_for(nl, model, pooled_dm, source)
    if sp.probe_estimators:
        estimators = sp.probe_estimators
    else:
        estimators = ("laplace",) if sp.acceptance == "nn" else ESTIMATORS
    out = {}
    for estimator in estimators:
        pooled_value, pooled_grad = objective_and_gradient(nl, pooled_dm, theta, estimator, ghq_level)
        # Site i runs site_estimator(estimator, i): MAP only on the carrier (site 0), MLE
        # elsewhere, so the summed payload is the pooled objective and gradient.
        pairs = [objective_and_gradient(nl, dm, theta, site_estimator(estimator, i), ghq_level)
                 for i, dm in enumerate(site_dms)]
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


def mcem_additivity_probe(model: str = DEFAULT_MODEL, seed: int = DEFAULT_SEED,
                          source: str = DEFAULT_SOURCE):
    """MCEM M-step exactness: sum over subjects of the per-subject Q (value, gradient) at
    FIXED draws == the population Q, for both parts (q1, q2). Machine-precision, so this is
    the proof that the federated M-step IS the pooled M-step at the same draws.

    One Julia boot. The draws come from a single E-step on the pooled DataModel; each batch is
    one subject, so the per-idx sum equals summing whole sites (a site is a set of batches).
    """
    import NoLimitsPy as nl
    dm = build_data_model(nl, model, dataset(model, source, seed, nl))
    theta0 = np.asarray(nl.seval("nlf_theta0")(dm), dtype=float)
    parts = nl.seval("nlf_mcem_parts")(dm)
    part_names = {"q1": [str(s) for s in parts.q1], "q2": [str(s) for s in parts.q2]}
    # One E-step (state=nothing) at theta0 to produce the fixed draws.
    draws, _ = nl.seval("nlf_mcem_estep")(
        dm, theta0, MCEM_SAMPLE_SCHEDULE, MCEM_OUTER_ITERS, MCEM_SEED, None
    )
    nb = int(len(draws))
    out = {}
    for part, fnames in part_names.items():
        if not fnames:
            continue
        pooled_Q, pooled_g = nl.seval("nlf_mcem_q")(dm, theta0, draws, part, fnames)
        pooled_Q = float(pooled_Q)
        pooled_g = np.asarray(pooled_g, dtype=float)
        fed_Q = 0.0
        fed_g = np.zeros_like(pooled_g)
        for i in range(1, nb + 1):
            Qi, gi = nl.seval("nlf_mcem_q_idx")(dm, theta0, draws, i, part, fnames)
            fed_Q += float(Qi)
            fed_g += np.asarray(gi, dtype=float)
        out[part] = {
            "names": fnames,
            "pooled_value": pooled_Q,
            "federated_value": fed_Q,
            "value_rel": abs(fed_Q - pooled_Q) / max(abs(pooled_Q), 1e-300),
            "gradient_rel": float(
                np.linalg.norm(fed_g - pooled_g) / max(np.linalg.norm(pooled_g), 1e-300)
            ),
        }
    return {"model": model, "subjects": nb, "theta": theta0.tolist(), "probes": out}


def saem_additivity_probe(model: str = DEFAULT_MODEL, seed: int = DEFAULT_SEED,
                          source: str = DEFAULT_SOURCE):
    """SAEM sufficient-statistics exactness: sum over subjects of the per-subject DE-NORMALIZED
    additive statistics == the population statistics, to machine precision. This is the proof
    that the server's numpy sum of the per-site payloads IS the pooled sufficient statistics.

    One Julia boot. The draws come from a single E-step on the pooled DataModel; each batch is
    one subject, so the per-idx sum equals summing whole sites (a site is a set of batches).
    """
    import NoLimitsPy as nl
    dm = build_data_model(nl, model, dataset(model, source, seed, nl))
    theta0 = np.asarray(nl.seval("nlf_theta0")(dm), dtype=float)
    draws, _ = nl.seval("nlf_mcem_estep")(
        dm, theta0, SAEM_SAMPLE_SCHEDULE, SAEM_OUTER_ITERS, SAEM_SEED, None
    )
    nb = int(len(draws))
    pooled = np.asarray(nl.seval("nlf_saem_stats_flat")(dm, theta0, draws), dtype=float)
    fed = np.zeros_like(pooled)
    for i in range(1, nb + 1):
        fed += np.asarray(
            nl.seval("nlf_saem_stats_flat_idx")(dm, theta0, draws, i), dtype=float
        )
    parts = nl.seval("nlf_saem_parts")(dm)
    return {
        "model": model, "subjects": nb, "theta": theta0.tolist(),
        "closed_form": [str(s) for s in parts.closed_form],
        "numerical": [str(s) for s in parts.numerical],
        "value_rel": float(np.max(np.abs(fed - pooled) / np.maximum(np.abs(pooled), 1e-12))),
        "pooled": pooled.tolist(), "federated": fed.tolist(),
    }


if __name__ == "__main__":
    # python -m nolimits_flower.task {fit|ref|probe|mcem-probe|saem-probe} <model> [estimator] [ghq] [seed] [source]
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
    elif mode == "mcem-probe":
        result = mcem_additivity_probe(model, rest[2], rest[3])
    elif mode == "saem-probe":
        result = saem_additivity_probe(model, rest[2], rest[3])
    elif mode == "fit":
        result = pooled_fit(model, *rest)
    else:
        result = pooled_reference(model, *rest)
    print("POOLED_JSON " + json.dumps(result))
