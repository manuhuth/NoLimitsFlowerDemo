# Estimators

Four estimators are wired, all through the same protocol —
`objective_and_gradient(method, ctx, theta)` — so a site's handler is estimator-agnostic
(`task._method` maps the `estimator` string to the NoLimits method; `task.objective_and_gradient`
calls it). Every one of them is a **sum over subjects**, so the summed site contributions are
the pooled-data value and gradient **exactly**. Additivity (federated == pooled) holds for
**all four**. (`mle`/`map` federate the same way on the no-RE model; `mcem` and `saem` are the
**nested** estimators - local E-step, federated M-step - documented at the end.)

| `estimator` | NoLimits method | additivity of (value, gradient) | fit acceptance |
|---|---|---|---|
| `laplace` (default) | `Laplace()` | value 0.0, gradient 1.8e-16 | strict: objective `1e-6`, every parameter `1e-3` |
| `focei` | `FOCEI()` | value 1.7e-16, gradient 2.1e-16 | strict, same tolerances |
| `ghq` | `GHQuadrature(level=ghq-level)` | value 2.0e-16, gradient 2.3e-16 (level 5) | one-sided: no worse than the pooled fit |
| `pooled` | `Pooled()` | value 0.0, gradient 2.4e-16 | objective `1e-6`, parameters `1e-2` |

Why the sums are exact in every case: subjects are independent, and each estimator's
objective is a per-subject (per-random-effect-batch) term. `Laplace` and `FOCEI` find each
subject's empirical-Bayes mode from that subject's own data; `GHQuadrature` integrates each
subject's batch on its own quadrature grid; `Pooled` plugs in a per-subject `eta` and
evaluates a per-subject likelihood. Nothing in any of them couples two subjects, so splitting
subjects across sites cannot change the total.

The estimator-agnostic additivity check needs no federation and boots Julia once for all four:

```bash
python -m nolimits_flower.task probe        # per-site sums vs the pooled-data call
```

## `laplace` — the workhorse

```bash
flwr run . --stream --run-config 'estimator="laplace"' --federation-config \
  "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
```

The default. `Laplace()` approximates each subject's marginal likelihood by a second-order
expansion at that subject's **empirical-Bayes mode**, using the **exact Hessian** of the
integrand. It is accurate, differentiable end to end, and its objective is smooth enough that
the server's L-BFGS-B converges cleanly and the federated `theta*` matches the pooled
`fit_model` to `1e-6` in objective and `1e-3` in every natural-scale parameter.

**Reach for it** as the default for any identifiable NLME model — it is what all four models
document and what the neural model's additivity gate uses.

## `focei` — Gauss-Newton / Fisher-information curvature

```bash
flwr run . --stream --run-config 'estimator="focei"' --federation-config \
  "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
```

FOCEI is the classical pharmacometrics estimator (First-Order Conditional Estimation with
Interaction). Like Laplace it expands each subject's likelihood at the empirical-Bayes mode,
but instead of the full exact Hessian it uses a **Gauss-Newton / Fisher-information
approximation** to the curvature: it keeps the outer-product-of-gradients term (the Fisher
information of the observation model) and drops the second-derivative-of-residual term. On
well-specified PK/PD models the dropped term is small near the mode, so FOCEI tracks Laplace
closely while being cheaper and often better-conditioned.

In this package FOCEI is federated identically to Laplace — a per-subject sum — so it hits
the **same strict acceptance** (objective `1e-6`, parameters `1e-3`) and its additivity is
exact (value 1.7e-16, gradient 2.1e-16). On the warfarin data it also needed the fewest
rounds of any estimator (34).

**Reach for it** for classical PK/PD-style models where the Fisher-information curvature is a
good approximation and you want the pharmacometrics-standard estimator.

## `ghq` — Gauss-Hermite quadrature

```bash
# ghq-level=1 (the [tool.flwr.app.config] default) is the level whose federated fit reliably
# converges to the pooled optimum on this small data; levels 2-3 converge to a rougher local
# optimum (caveat below).
flwr run . --stream --run-config 'estimator="ghq" ghq-level=1 max-rounds=200' \
  --federation-config "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
```

`GHQuadrature(level=ghq-level)` integrates each subject's random-effect batch numerically on
a Gauss-Hermite grid, rather than approximating it at the mode. Higher `ghq-level` means more
quadrature nodes.

!!! warning "The `ghq-level` local-optima caveat — honest"
    The quadrature objective is **rough** on this model, so the federated L-BFGS-B and
    `fit_model`'s Optim LBFGS can settle in **different local optima** above level 1
    (measured on the warfarin data, current scipy). Every level *converges* — none aborts —
    but only level 1 lands on the pooled optimum:

    - **level 1** converges in ~23 rounds and passes the one-sided gate: the federated
      optimum equals the pooled `fit_model` optimum to `4e-14`.
    - **level 2** converges (~81 rounds) but lands `6.1e-03` *worse* than pooled, so the
      one-sided gate **fails**.
    - **level 3** converges (~68 rounds) but lands `3.6e-02` *worse* than pooled, so the
      one-sided gate **fails**.

    None of this is a federation error: the summed objective **is** the pooled objective
    exactly (additivity `2.3e-16`, verified by the probe). It is an optimizer-path problem —
    two different optimizers on the same non-convex, rough surface — which is why the run can
    only ever apply a **one-sided** gate ("no worse than pooled") and why, on this demo's
    small sites, only `ghq-level=1` currently satisfies it.

    A rough GHQ probe *theta* whose marginal is non-finite no longer aborts the fit: the
    site reports the non-finite contribution as a normal reply and the server backtracks on a
    finite penalty, so L-BFGS-B steps back and continues (a genuine **site failure** — an
    error reply — still aborts, see [Architecture](architecture.md)). Before that fix, level 3
    could drive the line search into such a *theta* and abort mid-optimization; it now
    converges (to the rougher optimum above). For a robust federated fit on this demo, prefer
    `laplace` or `focei`, or keep `ghq-level=1`.

**Reach for it** when you want a quadrature-based marginal likelihood rather than a
mode-based one, and are prepared to tune `ghq-level` per data set; on this demo only level 1
passes the one-sided gate.

## `pooled` — naive-pooled plug-in

```bash
flwr run . --stream --run-config 'estimator="pooled"' --federation-config \
  "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
```

`Pooled()` first **calibrates** a plug-in strategy per random effect and then optimizes
`loglikelihood(dm, theta, eta(theta))` — it plugs a deterministic `eta` into a per-subject
likelihood rather than integrating the random effect out.

!!! note "`pooled` is exact here, but that is model-conditional"
    The calibration looks at the data: it probes the first individual's random-effect
    distributions and demotes a strategy (`:mean` → `:median` → `:zero`, or to Monte-Carlo
    draws) if it is not ForwardDiff-safe there. In **this** model every random effect is
    `LogNormal`, whose mean is finite and smooth, so every site resolves to `:mean` and the
    plug-in `eta` is `exp(omega²/2)` — a function of **`theta` alone**, identical on every
    site, hence exact additivity (`2.4e-16`).

    A model where the resolution depends on the **data** (a normalizing-flow random effect,
    or a strategy demoted on one site's data only) would calibrate per site, and the
    federated objective would stop being the pooled-data objective. **Re-run
    `python -m nolimits_flower.task probe` after changing the model** — the slow test does
    exactly that.

Its acceptance uses a looser `1e-2` parameter tolerance: the plug-in `eta` depends on `omega`
only through `exp(omega²/2)`, so the objective is nearly flat in the omegas. The two fits
agree to `1.8e-10` in the objective while `omega_cl` differs by `2.8e-03` — a plateau, not a
federation error.

**Reach for it** when a fast plug-in fit is enough and every random effect resolves to a
data-independent plug-in (as `LogNormal` REs do here). It is exact when the plug-in `eta` is
data-independent.

## `mle` and `map` — fixed-effects-only (naive-pooled)

```bash
# MLE: maximum likelihood, priors ignored.
flwr run . --stream --run-config 'model="theoph-pooled" estimator="mle"' \
  --federation-config "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"

# MAP: adds the fixed-effects log-priors.
flwr run . --stream --run-config 'model="theoph-pooled" estimator="map"' \
  --federation-config "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
```

`MLE()` and `MAP()` fit a model with **no random effects** - population fixed effects plus a
residual error only. They **require** such a model and error on a mixed-effects one, so they
run on the dedicated `model="theoph-pooled"` (the same real theoph PK data and 1-compartment
model as `theophylline`, but naive-pooled). `MLE` maximizes the log-likelihood; `MAP` adds the
fixed effects' log-priors (weakly-informative `LogNormal` priors on `ka`, `cl`, `v`, `sigma`),
so the **same** model serves both: `MLE` ignores the priors, `MAP` uses them. Both hit the
strict acceptance (objective `1e-6`, parameters `1e-3`).

**MLE is a pure per-subject sum**, so the federated sum is the pooled objective exactly, like
the other estimators.

!!! note "Federated MAP: the prior carrier"
    MAP's objective is `[Σ over subjects loglik] + one shared log-prior`. The server just sums
    the site payloads, so exactly **one** site must contribute the prior, or it is counted
    once per site. The demo designates **site index 0** as the deterministic prior carrier: it
    runs `MAP` (its sweep includes the prior), every other site runs `MLE` (log-likelihood
    only). The naive server sum is then the pooled MAP objective, additive to machine precision
    (`1.2e-16`). Under DP the prior is **public** (data-independent), so the carrier adds it as
    an un-clipped, un-noised offset after the data aggregation; the prior never enters the
    accountant, and `ε(map+dp) == ε(mle+dp)` at matched knobs.

**Reach for them** when the model is fixed-effects-only (no between-subject random effects) and
you want a point estimate: `mle` without priors, `map` with them.

## `mcem` — Monte-Carlo EM (nested: local E-step, federated M-step)

```bash
flwr run . --stream --run-config 'model="warfarin" data-source="simulated" estimator="mcem"' \
  --federation-config "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
```

MCEM is a **nested** estimator (SAEM below is the other). It does not go through the single-shot
`objective_and_gradient` protocol above; there is no deterministic `theta`-objective to sum.
Instead the server drives an outer EM loop, and each **outer iteration** is three federated
rounds:

1. **E-step (LOCAL).** The server broadcasts the current `theta`; each site runs
   `mcem_e_step` over **its own subjects**, drawing `p(eta_i | y_i, theta)` per subject (the
   `SaemixMH` sampler, 100 draws each). The draws and the sampler's warm-start state are
   **cached in the site** (a module global, keyed by the outer-iteration index) and **held
   fixed** for this iteration's M-step. Nothing about the draws ever leaves the site - the
   E-step is not aggregated.
2. **M-step over `q1` (FEDERATED).** The Monte-Carlo `Q(theta) = Σ_subject (1/M) Σ_m log
   f(y_i, eta_i^m | theta)` at those fixed draws is a **per-subject sum**, so its value and
   gradient federate exactly like the other estimators. The server runs a small L-BFGS-B over
   the **`q1`** parameters (the observation-side ones - `ka`, `cl`, `v`, `sigma`); each
   objective evaluation broadcasts a candidate `theta`, each site returns
   `mcem_q_objective_and_gradient(part=:q1, …)` over its cached draws, and the server sums.
3. **M-step over `q2` (FEDERATED).** The same, over the **`q2`** parameters (the
   random-effect-distribution ones - `omega_ka`, `omega_cl`, `omega_v`). The `q1`/`q2` split
   is `mcem_q_partition`; it mirrors the two independent maximizations `fit_model(dm, MCEM())`
   does, and the sites report it in the prepare round.

The server stays **Julia-free** - it only sums per-site `(value, gradient)` pairs and runs
scipy, exactly the single-shot pattern, wrapped in the outer loop. A **fixed outer budget**
(`task.MCEM_OUTER_ITERS`, 15) is used with no convergence test, and each inner M-step is
capped at `task.MCEM_MSTEP_MAXFUN` evaluations - MCEM tolerates approximate M-steps. MCEM
**requires random effects**, so it runs on `warfarin`/`theophylline`, **not** the no-RE
`theoph-pooled`.

**Exactness.** At fixed draws the federated M-step **is** the pooled M-step, to machine
precision: summing the per-subject `Q` (value and gradient) reproduces the population `Q` at
**0.0** relative error for both parts (`python -m nolimits_flower.task mcem-probe warfarin` /
`tests/test_equivalence.py::test_mcem_q_additivity_at_fixed_draws`). This is the exactness
proof. The end-to-end acceptance is **parameter-wise vs `fit_model(dm, MCEM())`** at a
**Monte-Carlo** tolerance (`5e-2`): MCEM is stochastic and the per-site RNG partition differs
from the pooled run, so the two optima agree only up to sampling noise (measured worst
parameter `8.85e-3` on warfarin/simulated).

!!! warning "DP-MCEM is the most privacy-expensive estimator"
    Under `dp=true` each inner M-step becomes fixed-schedule **DP-Adam** on per-subject-clipped,
    Gaussian-noised part gradients (the E-step still stays local; only the aggregated noised
    gradient is released). But **DP composes over every Adam step across every outer
    iteration**: the composition length is `outer × 2 parts × mcem-dp-mstep-steps`, and the
    accountant counts **all** of them. On warfarin/simulated, `15 × 2 × 2 = 60` releases at
    `σ=0.5`, `δ=1e-5` already spends `ε ≈ 194` (and `ε ≈ 489` for a site's own release against
    the server without SecAgg). A single-shot estimator spends one release per round; MCEM
    spends one per M-step gradient of every outer iteration, so its budget is far larger for a
    comparable fit. Choose it under DP only with that in mind.

    ```bash
    flwr run . --stream --run-config \
      'model="warfarin" data-source="simulated" estimator="mcem" dp=true dp-noise-multiplier=0.5 dp-clip-mode="joint" mcem-dp-mstep-steps=2' \
      --federation-config "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
    ```

**Reach for it** when you want a simulation-based (rather than Laplace-approximate) marginal
likelihood and can afford the nested loop; avoid it under DP unless the large `ε` is acceptable.

## `saem` — Stochastic-Approximation EM (nested: local E-step, federated hybrid M-step)

```bash
flwr run . --stream --run-config 'model="warfarin" data-source="simulated" estimator="saem"' \
  --federation-config "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
```

SAEM is the second **nested** estimator, and like MCEM it does not go through the single-shot
`objective_and_gradient` protocol. Its E-step is the **same** local MH sampler MCEM uses (the
draws are cached in the site and never leave it), but its M-step is **hybrid**: the
random-effect covariances and the residual variance have a **closed form** in the sufficient
statistics, while the remaining parameters are maximized numerically. Each **outer iteration**
is four rounds:

1. **E-step (LOCAL).** Identical to MCEM - the server broadcasts `theta`, each site draws
   `p(eta_i | y_i, theta)` for **its own subjects** with `SaemixMH` and caches the FIXED draws.
   Nothing is aggregated.
2. **Sufficient-statistics round (FEDERATED sum).** Each site emits its **per-subject-additive**
   SAEM sufficient statistics (`saem_sufficient_statistics`) over those draws, DE-NORMALIZED so
   the population statistics are a plain coordinate-wise sum (RE moments as `Σx = mean·n` and
   `Σxx' = second·n`; outcome/HMM fields are already sums). The server **sums** the per-site
   payloads in numpy - one vector, no Julia.
3. **Closed-form M-step (COORDINATOR site).** Site 0 re-normalizes the summed statistics and
   runs the **stateful** closed-form update (`saem_closed_form_mstep`, bit-identical to the
   fit) at this iteration's stochastic-approximation step size `γ`, threading the smoothed state
   to the next iteration. It returns the closed-form-eligible parameters - here `omega_ka`,
   `omega_cl`, `omega_v`, `sigma`.
4. **Numerical M-step (FEDERATED).** A small L-BFGS-B over the remaining parameters (`ka`, `cl`,
   `v`), reusing the **MCEM `Q` kernel** restricted to the non-closed-form names; each evaluation
   broadcasts a candidate `theta` and the server sums the per-site `(value, gradient)` exactly as
   MCEM does. The closed-form/numerical split (with the `q1`/`q2` part each numerical name lives
   in) is `saem_closed_form_eligibility` + `mcem_q_partition`, reported in the prepare round.

The server stays **Julia-free** - it only sums per-site payloads and runs scipy. A **fixed outer
budget** (`task.SAEM_OUTER_ITERS`, 20, == the pooled fit's `maxiters`) drives the γ schedule with
no convergence test. SAEM **requires random effects**, so it runs on `warfarin`/`theophylline`,
**not** the no-RE `theoph-pooled`.

**Exactness.** The summed per-site payload **is** the pooled sufficient statistics, to machine
precision: summing each subject's DE-NORMALIZED additive statistics reproduces the population
statistics at **< 1e-10** relative error (`python -m nolimits_flower.task saem-probe warfarin` /
`tests/test_equivalence.py::test_saem_sufficient_stats_additivity`). That is the exactness proof
for the closed-form half; the numerical half is exact for the same reason MCEM's is. The
end-to-end acceptance is **parameter-wise vs `fit_model(dm, SAEM())`** at a **Monte-Carlo**
tolerance (`5e-2`): SAEM is stochastic and the per-site RNG partition differs from the pooled
run, so the two optima agree only up to sampling noise.

!!! warning "DP-SAEM composes over the stats release AND the numerical Adam steps - but is cheaper than DP-MCEM"
    Under `dp=true` the E-step still stays local; what is released each outer iteration is (a)
    **one** noised sufficient-statistics vector - each site clips **per subject** (the DP unit)
    and adds its Gaussian noise share, the server sums - and (b) the numerical M-step, now
    fixed-schedule **DP-Adam** on per-subject-clipped, noised part gradients
    (`mcem-dp-mstep-steps` steps per numerical part - the knob SAEM shares with MCEM). The
    composition length is therefore
    `outer × (1 stats release + numerical-parts × mcem-dp-mstep-steps)`, and the accountant counts
    **all** of them at the one noise multiplier. This is **cheaper per iteration than DP-MCEM**:
    the closed-form parameters (`omega_*`, `sigma`) ride the single shared stats release instead
    of each getting its own DP-Adam block. Robustness to the noise is built in: the closed-form
    M-step **floors a noise-perturbed covariance** (NoLimits clamps a negative variance estimate
    to zero), and the numerical Adam step **reverts on a non-finite gradient**, so the fit stays
    finite. Nothing un-noised is released: no objective, no per-site contribution.

    ```bash
    flwr run . --stream --run-config \
      'model="warfarin" data-source="simulated" estimator="saem" dp=true dp-noise-multiplier=0.5 dp-clip-mode="joint" mcem-dp-mstep-steps=2' \
      --federation-config "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
    ```

**Reach for it** when you want the classical SAEM closed-form M-step (rather than MCEM's fully
numerical one) and a single sufficient-statistics release per iteration under DP.

## All wired estimators federate

Every estimator above is federated: the six single-shot ones (`laplace`, `focei`, `ghq`,
`pooled`, `mle`, `map`) as exact per-subject sums, and the two nested ones (`mcem`, `saem`) as a
local E-step plus a federated M-step. No wired estimator runs pooled-only.
