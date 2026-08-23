# Estimators

Four estimators are wired, all through the same protocol —
`objective_and_gradient(method, ctx, theta)` — so a site's handler is estimator-agnostic
(`task._method` maps the `estimator` string to the NoLimits method; `task.objective_and_gradient`
calls it). Every one of them is a **sum over subjects**, so the summed site contributions are
the pooled-data value and gradient **exactly**. Additivity (federated == pooled) holds for
**all four**.

| `estimator` | NoLimits method | additivity of (value, gradient) | fit acceptance | per-round cost |
|---|---|---|---|---|
| `laplace` (default) | `Laplace()` | value 0.0, gradient 1.8e-16 | strict: objective `1e-6`, every parameter `1e-3` | 0.10 s |
| `focei` | `FOCEI()` | value 1.7e-16, gradient 2.1e-16 | strict, same tolerances | 0.11 s |
| `ghq` | `GHQuadrature(level=ghq-level)` | value 2.0e-16, gradient 2.3e-16 (level 5) | one-sided: no worse than the pooled fit | 0.12 s (lvl 3), 0.13 s (lvl 5) |
| `pooled` | `Pooled()` | value 0.0, gradient 2.4e-16 | objective `1e-6`, parameters `1e-2` | 0.11 s |

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

## Not federated

`SAEM` and `MCEM` are **not** federated: they need a per-site E-step sufficient-statistics
primitive upstream in NoLimits. `MLE` and `MAP` have the protocol but require a model without
random effects, which is not what this package is for.
