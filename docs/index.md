# NoLimitsFlower

Federated nonlinear mixed-effects (NLME) estimation with [Flower](https://flower.ai) and
[NoLimitsPy](https://github.com/manuhuth/NoLimitsPy), the Python interface to
[NoLimits.jl](https://github.com/manuhuth/NoLimits.jl).

!!! info "This is a showcase / testbed"
    This repository is the **local-only showcase** of NoLimits' federated-learning
    capabilities. It runs as a reproducible Flower *simulation* on a single machine, with
    the acceptance gates and equivalence checks that prove the federation math is exact. The
    **full deployable version — one `flower-superlink` at the coordinator, one
    `flower-supernode` per real institution, over TLS — will be published soon.** Everything
    on these pages is documented *as-built* from the code at the current commit; where a
    capability is designed but not yet wired in (differential privacy, secure aggregation),
    the page says so explicitly.

## What it is

Sites hold their own subjects and never send subject-level data anywhere. At a common
parameter vector `theta`, each site returns two aggregates: its marginal log-likelihood and
that value's gradient. Because subjects are **disjoint across sites**, the sum of those site
contributions *is* the pooled marginal log-likelihood and its gradient — so the federated
optimum is the pooled `fit_model` optimum, not an approximation of it.

This is the property the whole package is built to demonstrate:

$$
L_\text{fed}(\theta) \;=\; \sum_{s=1}^{S} \ell_s(\theta) \;=\; \ell_\text{pooled}(\theta),
\qquad
\nabla L_\text{fed}(\theta) \;=\; \sum_{s=1}^{S} \nabla \ell_s(\theta) \;=\; \nabla \ell_\text{pooled}(\theta).
$$

Every wired estimator is a **sum of per-subject terms**, so nothing couples two subjects and
splitting subjects across sites cannot change the total. This is checked to `1e-8` for every
model — the property is called **additivity** throughout these docs.

## The equivalence headline

Federated fit versus the pooled `fit_model` on identical data, model and start point:

| what | result |
|---|---|
| classical PK / growth models | federated `theta*` matches pooled to `1e-6` objective, `1e-3` per parameter |
| **additivity** (exact-FL property, all 4 models) | value and gradient rel. diff `≤ 2.4e-16` at `theta0` |
| **federated *neural* NLME == pooled** | additivity gated to `1e-8`; objective agreement reported (weights non-identifiable, so parameters are not compared) |

The neural result is the striking one: a feed-forward network embedded as the mean function
of a mixed-effects model federates **exactly**, the summed site `(value, gradient)` equalling
the pooled-data call to `1e-8`. See [Models](models.md) and [Estimators](estimators.md).

## Quickstart

Prerequisites: Python 3.11+, Julia 1.11 (the version juliapkg selects), git.

```bash
python3 -m venv .venv
.venv/bin/pip install -e . "git+https://github.com/manuhuth/NoLimitsPy"
```

NoLimits' federation primitives (`objective_and_gradient`, `build_fit_context`) shipped in
**NoLimits v0.2.6**. The demo pins a shared Julia project (`julia_env/`, gitignored):

```bash
julia +1.11 -e 'import Pkg; Pkg.activate("julia_env");
                Pkg.add([Pkg.PackageSpec(name="NoLimits", version="0.2.6"),
                         Pkg.PackageSpec(name="PythonCall")])'
export PYTHON_JULIAPKG_PROJECT="$PWD/julia_env"
export PYTHON_JULIAPKG_OFFLINE=yes
```

Run the default demo (real warfarin PK data, Laplace estimator). Each site boots its own
Julia, so cap the simulation concurrency:

```bash
flwr run . --stream --federation-config \
  "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
```

Equal `client-resources-num-cpus` and `init-args-num-cpus` size the Ray actor pool to **one
actor** that serves all three sites — one Julia boot, one model compilation, no re-warm
mid-fit (see [Architecture](architecture.md)).

## The knobs at a glance

All are `--run-config 'key=value ...'` overrides of `[tool.flwr.app.config]` in
`pyproject.toml`. These are the knobs that **exist in the code today**:

| key | default | meaning |
|---|---|---|
| `model` | `"warfarin"` | `warfarin`, `theophylline`, `warfarin-nn` (neural) or `orange` (growth) — see [Models](models.md) |
| `estimator` | `"laplace"` | `laplace`, `focei`, `ghq` or `pooled` — see [Estimators](estimators.md) |
| `ghq-level` | `3` | Gauss-Hermite quadrature level when `estimator="ghq"` (1–3 is the stable range) |
| `data-source` | `"warfarin"` | the real warfarin PK data, or `"simulated"` for the seeded synthetic set |
| `data-seed` | `20260818` | which simulated data set; ignored when `data-source="warfarin"` |
| `max-rounds` | `100` | cap on federated rounds (L-BFGS-B `maxfun`) |
| `fail-site` | `-1` | **testing only**: that site id raises in its handler |

`--run-config` values are TOML, so a string needs its own quotes inside the shell quotes:
`--run-config 'model="theophylline"'`.

!!! note "Secure aggregation and differential privacy"
    A `secagg` knob, the `dp*` family, and the RDP accountant described on the
    [Differential privacy](differential-privacy.md) page are the **designed** next privacy
    layer. They are **not yet wired into this showcase's run-config** — that page documents
    the design and its accounting, and ships with the deployable version.

## Where to go next

- **[Models](models.md)** — the four models, their real data and licenses, and the
  classical-vs-neural-on-identical-warfarin highlight.
- **[Estimators](estimators.md)** — `laplace | focei | ghq | pooled`, one `flwr run`
  command each, and when to reach for which.
- **[Differential privacy](differential-privacy.md)** — the planned DP guarantee, its
  knobs, the per-group clipping rationale, and a worked `(σ, T) → ε` accountant table.
- **[Optimizer & scipy interface](optimizer.md)** — the pure-numpy/scipy server, one round
  per objective evaluation, and preconditioning.
- **[Architecture](architecture.md)** — prepare round, main-thread Julia warm-up, and
  additivity as the exactness proof.
