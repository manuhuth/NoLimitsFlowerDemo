# NoLimitsFlower

Federated nonlinear mixed-effects (NLME) estimation with [Flower](https://flower.ai) and
[NoLimitsPy](https://github.com/manuhuth/NoLimitsPy), the Python interface to
[NoLimits.jl](https://github.com/manuhuth/NoLimits.jl).

[![CI](https://github.com/manuhuth/NoLimitsFlowerDemo/actions/workflows/ci.yml/badge.svg)](https://github.com/manuhuth/NoLimitsFlowerDemo/actions/workflows/ci.yml)
[![Docs](https://github.com/manuhuth/NoLimitsFlowerDemo/actions/workflows/docs.yml/badge.svg)](https://manuhuth.github.io/NoLimitsFlowerDemo/)

Sites hold their own subjects and never send subject-level data anywhere. At a common
parameter vector each site returns two aggregates - its marginal log-likelihood and that
value's gradient - and, because subjects are disjoint across sites, the sum of those
contributions *is* the pooled marginal log-likelihood and its gradient. This repository is a
**local-only showcase**: a reproducible Flower simulation on one machine, with acceptance
gates that check the federation math. The deployable version (one `flower-superlink`, one
`flower-supernode` per institution, over TLS) is coming soon.

## The headline

The federated fit reproduces the pooled `fit_model` fit across `laplace`, `focei`, `ghq` and
`pooled` - including an 87-parameter **neural** mixed-effects model. The exact-FL property
(summed site `(value, gradient)` equals the pooled-data call) holds to `~1e-9` or better for
every model; classical PK and growth fits match pooled to `1e-6` objective and `1e-3` per
parameter. Details and the full tables are in the docs under
[Estimators](https://manuhuth.github.io/NoLimitsFlowerDemo/estimators/) and
[Models](https://manuhuth.github.io/NoLimitsFlowerDemo/models/).

## Installation

You need only **Python 3.11+** and **git**. Julia and NoLimits are provisioned
automatically by `juliapkg` on the first run (several minutes; later runs start in seconds).
Every step after step 2 runs from the repository root - the directory holding
`pyproject.toml`.

1. Clone: `git clone https://github.com/manuhuth/NoLimitsFlowerDemo`
2. Enter the root: `cd NoLimitsFlowerDemo`
3. Create a venv: `python -m venv .venv`
4. Activate it: `source .venv/bin/activate` (Windows PowerShell: `.venv\Scripts\activate`)
5. Install: `pip install -e .`
6. Run the demo:

   ```bash
   flwr run . --stream --federation-config \
     "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
   ```

If you see `does not appear to be a Python project`, you are not in the repository root;
`cd` into the cloned `NoLimitsFlowerDemo` directory and retry.

## What you get

The run prints a prepare-round table, one line per federated round, the federated `theta*`,
the per-site contributions, an acceptance table, and a final `PASS:` line. By default it
federates the real warfarin PK data from `data/warfarin.csv`, no download needed.

The behaviour is controlled by `--run-config 'key=value ...'` knobs - `model` (`warfarin`,
`theophylline`, `warfarin-nn`, `orange`, `theoph-pooled`), `estimator` (`laplace`, `focei`,
`ghq`, `pooled`, `mcem` for the mixed-effects models; `mle`, `map` for the naive-pooled
`theoph-pooled`), and the differential-privacy `dp*` family. TOML strings need their own quotes inside the
shell quotes, e.g. `--run-config 'model="theophylline"'`. The full knob table is in the docs
[Overview](https://manuhuth.github.io/NoLimitsFlowerDemo/).

## Documentation

Full documentation is at **https://manuhuth.github.io/NoLimitsFlowerDemo/**:

- [Models](https://manuhuth.github.io/NoLimitsFlowerDemo/models/) - the four models, their real data, and the classical-vs-neural-on-identical-warfarin highlight.
- [Estimators](https://manuhuth.github.io/NoLimitsFlowerDemo/estimators/) - `laplace | focei | ghq | pooled` (mixed-effects), `mle | map` (naive-pooled), and `mcem` (nested: local E-step, federated M-step), when to use which.
- [Differential privacy](https://manuhuth.github.io/NoLimitsFlowerDemo/differential-privacy/) - the DP-Adam mode, its knobs, and the `(σ, T) → ε` accountant.
- [Optimizer & scipy interface](https://manuhuth.github.io/NoLimitsFlowerDemo/optimizer/) - the pure numpy/scipy server, one round per objective evaluation.
- [Architecture](https://manuhuth.github.io/NoLimitsFlowerDemo/architecture/) - prepare round, main-thread Julia warm-up, additivity as the exactness proof.

## License and data

The code is MIT, see `LICENSE`. The bundled datasets keep their own licenses and are not
covered by the MIT license (warfarin is GPL >= 3; Theoph and Orange come from R's base
`datasets`). See [`data/README.md`](data/README.md) for source, license and citation.
