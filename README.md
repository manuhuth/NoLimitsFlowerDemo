# NoLimitsFlower

Federated nonlinear mixed-effects estimation with [Flower](https://flower.ai) and
[NoLimitsPy](https://github.com/manuhuth/NoLimitsPy). Subject-level data never leaves a
site; sites exchange only the marginal log-likelihood (and later its gradient) at a
common parameter vector.

Status: Phase 3. `flwr run .` runs the federated fit. The server optimizes the
transformed-scale theta with scipy L-BFGS-B (`jac=True`); one objective evaluation is one
federated round, in which every site answers with
`objective_and_gradient(Laplace(), dm_site, theta)` and the server sums the values and
gradients. Because the sites hold disjoint subjects, that sum IS the pooled marginal
log-likelihood and its gradient, so the optimum is the pooled `fit_model` optimum. The
server never runs Julia; the pooled reference fit runs in a child process.

Observed on the seeded demo data (3 sites, 8 subjects each):

```
seed 20260818: 14 rounds, 55.0 s   loglik -123.8749019826  (pooled -123.8749019827, rel 1.6e-11)
               A0 9.66228678 / 9.66226779   k 0.28862568 / 0.28862559
               omega 0.26601657 / 0.26601586   sigma 0.55779218 / 0.55779281   worst rel 2.7e-06
seed 20260819: 11 rounds, 54.5 s   loglik -119.2722299325  (pooled -119.2722299326, rel 2.1e-11)
               worst parameter rel.diff 2.4e-06
```

Acceptance thresholds enforced by the run: objective within 1e-6 relative, every
natural-scale parameter within 1e-3 relative.

## Development setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e . -e ../NoLimitsPy
```

NoLimits' federation primitives are not in a registered release yet, so the repo uses one
shared pre-release Julia project (`julia_env/`, gitignored). Build it with the same Julia
minor version juliapkg selects (1.11):

```bash
julia +1.11 -e 'import Pkg; Pkg.activate("julia_env"); Pkg.add(url="https://github.com/manuhuth/NoLimits.jl", rev="main"); Pkg.add("PythonCall")'
```

Refresh it with `Pkg.update()` when new Julia fixes land on main. Run every Python entry
point with:

```bash
export PYTHON_JULIAPKG_PROJECT="$PWD/julia_env"
export PYTHON_JULIAPKG_OFFLINE=yes
```

Drop both once the primitives ship in a registered NoLimits release.

Flower's per-run runtime environment does not pass `PYTHON_JULIAPKG_*` on to the
ServerApp/ClientApp processes, so `nolimits_flower/__init__.py` re-pins `julia_env/`
(resolved relative to the package) before anything boots Julia. Without that, clients fall
back to the venv's own Julia project, which has the registered NoLimits and no
`objective_and_gradient`.

Julia boot rules observed in the flwr 1.33 simulation runtime:

- The ClientApp module is imported on the **main thread** of its ClientAppActor process and
  the query handler also runs on that main thread, so the module-level warm-up in
  `client_app.py` is enough; no fallback was needed.
- The ServerApp is imported and run on a worker thread
  (`Thread-9 (server_th_with_start_checks)`), so it can never boot Julia itself. Its pooled
  reference runs as a child process (`python -m nolimits_flower.task`) whose output is
  captured - letting the child write Julia's chatter into the inherited log pipe deadlocked
  it.

## Run the demo

Each client boots its own Julia (0.5-2 GB), so cap simulation concurrency:

```bash
flwr run . --stream --federation-config \
  "num-supernodes=3 client-resources-num-cpus=1 init-args-num-cpus=2"
```

`init-args-num-cpus=2` with one CPU per ClientApp gives 3 sites, 2 running at a time.

Run config knobs (`--run-config`): `data-seed` (the simulated data set; a second seed must
also pass the acceptance), `estimator` (`laplace`, or `ghq` with `ghq-level`), `max-rounds`
(the federated-round cap, passed to L-BFGS-B as `maxfun`). The `ghq` path optimizes but has
no acceptance assertion yet:

```bash
flwr run . --stream --run-config 'estimator="ghq" max-rounds=4' --federation-config ...
```
