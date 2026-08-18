# NoLimitsFlower

Federated nonlinear mixed-effects estimation with [Flower](https://flower.ai) and
[NoLimitsPy](https://github.com/manuhuth/NoLimitsPy). Subject-level data never leaves a
site; sites exchange only the marginal log-likelihood (and later its gradient) at a
common parameter vector.

Status: Phase 2. `flwr run .` runs one verification round: the server broadcasts the true
simulation theta (transformed scale), each of 3 sites answers with
`objective_and_gradient(Laplace(), dm_site, theta)`, and the summed value and gradient are
compared against the same call on the pooled data set. Observed on the seeded demo data:

```
sites=3 summed value=-126.0322291370 pooled value=-126.0322291370 rel=4.510e-16
gradient rel-diffs=[1.50e-16 0 0 0] (worst 1.503e-16)
```

The federated optimizer loop is Phase 3.

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
