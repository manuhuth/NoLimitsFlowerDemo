# NoLimitsFlower

Federated nonlinear mixed-effects estimation with [Flower](https://flower.ai) and
[NoLimitsPy](https://github.com/manuhuth/NoLimitsPy). Subject-level data never leaves a
site; sites exchange only the marginal log-likelihood (and later its gradient) at a
common parameter vector.

Status: scaffold. The Flower round trip runs; NoLimits is not wired in yet.

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

## Run the demo

Each client boots its own Julia (0.5-2 GB), so cap simulation concurrency:

```bash
flwr run . --stream --federation-config \
  "num-supernodes=3 client-resources-num-cpus=1 init-args-num-cpus=2"
```

`init-args-num-cpus=2` with one CPU per ClientApp gives 3 sites, 2 running at a time.
