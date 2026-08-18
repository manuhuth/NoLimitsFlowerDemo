# NoLimitsFlower

Federated nonlinear mixed-effects (NLME) estimation with [Flower](https://flower.ai) and
[NoLimitsPy](https://github.com/manuhuth/NoLimitsPy), the Python interface to
[NoLimits.jl](https://github.com/manuhuth/NoLimits.jl).

Sites hold their own subjects and never send subject-level data anywhere. At a common
parameter vector each site returns two aggregates: its marginal log-likelihood and that
value's gradient. Because the subjects are disjoint across sites, the sum of those site
contributions **is** the pooled marginal log-likelihood and its gradient, so the federated
optimum is the pooled `fit_model` optimum, not an approximation of it. This is exact
federated NLME, not an averaging heuristic.

## How it works

One L-BFGS-B objective evaluation is one federated round:

1. The server broadcasts the current transformed-scale theta to every site.
2. Each site computes `objective_and_gradient(Laplace(), dm_site, theta)` over its own
   subjects (the Laplace marginal likelihood finds its own empirical-Bayes modes) and
   replies with the scalar value and the gradient vector.
3. The server sums the values and the gradients and hands them to L-BFGS-B (`jac=True`).

Theta crosses the wire on the transformed (unconstrained) scale, so positivity constraints
stay implicit and the server needs no bounds. `max-rounds` caps the number of federated
rounds through the optimizer's `maxfun`; a truncated run fails the acceptance rather than
reporting a half-optimized theta as the optimum.

Estimators: `laplace` (default) and `ghq` (Gauss-Hermite quadrature, `ghq-level`). Both are
sums over subjects, hence exactly federable. SAEM and MCEM are not yet federated; they need
a per-site E-step sufficient-statistics primitive upstream in NoLimits.

## Equivalence, measured

Federated fit versus the pooled `fit_model` on the same simulated data, 24 subjects split
into 3 sites of 8, same model, same start point. Acceptance enforced by the run: objective
within 1e-6 relative, every natural-scale parameter within 1e-3 relative.

| data-seed | rounds | wall | federated loglik | pooled loglik | loglik rel.diff | worst parameter rel.diff |
|---|---|---|---|---|---|---|
| 20260818 | 14 | 53.8 s | -123.8749019813 | -123.8749019826 | 5.5e-12 | 2.7e-06 (omega) |
| 20260819 | 11 | 54.5 s | -119.2722299325 | -119.2722299326 | 2.1e-11 | 2.4e-06 |

Per-parameter for seed 20260818:

| parameter | federated | pooled | rel.diff |
|---|---|---|---|
| A0 | 9.66228678 | 9.66226779 | 2.0e-06 |
| k | 0.28862568 | 0.28862559 | 3.2e-07 |
| omega | 0.26601657 | 0.26601586 | 2.7e-06 |
| sigma | 0.55779218 | 0.55779281 | 1.1e-06 |

The residual parameter differences are optimizer tolerance, not federation error: the site
contributions themselves agree with the pooled quantity to floating-point precision.

## Quickstart

Prerequisites: Python 3.11 or newer, Julia 1.11 (the version juliapkg selects), and git.

```bash
python3 -m venv .venv
.venv/bin/pip install -e . "git+https://github.com/manuhuth/NoLimitsPy"
```

NoLimits' federation primitives (`objective_and_gradient`) are on NoLimits main and not yet
in a registered release, so the repo uses one shared pre-release Julia project
(`julia_env/`, gitignored):

```bash
julia +1.11 -e 'import Pkg; Pkg.activate("julia_env"); Pkg.add(url="https://github.com/manuhuth/NoLimits.jl", rev="main"); Pkg.add("PythonCall")'
```

Refresh it with `Pkg.update()` when new Julia fixes land on main. Run Python entry points
outside Flower with:

```bash
export PYTHON_JULIAPKG_PROJECT="$PWD/julia_env"
export PYTHON_JULIAPKG_OFFLINE=yes
```

Once NoLimits v0.2.6 is registered, both variables and the whole `julia_env/` step go away:
juliapkg then resolves a released NoLimits that already has the primitives.

Run the demo. Each site boots its own Julia (0.5 to 2 GB resident, one model compilation),
so cap the simulation concurrency:

```bash
flwr run . --stream --federation-config \
  "num-supernodes=3 client-resources-num-cpus=1 init-args-num-cpus=2"
```

`init-args-num-cpus=2` with one CPU per ClientApp gives 3 sites with 2 running at a time.
The run logs every round, the federated theta*, the per-site contributions, the acceptance
table above and a final `PASS:` line.

Run-config knobs (`--run-config 'key=value ...'`):

| key | default | meaning |
|---|---|---|
| `estimator` | `"laplace"` | `laplace`, or `ghq` for Gauss-Hermite quadrature |
| `ghq-level` | 5 | quadrature level when `estimator="ghq"` |
| `data-seed` | 20260818 | which simulated data set to federate |
| `max-rounds` | 100 | cap on federated rounds (L-BFGS-B `maxfun`) |
| `fail-site` | -1 | TESTING ONLY: that site id raises in its handler |

```bash
flwr run . --stream --run-config 'estimator="ghq" max-rounds=4' --federation-config ...
```

The `ghq` path optimizes and reports, but has no acceptance assertion yet.

## Architecture

```
server_app.py   ServerApp: L-BFGS-B over the summed site (value, gradient); acceptance check
client_app.py   ClientApp: one site, Julia warmed at import, answers theta with aggregates
task.py         model string, simulation, partitioning, theta glue, pooled reference
                (no Flower imports, so it is unit-testable without a federation)
```

Three constraints of the flwr 1.33 simulation runtime shape this code and are worth knowing
before changing it:

- **Client warm-up rule.** juliacall cannot cold-boot Julia from a non-main Python thread
  (the process hangs; NoLimitsPy raises instead). The ClientApp module is imported on the
  main thread of its ClientAppActor process, so `client_app.py` boots Julia at module level.
  Every client process must keep doing that, whatever thread handlers later run on. The site
  DataModel is cached in a module global keyed by (partition, seed): `context.state` holds
  records only, and ClientApp objects are rebuilt per message, so a module global is the only
  place a live Julia object survives across rounds.
- **Server worker-thread constraint.** The ServerApp runs on a worker thread
  (`server_th_with_start_checks`), so it can never boot Julia in-process.
- **Child-process pattern.** Everything server side that needs Julia (the pooled reference
  fit, the parameter names, the model's default start theta) runs as
  `python -m nolimits_flower.task fit ...`, a child process whose main thread is free, with
  its output **captured**. Letting the child inherit the simulation's log pipe and write
  Julia's chatter into it deadlocked the child.

Flower's per-run runtime environment does not pass `PYTHON_JULIAPKG_*` through to the
ServerApp and ClientApp processes, so `nolimits_flower/__init__.py` re-pins `julia_env/`
relative to the package before anything boots Julia. Without that, clients silently fall
back to the venv's own Julia project, which has the registered NoLimits and no
`objective_and_gradient`.

## Failure handling

A federated sum is only meaningful if every site is in it. If a site errors, becomes
unreachable, or returns a non-finite contribution (NoLimits reports `-Inf` on a failed
solve), the server aborts the whole fit with one actionable line naming the site, the node
and the site's own error message. It never sums the survivors and never reports a partial
optimum. Observed with `fail-site=1`:

```
ERROR: FEDERATED FIT ABORTED: round 1: site unknown (first round) (node 3963878865543444100)
       failed: fault injection: site 1 refuses to answer (error code 2) - aborting the
       federated fit rather than summing the remaining sites; see that node's ClientApp log
```

Site ids are learned from successful replies, because an error reply carries no content;
a site that fails in the very first round can only be named by its node id.

`fail-site` exists solely to test this path and is exercised by the slow test suite. Leave
it at its `-1` default.

## Tests

```bash
pytest tests -m "not slow" -q     # fast: partitioning, theta scales, config, error parsing
pytest tests -m slow -q -s        # federated fit plus the fault-injection abort
```

The fast tests need neither Julia nor a federation and run in seconds. The slow tests submit
real `flwr run` invocations and poll `flwr log`: about 3 minutes together on a laptop
(equivalence 111 s, fault injection 79 s). `.github/workflows/ci.yml` runs the fast tests on
every push and the slow suite with a 45 minute ceiling.

## Deployment outlook

The demo is a reproducible simulation on one machine. Running this across real institutions
needs no changes to the aggregation math, only infrastructure:

- A `flower-superlink` at the coordinator and one `flower-supernode` per site, with TLS on,
  `--insecure` off, and SuperNode authentication configured.
- Site container images that carry **Julia plus a precompiled NoLimits**, ideally as a
  sysimage: the current cost is one model compilation per client process, which dominates
  the wall clock of a short fit.
- The main-thread warm-up rule above holds in deployment too. A SuperNode's ClientApp
  process must boot Julia before Flower dispatches any handler.
- Per-site data loading replaces `task.simulate` and `task.partition`: each site points at
  its own table and builds the same model. The model string must be identical everywhere,
  since theta is matched positionally against the model's parameter axes.
- The FAB bundle is shipped to every SuperNode. `julia_env/` and `.venv/` are kept out of it
  by `.gitignore`, which `flwr build` honours; inspect the bundle after any flwr upgrade:
  `flwr build && python -c "import zipfile,glob;print(sorted(zipfile.ZipFile(sorted(glob.glob('*.fab'))[-1]).namelist()))"`.
- Round cost is one Laplace evaluation per site plus one message round trip, so wide-area
  latency is negligible next to the Julia work. Sites with very different subject counts
  make the slowest site the round's clock.

## Privacy roadmap

Federated is not private. Per-site log-likelihoods and gradients leak information about a
site's subjects across rounds, so "no subject data leaves the site" is a data-residency
claim, not a formal privacy guarantee. The planned hardening, in order:

1. **Secure aggregation** (SecAgg+, built into Flower): the server sees only the sum of site
   contributions, never per-site values, at no accuracy cost. The message protocol keeps
   site payloads as plain records so the SecAgg mod can wrap them without a redesign.
2. **Differential privacy**, gradient route: per-subject gradient contributions clipped to a
   norm C, summed, Gaussian noise added, with an accountant over rounds. This forces an
   optimizer change, since L-BFGS line searches break under noise, so the DP mode would use
   noisy full-batch gradient ascent or Adam. The NLME-specific work is the per-subject
   clipping semantics and the bias it introduces.
3. A DataSHIELD (R) port reusing the same aggregation math, whose disclosure filters are
   complementary to aggregate-level DP.

## License

MIT, see `LICENSE`.
