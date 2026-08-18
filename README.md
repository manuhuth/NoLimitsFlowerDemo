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

## The model

The demo federates a warfarin population PK model: a single 100 mg oral dose, one
compartment with first-order absorption, and multiplicative log-normal random effects on
absorption rate, clearance and volume. It is the same shape as the theophylline example in
NoLimitsPy, with warfarin-typical values (`ka` 1/h, `cl` L/h, `v` L, concentrations mg/L).

```julia
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
```

Twenty-four subjects are sampled at 0.5, 1, 2, 4, 8, 24, 36, 48, 72, 96 and 120 h. The
random effects enter nonlinearly and the parameters mix scales (the structural parameters
are plain, the variance parameters are `scale=:log`), which is the realistic case for the
federated gradient: this is a genuine ODE mixed-effects fit, not a linear toy. The ODE is
linear in the states, so NoLimits takes its closed-form fast path and one site objective
plus gradient evaluation costs about 50 ms after compilation.

## How it works

A run is one prepare round followed by pure evaluation rounds.

**Round 0, prepare.** The server broadcasts the run configuration to every site. Each site
builds its own `DataModel` and burns one throwaway `objective_and_gradient` call at the
model's default theta, then replies with a ready flag, its setup wall time, its subject
count, its parameter names and the model-default transformed theta0. This is where the
one-off cost lives: Julia boot plus model codegen plus the first evaluation is about 85 s
per site, against about 0.05 s for a warm one. Paying it in a round of its own keeps it out
of optimization round 1 and makes it visible in the log:

```
PREPARE ROUND (3 sites)
  site   subjects      setup (s)
  0             8           84.9
  1             8           85.3
  2             8           82.4
```

Caveat specific to the *simulation* runtime: Ray's ClientAppActors are not pinned to a
node, so with fewer actors than sites (`init-args-num-cpus=2`, 3 sites) one actor serves
several partitions and pays a build for each partition it has not seen yet. Measured: after
a 170 s prepare round, round 1 still cost 129 s and every round after it 0.1 s. Give each
site its own actor (`init-args-num-cpus` >= number of sites) if you want the prepare round
to absorb all of it. In deployment the question does not arise: one SuperNode per site, one
process, one DataModel.

The server asserts that every site reports ready and that all sites report *identical*
names and theta0 (they run the same model, so a mismatch means they do not, and the summed
objective would be meaningless). The agreed theta0 is the fit's start point, so the server
needs no Julia at all for the optimization.

**Every following round** is one L-BFGS-B objective evaluation:

1. The server broadcasts the current transformed-scale theta to every site.
2. Each site computes `objective_and_gradient(Laplace(), dm_site, theta)` over its own
   subjects (the Laplace marginal likelihood finds its own empirical-Bayes modes) on its
   already-warm DataModel and replies with the scalar value and the gradient vector.
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
| 20260818 | 45 | 286.8 s | -324.1753622672 | -324.1753625815 | 9.7e-10 | 1.2e-04 (omega_ka) |
| 20260819 | 43 | 302.1 s | -335.5802494858 | -335.5802492634 | 6.6e-10 | 1.0e-04 (omega_v) |

Per-parameter for seed 20260818:

| parameter | federated | pooled | rel.diff |
|---|---|---|---|
| ka | 1.00529192 | 1.00530383 | 1.2e-05 |
| cl | 0.11996099 | 0.11996090 | 7.1e-07 |
| v | 8.01790190 | 8.01790057 | 1.7e-07 |
| omega_ka | 0.39776485 | 0.39781141 | 1.2e-04 |
| omega_cl | 0.40613069 | 0.40612379 | 1.7e-05 |
| omega_v | 0.23978401 | 0.23978801 | 1.7e-05 |
| sigma | 0.48382927 | 0.48382740 | 3.9e-06 |

Wall is the federated loop only (the rounds), excluding the one-off model compilation in
each site process and the pooled reference fit. The additivity of the site quantities is
exact: at the true theta the three site log-likelihoods sum to the pooled value with
relative difference 1.7e-16, and the summed gradients match the pooled gradient to 1.1e-14.

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
server_app.py   ServerApp: prepare round, then L-BFGS-B over the summed site
                (value, gradient); demo-only pooled comparison after convergence
client_app.py   ClientApp: one site; Julia warmed at import; `query.prepare` builds and
                warms the DataModel, `query` answers theta with aggregates
task.py         model string, simulation, partitioning, theta glue, pooled reference
                (no Flower imports, so it is unit-testable without a federation)
```

The server side of the fit is Julia-free: names and the start theta come from the prepare
round, not from a local model build. The pooled `fit_model` reference still runs, but only
after convergence and only to produce the demo's acceptance table - a production deployment
has no pooled dataset and deletes that call.

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
- **Child-process pattern.** The one remaining server-side Julia user, the demo's pooled
  reference fit, runs as `python -m nolimits_flower.task fit ...`, a child process whose main
  thread is free, with its output **captured**. Letting the child inherit the simulation's
  log pipe and write Julia's chatter into it deadlocked the child.

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
real `flwr run` invocations and poll `flwr log`: 793 s together on a laptop, dominated by the
equivalence run (a 45 round federated fit plus the pooled reference fit, with one model
compilation per site process). `.github/workflows/ci.yml` runs the fast tests on every push
and the slow suite with a 45 minute ceiling.

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
