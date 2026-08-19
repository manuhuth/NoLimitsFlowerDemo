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

## The data

The default demo federates the **real** warfarin PK data, loaded through NoLimits' own
`load_warfarin_from_monolix()`: the Monolix tutorial dataset of single-dose oral warfarin,
32 dosed subjects of which the loader returns the 30 that carry its required baseline
record. The PK rows are the ones with a non-missing concentration `C`, giving **227
observations from 30 subjects**, mapped to the model's columns as `ID=id`, `t`, `Dose=d`
(the per-subject dose, 60 to 153 mg) and `conc=C`. Subjects are split into 3 contiguous
sites of 10.

The first run downloads the file and caches the raw frame at `data/warfarin.csv`
(gitignored); every later run and every test reads the cache, so repeat runs are offline.

The seeded synthetic data set is still there behind `data-source="simulated"`: 24 subjects
at TRUE_THETA, split 8/8/8. It powers the fast tests and the fault-injection test, and it
is the only mode with a known true theta, so it is where the additivity of the site
quantities is checked.

## The model

The model is a warfarin population PK model: single oral dose, one compartment with
first-order absorption, and multiplicative log-normal random effects on absorption rate,
clearance and volume (`ka` 1/h, `cl` L/h, `v` L, concentrations mg/L).

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

The real data are sampled irregularly from 0.5 to 120 h; the simulated mode uses 0.5, 1,
2, 4, 8, 24, 36, 48, 72, 96 and 120 h. The random effects enter nonlinearly and the parameters mix scales (the structural parameters
are plain, the variance parameters are `scale=:log`), which is the realistic case for the
federated gradient: this is a genuine ODE mixed-effects fit, not a linear toy. The ODE is
linear in the states, so NoLimits takes its closed-form fast path and one site objective
plus gradient evaluation costs about 50 ms after compilation.

## How it works

A run is one prepare round followed by pure evaluation rounds.

**Round 0, prepare.** The server broadcasts the run configuration to every site. Each site
builds its own `DataModel` and `FitContext` and burns one throwaway `objective_and_gradient` call at the
model's default theta, then replies with a ready flag, its setup wall time, its subject
count, its parameter names and the model-default transformed theta0. This is where the
one-off cost lives: Julia boot plus model codegen plus the first evaluation is about 82 s
per site, against about 0.1 s for a warm round (2.6 ms of it the actual site call, the
rest messaging). Paying it in a round of its own keeps it out
of optimization round 1 and makes it visible in the log:

```
PREPARE ROUND (3 sites)
  site   subjects      setup (s)
  0            10           77.3
  1            10           61.8
  2            10           62.1
```

Caveat specific to the *simulation* runtime: Ray's ClientAppActors are pulled from an idle
pool and are **not** pinned to a partition, so with several actors an actor can be handed a
site it has not built yet and pays that site's build mid-fit. Measured on the warfarin run
with one actor per site (`client-resources-num-cpus=1 init-args-num-cpus=3`): after a 108 s
prepare round the warm rounds cost 0.10 to 0.22 s, but rounds 1, 3 and 4 still cost about
63 s each. flwr 1.33 has no pinning knob (the pool holds
`floor(available_cpus / client-resources-num-cpus)` interchangeable actors), so the demo
sizes the pool to **one** actor - `client-resources-num-cpus` equal to
`init-args-num-cpus` - which pins by construction: that actor prepares all three sites and
then serves every round warm. Cost of the choice: the three site setups run sequentially, so
the prepare round is 226 s instead of 108 s, and each round evaluates the sites one after
another (0.10 s in total here, since a warm site call is 2.6 ms). End to end that is still
230 s against 372 s. In deployment the question does not arise: one SuperNode per site, one
process, one DataModel, one FitContext, and the prepare round absorbs the whole setup cost.

The server asserts that every site reports ready and that all sites report *identical*
names and theta0 (they run the same model, so a mismatch means they do not, and the summed
objective would be meaningless). The agreed theta0 is the fit's start point, so the server
needs no Julia at all for the optimization.

**Every following round** is one L-BFGS-B objective evaluation:

1. The server broadcasts the current transformed-scale theta to every site.
2. Each site computes `objective_and_gradient(method, ctx_site, theta)` over its own
   subjects (with `Laplace()`, the marginal likelihood finds its own empirical-Bayes modes) through
   the `FitContext` built in the prepare round, and replies with the scalar value and the
   gradient vector. The context caches the random-effect batch infos and the evaluation
   cache: on one warfarin site that is 2.6 ms per call against 17.6 ms for the equivalent
   `DataModel` call, which rebuilds them every time. Both forms return the same numbers.
3. The server sums the values and the gradients and hands them to L-BFGS-B (`jac=True`).

The server optimizes a **preconditioned** coordinate z with theta = theta0 + s * z (so the
gradient it reports is `s * grad_theta`). The scale s follows NoLimits' own rule, mirrored
in `task.precondition_scale` from `_precondition_scale` / `_precondition_maps` in
NoLimits.jl `src/estimation/common.jl`: `s_i = max(|theta0_i|, 1)` for a coordinate on the
identity scale, 1 for a log-scaled one. Here only `v` (about 8 L) differs from 1, and that
one number is worth 85 rounds down to 29 - the raw scale also made L-BFGS-B exit with the
cosmetic `ABNORMAL` flag, the preconditioned one converges cleanly.

Theta crosses the wire on the transformed (unconstrained) scale, so positivity constraints
stay implicit and the server needs no bounds. `max-rounds` caps the number of federated
rounds through the optimizer's `maxfun`; a truncated run fails the acceptance rather than
reporting a half-optimized theta as the optimum.

## Estimators

Four estimators are wired, all through the same `objective_and_gradient(method, ctx, theta)`
protocol, so a site's handler is estimator-agnostic. Every one of them is a **sum over
subjects**, so the summed site contributions are the pooled-data value and gradient exactly.
Measured on the three warfarin sites at the model's default theta
(`python -m nolimits_flower.task probe`, one Julia boot, all four estimators):

| `estimator` | NoLimits method | additivity of (value, gradient) | fit acceptance | per-round cost |
|---|---|---|---|---|
| `laplace` (default) | `Laplace()` | value 0.0, gradient 2.1e-16 | strict: objective 1e-6, every parameter 1e-3 | 0.10 s |
| `focei` | `FOCEI()` | value 2.0e-16, gradient 2.1e-16 | strict, same tolerances | 0.11 s |
| `ghq` | `GHQuadrature(level=ghq-level)` | value 2.3e-16, gradient 5.9e-16 (level 5) | one-sided: no worse than the pooled fit | 0.12 s (level 3), 0.13 s (level 5) |
| `pooled` | `Pooled()` | value 1.1e-16, gradient 1.5e-16 | objective 1e-6, parameters 1e-2 | 0.11 s |

The three sums are exact for the same reason in every case: subjects are independent, and
each estimator's objective is a per-subject (per-random-effect-batch) term. `Laplace` and
`FOCEI` find each subject's empirical-Bayes mode from that subject's own data; `GHQuadrature`
integrates each subject's batch on its own quadrature grid; `Pooled` plugs in a per-subject
eta and evaluates a per-subject likelihood. Nothing in any of them couples two subjects, so
splitting the subjects across sites cannot change the total.

Two of the four need a caveat, and both caveats are about the *optimizer*, not the sums:

- **`pooled` is exact here, but that is model-conditional.** `Pooled()` first *calibrates* a
  plug-in strategy per random effect (`_pooled_plugin_strategies` in NoLimits'
  `src/estimation/pooled.jl`) and then optimizes `loglikelihood(dm, theta, eta(theta))`. The
  calibration looks at the data set: it probes the first individual's random-effect
  distributions, and demotes a strategy (`:mean` to `:median` to `:zero`, or to Monte-Carlo
  draws) if it is not ForwardDiff-safe there. In this model every random effect is
  `LogNormal`, whose mean is finite and smooth, so every site resolves to `:mean` and the
  plug-in eta is `exp(omega^2/2)`, a function of **theta alone** - identical on every site,
  hence exact additivity (1.1e-16). A model where the resolution depends on the data (a
  normalizing-flow random effect, or a strategy demoted on one site's data only) would
  calibrate per site and the federated objective would stop being the pooled-data objective.
  Re-run the probe after changing the model; the slow test does exactly that.
  Its acceptance also uses a looser 1e-2 parameter tolerance: the plug-in eta depends on
  omega only through `exp(omega^2/2)`, so the objective is nearly flat in the omegas. The two
  fits agree to 3.1e-10 in the objective while `omega_cl` differs by 6.2e-03 - a plateau, not
  a federation error.
- **`ghq` gets a one-sided gate.** The quadrature objective is rough on this model: NoLimits
  warns that levels above 3 can cancel in the signed logsumexp, and it falls back to the
  level-1 rule for a batch when the prior-centred rule goes unstable. scipy's L-BFGS-B and
  `fit_model`'s Optim LBFGS therefore settle in *different* local optima, and neither side
  wins consistently - measured against the pooled `fit_model` reference, the federated
  optimum is 5.6e-02 **better** at level 3 and 6.7e-02 worse at level 5. A parameter-wise
  gate is unreachable in either direction, so the run asserts what federation is actually
  responsible for: the summed objective is the pooled objective (2.3e-16), and optimizing it
  loses nothing, i.e. the federated optimum is no worse than the pooled one. The default
  `ghq-level` is therefore 3, NoLimits' own default and its documented stable range; level 5
  is reported but not gated. GHQ also needs more rounds than the others (127 at level 3
  against 34 for FOCEI), so raise `max-rounds` above its default 100 for it.

SAEM and MCEM are not federated; they need a per-site E-step sufficient-statistics primitive
upstream in NoLimits. `MLE` and `MAP` have the protocol too but require a model without
random effects, which is not what this package is for.

## Equivalence, measured

Federated fit versus the pooled `fit_model` on the same data, same model, same start point
(the model's default theta, agreed in the prepare round). Acceptance enforced by the run:
objective within 1e-6 relative, every natural-scale parameter within its estimator's
tolerance from the table above.

Per estimator, on the real warfarin data, 3 sites of 10 subjects (`ghq` at level 3, which is
the default; the run's own `max-rounds` raised to 200 for it):

| `estimator` | rounds | loop wall | federated loglik | pooled loglik | loglik rel.diff | worst parameter rel.diff | verdict |
|---|---|---|---|---|---|---|---|
| `laplace` | 29 | 3.5 s | -403.2872869375 | -403.2872858633 | 2.7e-09 | 1.8e-04 (omega_v) | PASS (strict) |
| `focei` | 34 | 3.8 s | -401.2000213813 | -401.1999998500 | 5.4e-08 | 8.9e-04 (omega_v) | PASS (strict) |
| `ghq`, level 3 | 127 | 15.4 s | -394.5270561672 | -417.8769948600 | 5.6e-02 (federated better) | 9.8e-01 | PASS (one-sided) |
| `ghq`, level 5 | 67 | 6.9 s | -433.2232828373 | -406.1255437100 | 6.7e-02 (federated worse) | 1.5e-01 | reported, gate fails |
| `pooled` | 28 | 4.0 s | -473.9759462922 | -473.9759461400 | 3.1e-10 | 6.2e-03 (omega_cl) | PASS (1e-2 parameters) |

Loop wall excludes the prepare round (48 to 132 s, one model compilation plus three
`DataModel` builds in a single Ray actor) and the pooled reference fit. Per-round cost is
almost identical across estimators (0.10 to 0.14 s, of which a few ms is the actual site
call), so what separates them is the number of rounds their objective needs. The `ghq`
level-5 row is the documented failure of the *fit* comparison, not of the federation: its
additivity is 2.3e-16 and both sides simply converge to different local optima of a rough
quadrature objective, with `fit_model` winning at level 5 and the federated fit winning at
level 3.

| data-source | sites | subjects | rounds | wall | federated loglik | pooled loglik | loglik rel.diff | worst parameter rel.diff |
|---|---|---|---|---|---|---|---|---|
| `warfarin` (real) | 3 | 10/10/10 | 29 | 3.5 s | -403.2872869375 | -403.2872858633 | 2.7e-09 | 1.8e-04 (omega_v) |
| `simulated` | 3 | 8/8/8 | 33 | 3.3 s | -324.1753626167 | -324.1753625815 | 1.1e-10 | 1.4e-05 (omega_cl) |

Before preconditioning and actor pinning the same two runs took 85 rounds / 263.9 s and 45
rounds / 199.5 s, with individual rounds up to 63 s:

| run | rounds | loop wall | slowest round | L-BFGS-B exit |
|---|---|---|---|---|
| warfarin, before | 85 | 263.9 s | ~63 s | ABNORMAL_TERMINATION_IN_LNSRCH |
| warfarin, after | 29 | 3.5 s | 0.22 s | CONVERGENCE |
| simulated, before | 45 | 199.5 s | ~63 s | ABNORMAL_TERMINATION_IN_LNSRCH |
| simulated, after | 33 | 3.3 s | 0.11 s | CONVERGENCE |

Per-parameter on the real warfarin data:

| parameter | federated | pooled | rel.diff |
|---|---|---|---|
| ka | 0.56804852 | 0.56807048 | 3.9e-05 |
| cl | 0.12815179 | 0.12815181 | 1.7e-07 |
| v | 7.80891424 | 7.80893398 | 2.5e-06 |
| omega_ka | 0.47452538 | 0.47450662 | 4.0e-05 |
| omega_cl | 0.23369895 | 0.23366642 | 1.4e-04 |
| omega_v | 0.22572954 | 0.22568894 | 1.8e-04 |
| sigma | 1.04295630 | 1.04296538 | 8.7e-06 |

Wall is the federated loop only (the rounds), excluding the 226 s prepare round and the
pooled reference fit. Every round is warm (0.10 to 0.22 s): with a single-actor pool no
round after prepare re-pays a DataModel build, see the caveat under *How it works*.

The residual parameter differences are optimizer tolerance, not federation error: the site
contributions themselves are exact. At the true theta of the simulated data the three site
log-likelihoods sum to the pooled value with relative difference 1.7e-16 and the summed
gradients match the pooled gradient to 1.1e-14; on the real warfarin data the same holds for
all four estimators (the additivity table above).

Both runs now end with `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`; on the raw
transformed scale the same fits ended with the cosmetic `ABNORMAL_TERMINATION_IN_LNSRCH`
flag. The flag is reported but nothing is gated on it - the acceptance table is.

## Quickstart

Prerequisites: Python 3.11 or newer, Julia 1.11 (the version juliapkg selects), and git.

```bash
python3 -m venv .venv
.venv/bin/pip install -e . "git+https://github.com/manuhuth/NoLimitsPy"
```

NoLimits' federation primitives (`objective_and_gradient`, `build_fit_context`) shipped in
**NoLimits v0.2.6**, so the shared Julia project (`julia_env/`, gitignored) now tracks the
registered release instead of `main`. It stays because the warfarin loader needs `CSV` as a
weak dependency and because pinning it keeps every entry point on one env:

```bash
julia +1.11 -e 'import Pkg; Pkg.activate("julia_env");
                Pkg.add([Pkg.PackageSpec(name="NoLimits", version="0.2.6"),
                         Pkg.PackageSpec(name="PythonCall"), Pkg.PackageSpec(name="CSV")])'
```

Use the same Julia minor version juliapkg selects for NoLimitsPy (`julia +1.11`); mixing
minor versions invalidates the precompile cache. Refresh with `Pkg.update()`. Run Python
entry points outside Flower with:

```bash
export PYTHON_JULIAPKG_PROJECT="$PWD/julia_env"
export PYTHON_JULIAPKG_OFFLINE=yes
```

Drop both variables and `julia_env/` entirely if you do not need the `CSV` warfarin loader:
juliapkg then resolves NoLimits >= 0.2.5 from the registry by itself, which is what the
deployable counterpart of this demo does.

Run the demo. Each site boots its own Julia (0.5 to 2 GB resident, one model compilation),
so cap the simulation concurrency:

```bash
flwr run . --stream --federation-config \
  "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
```

Equal `client-resources-num-cpus` and `init-args-num-cpus` size the Ray actor pool to one
actor, which serves all three sites: one Julia, one model compilation, and no site build
after the prepare round (see the caveat above). Setting `client-resources-num-cpus=1`
instead gives one actor per site and a faster prepare round, at the price of ~60 s
re-warm rounds mid-fit. The run logs the prepare
table, every round, the federated theta*, the per-site contributions, the acceptance
table above and a final `PASS:` line. It federates the real warfarin data by default; the
first run downloads it and writes the `data/warfarin.csv` cache.

Run-config knobs (`--run-config 'key=value ...'`):

| key | default | meaning |
|---|---|---|
| `estimator` | `"laplace"` | `laplace`, `focei`, `ghq` (Gauss-Hermite quadrature) or `pooled` (naive-pooled plug-in); see *Estimators* |
| `ghq-level` | 3 | quadrature level when `estimator="ghq"`; 1 to 3 is NoLimits' numerically stable range |
| `data-source` | `"warfarin"` | the real warfarin PK data, or `"simulated"` for the seeded synthetic set |
| `data-seed` | 20260818 | which simulated data set; ignored when `data-source="warfarin"` |
| `max-rounds` | 100 | cap on federated rounds (L-BFGS-B `maxfun`) |
| `fail-site` | -1 | TESTING ONLY: that site id raises in its handler |

Note the embedded quotes: `--run-config` values are TOML, so a string needs its own quotes
inside the shell quotes.

```bash
flwr run . --stream --run-config 'estimator="focei"' --federation-config ...
flwr run . --stream --run-config 'estimator="ghq" ghq-level=3 max-rounds=200' --federation-config ...
flwr run . --stream --run-config 'estimator="pooled"' --federation-config ...
```

The estimator-agnostic additivity check needs no federation and boots Julia once for all
four estimators:

```bash
python -m nolimits_flower.task probe        # per-site sums vs the pooled-data call
```

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
  DataModel (and its FitContext, on the Julia side) is cached in a module global keyed by
  (partition, number of partitions, data source, seed): `context.state` holds
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

A local `flwr run` starts a `flower-superlink`/`flower-superexec` pair that **outlives the
run**. They are harmless but they hold the control API port, so kill them (`pkill -f
flower-superlink`) before switching branches or debugging a run that seems to hang.

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
pytest tests -m slow -q -s        # additivity of all four estimators, the laplace
                                  # federated fit, and the fault-injection abort
```

The fast tests need neither Julia nor a federation and run in 0.4 s (24 tests: partitioning,
theta scales, estimator-name validation, prepare-round agreement, error parsing). The three
slow tests took 957 s together on a laptop (360 s laplace equivalence, 346 s the four-estimator
additivity probe, 251 s the fault injection), dominated by Julia boot and model compilation -
the federated loops themselves are seconds. The additivity probe is the one slow test that
needs no federation: one Julia boot, all four estimators. The per-estimator *fit* acceptance
runs (the table above) stay a report-level verification; CI keeps only the laplace
equivalence run. `.github/workflows/ci.yml` runs the fast tests on every push and the slow
suite with a 45 minute ceiling.

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
