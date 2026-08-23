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

The default demo federates the **real** warfarin PK data from
[`nlmixr2data::warfarin`](https://cran.r-project.org/package=nlmixr2data) (CRAN, GPL-3):
the O'Reilly single-dose oral warfarin PK/PD study, committed verbatim to
`data/warfarin.csv` (515 rows, 32 subjects, long PK/PD with a `dvid` of `cp`/`pca`). The
demo's models are PK, so the loaders keep the plasma-concentration observations
(`dvid == "cp"`, `evid == 0`) and carry each subject's single dose (`amt`) as a constant
covariate, giving **251 concentration observations from 32 subjects**, mapped to the
model's columns as `ID=id`, `t=time`, `Dose=amt` (the per-subject dose, 60 to 153 mg) and
`conc=dv`. Subjects are split into 3 contiguous sites of 11/11/10.

The demo bundles small **real public** datasets so it is self-contained and offline: the
warfarin frame at `data/warfarin.csv` (nlmixr2data, GPL-3), the Theoph data at
`data/theoph.csv` (R's `datasets::Theoph`) and the Orange data at `data/orange.csv` (R's
`datasets::Orange`). Each dataset keeps its own license; see
[`data/README.md`](data/README.md). No network is needed after checkout.

The seeded synthetic data set is still there behind `data-source="simulated"` (warfarin
only): 24 subjects at TRUE_THETA, split 8/8/8. It powers the fast tests and the
fault-injection test, and it is the only mode with a known true theta.

## Models

A single run-config knob, `model`, selects one of four models. All four federate through
the same primitive - a per-subject sum, so the summed site contributions ARE the pooled
value and gradient exactly (**additivity**, checked to 1e-8 for every model). What differs
is the acceptance each model can support. The spread is deliberate: two classical PK
models, one **neural** mixed-effects model, and one **growth curve** - and the classical
warfarin PK model against a neural network on the *identical* warfarin data.

| `model` | data (all real) | kind | parameters | sites | acceptance |
|---|---|---|---|---|---|
| `warfarin` (default) | nlmixr2data warfarin, 32 subj / 251 obs | 1-cmt oral PK, closed-form ODE | 7 | 3 × 11/11/10 | strict: objective 1e-6, params 1e-3 |
| `theophylline` | R `Theoph`, 12 subj / 132 obs | 1-cmt oral PK, closed-form ODE | 7 | 3 × 4 | strict: objective 1e-6, params 1e-3 |
| `warfarin-nn` | same warfarin frame | **neural** mixed effects (FFNN mean) | 87 | 3 × 11/11/10 | **additivity gate** (1e-8) + reported objective agreement |
| `orange` | R `Orange`, 5 trees / 35 obs | logistic **growth** curve, algebraic | 5 | 2/2/1 | strict: objective 1e-6, params 1e-3 |

**warfarin / theophylline** are the same 1-compartment oral-absorption family (depot →
central, log-normal REs on `ka`, `cl`, `v`); the ODE is linear so NoLimits uses its
closed-form fast path. Both are fully identifiable, so the federated fit must match the
pooled `fit_model` on both objective and every natural-scale parameter.

**warfarin-nn** replaces the PK structure with a feed-forward network: the mean
concentration is `NN([d, t, eta], nn_params)` where `nn_params` is an `FFNNParameters`
block (a `(3, 5, 5, 5, 1)` tanh MLP, 86 weights) and `eta` is a per-subject random effect.
Its acceptance is different by necessity. The ~86 network weights are **non-identifiable**:
permutation and sign symmetries of the hidden units mean many distinct weight vectors give
the same predictions and the same likelihood, so two valid fits agree in objective and
predictions while differing in weights. Comparing parameters would be meaningless.
Federation is instead gated on the property federation is actually responsible for -
**additivity**: the sum over sites of `(value, gradient)` equals the pooled-data call at
theta0 to 1e-8 (this is exact, the headline claim for the neural model). The full federated
fit's objective is then compared to the pooled `fit_model` objective and *reported*, not
gated on parameters.

Two seeds matter for the neural model. The `FFNNParameters` **`seed` is pinned in the model
string** (`seed=1234`), so every site's Glorot-uniform weight initialization is identical;
without it theta0 would differ across sites and the prepare-round agreement check would
abort the run (which is the correct behaviour - unpinned, the sites are not fitting the same
model). The pooled reference fit uses the user's verified recipe (`Random.seed!(1234)`,
`Laplace()`, `pooled_init=true`). Because the weights are non-identifiable, the federated
fit is **warm-started from the pooled optimum** so the objective comparison is on the same
basin. That warm-start is **demo-only**: a real deployment has no pooled dataset and would
warm-start from a federated naive-pooled pass instead (naive-pooled is itself a per-subject
sum, so it federates the same way).

**orange** is the classic nlme Orange dataset - trunk circumference of 5 orange trees
against age - fit with a logistic growth curve `(Asym + eta) / (1 + exp((xmid - age) /
scal))` and a per-tree random effect on the asymptote. It is non-PK, non-ODE and fully
algebraic (the covariate `age` is referenced by name; the reserved-`t` gotcha only applies
to differential-equation state access). It shows the federation math is not tied to PK or
to ODEs. With only 5 trees the RE-variance estimate `omega` is uncertain, but additivity is
exact regardless of sample size and the fit still matches the pooled `fit_model` within the
strict tolerance (see *Equivalence, measured*).

## The warfarin model (baseline)

The default `model="warfarin"` is a warfarin population PK model: single oral dose, one compartment with
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
| `laplace` (default) | `Laplace()` | value 0.0, gradient 1.8e-16 | strict: objective 1e-6, every parameter 1e-3 | 0.10 s |
| `focei` | `FOCEI()` | value 1.7e-16, gradient 2.1e-16 | strict, same tolerances | 0.11 s |
| `ghq` | `GHQuadrature(level=ghq-level)` | value 2.0e-16, gradient 2.3e-16 (level 5) | one-sided: no worse than the pooled fit | 0.12 s (level 3), 0.13 s (level 5) |
| `pooled` | `Pooled()` | value 0.0, gradient 2.4e-16 | objective 1e-6, parameters 1e-2 | 0.11 s |

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

**Additivity** (the exact-FL property) on the real nlmixr2data warfarin data, 3 sites of
11/11/10 subjects, at the model's default theta. Summed site contributions vs the
pooled-data call, per estimator:

| `estimator` | pooled objective | value rel.diff | gradient rel.diff |
|---|---|---|---|
| `laplace` | -657.2007051064 | 0.0 | 1.8e-16 |
| `focei` | -655.8320382738 | 1.7e-16 | 2.1e-16 |
| `ghq`, level 5 | -1131.1546377109 | 2.0e-16 | 2.3e-16 |
| `pooled` | -2570.1705666892 | 0.0 | 2.4e-16 |

The federated **fit** (`laplace`, default) vs the pooled `fit_model` on the same data, same
start point (agreed in the prepare round), passes the strict gate (objective within 1e-6
relative, every natural-scale parameter within 1e-3). Pooled `laplace` fit, objective
-455.7966, with the federated fit matching it to the gate tolerance:

| parameter | value |
|---|---|
| ka | 0.5473 |
| cl | 0.1344 |
| v | 7.7002 |
| omega_ka | 0.4883 |
| omega_cl | 0.2837 |
| omega_v | 0.2200 |
| sigma | 1.0740 |

These estimates differ from the earlier Monolix-sourced numbers because this is a
differently-curated 32-subject frame (nlmixr2data), not a regression; they are sane
single-dose oral warfarin PK values (clearance ~0.13 L/h, volume ~7.7 L). The residual
federated-vs-pooled parameter differences are optimizer tolerance, not federation error:
the site contributions themselves are exact (the additivity table above).

The fit ends with `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`; the L-BFGS-B
exit flag is reported but nothing is gated on it - the acceptance table is.

## Quickstart

Prerequisites: Python 3.11 or newer, Julia 1.11 (the version juliapkg selects), and git.

```bash
python3 -m venv .venv
.venv/bin/pip install -e . "git+https://github.com/manuhuth/NoLimitsPy"
```

NoLimits' federation primitives (`objective_and_gradient`, `build_fit_context`) shipped in
**NoLimits v0.2.6**, so the shared Julia project (`julia_env/`, gitignored) now tracks the
registered release instead of `main`. Pinning it keeps every entry point on one env:

```bash
julia +1.11 -e 'import Pkg; Pkg.activate("julia_env");
                Pkg.add([Pkg.PackageSpec(name="NoLimits", version="0.2.6"),
                         Pkg.PackageSpec(name="PythonCall")])'
```

Use the same Julia minor version juliapkg selects for NoLimitsPy (`julia +1.11`); mixing
minor versions invalidates the precompile cache. Refresh with `Pkg.update()`. Run Python
entry points outside Flower with:

```bash
export PYTHON_JULIAPKG_PROJECT="$PWD/julia_env"
export PYTHON_JULIAPKG_OFFLINE=yes
```

Drop both variables and `julia_env/` entirely if you do not need the pinned env: juliapkg
then resolves NoLimits >= 0.2.5 from the registry by itself, which is what the deployable
counterpart of this demo does. All datasets are committed CSVs read with pandas, so no
Julia-side data download or `CSV` weak dependency is involved.

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
table above and a final `PASS:` line. It federates the real warfarin data by default, read
from the committed `data/warfarin.csv` (nlmixr2data, GPL-3); no download is needed.

Run-config knobs (`--run-config 'key=value ...'`):

| key | default | meaning |
|---|---|---|
| `model` | `"warfarin"` | `warfarin`, `theophylline`, `warfarin-nn` (neural) or `orange` (growth); see *Models* |
| `estimator` | `"laplace"` | `laplace`, `focei`, `ghq` (Gauss-Hermite quadrature) or `pooled` (naive-pooled plug-in); see *Estimators*. The non-warfarin models default to and are documented for `laplace` |
| `ghq-level` | 3 | quadrature level when `estimator="ghq"`; 1 to 3 is NoLimits' numerically stable range |
| `data-source` | `"warfarin"` | the real warfarin PK data, or `"simulated"` for the seeded synthetic set |
| `data-seed` | 20260818 | which simulated data set; ignored when `data-source="warfarin"` |
| `max-rounds` | 100 | cap on federated rounds (L-BFGS-B `maxfun`) |
| `fail-site` | -1 | TESTING ONLY: that site id raises in its handler |

Note the embedded quotes: `--run-config` values are TOML, so a string needs its own quotes
inside the shell quotes.

```bash
flwr run . --stream --run-config 'model="theophylline"' --federation-config ...
flwr run . --stream --run-config 'model="orange"' --federation-config ...
flwr run . --stream --run-config 'model="warfarin-nn" max-rounds=40' --federation-config ...
flwr run . --stream --run-config 'estimator="focei"' --federation-config ...
flwr run . --stream --run-config 'estimator="ghq" ghq-level=3 max-rounds=200' --federation-config ...
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
task.py         the 4-model CATALOG (model string, real-data loader + column map,
                estimator, site count, acceptance kind), partitioning, theta glue,
                pooled reference + additivity probe (no Flower imports, unit-testable)
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
pytest tests -m "not slow and not veryslow" -q   # fast: catalog, column maps, theta scales, config
pytest tests -m slow -q -s                        # additivity (all 4 models) + PK/growth fits + abort
pytest tests -m veryslow -q -s                    # the neural model (heaviest)
```

The fast tests (36) need neither Julia nor a federation and run in ~1 s: the 4-model
catalog selection, per-model column maps, unknown-model rejection, the FFNN-seed pin,
partitioning per primary id, the log-mask theta scaling, prepare-round agreement and error
parsing.

The slow tests are dominated by Julia boot and model compilation, not the federated loops:

- **Additivity** is parametrized over all four models (`test_site_contributions_add_up`):
  the sum over sites of `(value, gradient)` equals the pooled-data call to 1e-8 - for every
  estimator on the PK and growth models, and for `laplace` on the neural model. One Julia
  boot per model, no federation.
- **PK + growth fits** (`warfarin`, `theophylline`, `orange`) run the full federated fit and
  assert the ServerApp reached its strict acceptance (objective 1e-6, parameters 1e-3).
- The **neural model** is marked `veryslow` and round-capped (`max-rounds=40`): it runs a
  child additivity probe (the gate), a child pooled fit, and the warm-started federated fit.
  It asserts the additivity gate passed and the objective-agreement line was reported.
- **Fault injection** asserts a raising site aborts the whole fit.

`.github/workflows/ci.yml` runs the fast tests on every push and a subset of the slow suite;
the `veryslow` neural case can be skipped in CI and run on demand.

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

The code is MIT, see `LICENSE`. The **bundled datasets carry their own licenses** and are
not covered by the MIT license: `data/warfarin.csv` is GPL (>= 3) (from CRAN's
nlmixr2data), and `data/theoph.csv` / `data/orange.csv` come from R's base `datasets`
package (GPL-2 | GPL-3). All three are freely redistributable. Full source, license and
citation for each is in [`data/README.md`](data/README.md).
