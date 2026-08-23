# Architecture

Three files, one Flower app. `task.py` has **zero Flower imports** and is unit-testable
standalone.

```
server_app.py   ServerApp: prepare round, then L-BFGS-B over the summed site
                (value, gradient); demo-only pooled comparison after convergence
client_app.py   ClientApp: one site; Julia warmed at import; query.prepare builds and
                warms the DataModel, query answers theta with aggregates
task.py         the 4-model catalog (model string, real-data loader + column map,
                estimator, site count, acceptance kind), partitioning, theta glue,
                pooled reference + additivity probe (no Flower imports, unit-testable)
```

## A run is a prepare round, then pure evaluation rounds

**Round 0 — prepare.** The server broadcasts the run configuration to every site
(`server_app.prepare`). Each site (`client_app.prepare`) builds its own `DataModel` and
`FitContext`, burns one throwaway `objective_and_gradient` call at the model's default
`theta`, then replies with a ready flag, its setup wall time, its subject count, its
parameter names, and the model-default transformed `theta0`. This is where the one-off cost
lives — Julia boot plus model codegen plus the first evaluation, about 82 s per site, against
about 0.1 s for a warm round. Paying it in a round of its own keeps it out of optimization
round 1 and makes it visible:

```
PREPARE ROUND (3 sites)
  site   subjects      setup (s)
  0            10           77.3
  1            10           61.8
  2            10           62.1
```

The server then asserts every site is ready and that all sites report **identical** names
and `theta0` (`server_app.agree`) — they run the same model, so a mismatch means they do not,
and the summed objective would be meaningless. For the neural model this is what catches an
unpinned FFNN seed. The agreed `theta0` is the fit's start point, so **the server needs no
Julia at all** for the optimization.

**Every following round** is one L-BFGS-B objective evaluation: broadcast `theta`, each site
computes `objective_and_gradient` through its cached `FitContext` (2.6 ms per warm call on a
warfarin site), the server sums the values and gradients and hands them to L-BFGS-B — see
[Optimizer & scipy interface](optimizer.md).

## Client main-thread Julia warm-up

juliacall **cannot** cold-boot Julia from a non-main Python thread — the process hangs;
NoLimitsPy raises instead. The `ClientApp` module is imported on the main thread of its
process, so `client_app.py` boots Julia at **module level** (`nl.seval("1")` at import).
Every client process must do this, whatever thread the handlers later run on. The site
`DataModel` (and its `FitContext`) is cached in a **module global** keyed by
`(partition, num_partitions, model, data-source, seed)`: `context.state` holds records only,
and `ClientApp` objects are rebuilt per message, so a module global is the only place a live
Julia object survives across rounds.

The **ServerApp** runs on a worker thread and can therefore never boot Julia in-process. The
one remaining server-side Julia user — the demo's pooled reference fit — runs as a child
process (`python -m nolimits_flower.task fit ...`) with its output captured.

## The per-message client-process reality: sim vs deployment

- **Simulation.** Ray's ClientAppActors are pulled from an idle pool and are **not** pinned
  to a partition, so an actor can be handed a site it has not built yet and pay that site's
  build mid-fit. flwr 1.33 has no pinning knob, so the demo sizes the actor pool to **one**
  actor (`client-resources-num-cpus` equal to `init-args-num-cpus`), which pins by
  construction: that one actor prepares all three sites and then serves every round warm. The
  cost is a serial prepare round; the benefit is no re-warm mid-fit.
- **Deployment.** The question does not arise: **one SuperNode per site, one process, one
  `DataModel`, one `FitContext`**, and the prepare round absorbs the whole setup cost. Moving
  from the simulation to real institutions needs no change to the aggregation math — only a
  `flower-superlink` at the coordinator, one `flower-supernode` per site with TLS on and
  SuperNode auth, and site container images carrying Julia plus a precompiled NoLimits. This
  is the **deployable version, coming soon** — see the [Overview](index.md).

## Additivity is the exactness proof

Every wired estimator's objective is a **sum of per-subject terms**, and subjects are
**disjoint** across sites. So the sum over sites of `(value, gradient)` **is** the
pooled-data `(value, gradient)` — not an approximation — and the optimum of the summed
objective is the pooled `fit_model` optimum. Nothing couples two subjects, so where the
subjects physically live cannot change the total.

This is what the additivity probe checks to `1e-8` for all four models
(`task.additivity_probe`, exercised by `test_site_contributions_add_up`), and it is the
neural model's entire acceptance gate. It is also why the privacy layer
([Differential privacy](differential-privacy.md)) can be added without touching the math: DP
clips and noises the *per-subject* contributions, and SecAgg hides the *per-site* sum, but
the aggregate the optimizer consumes is the same summed quantity.

## Failure handling

A federated sum is only meaningful if every site is in it. If a site errors, becomes
unreachable, or returns a non-finite contribution (NoLimits reports `-Inf` on a failed
solve), the server aborts the whole fit with one actionable line naming the site, the node
and the site's own error message (`server_app._send_all`, `broadcast`, `_short_reason`). It
never sums the survivors and never reports a partial optimum. The `fail-site` run-config knob
exists solely to test this path and defaults to `-1` (off).
