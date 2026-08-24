"""ClientApp: answers a transformed-scale theta with this site's (value, gradient).

Two handlers:
- `query.prepare` (once per run, before the optimizer) builds this site's DataModel
  and its FitContext, and burns ONE warm-up objective call, so the ~85 s of Julia boot + model codegen +
  DataModel build is paid in a round of its own instead of hiding inside optimization
  round 1. It also reports this site's parameter names and the model-default
  transformed theta0, which is where the server gets its start point from.
- `query.default` (every optimizer round) is then a pure warm evaluation, ~0.05 s.

Julia is booted at MODULE IMPORT, i.e. at client process start, because juliacall
cannot cold-boot Julia from a non-main thread (NoLimitsPy raises instead of
hanging). The site DataModel is cached in a module global: `context.state` only
holds records, and ClientApp objects are rebuilt per message, so a module global
is the only place a live Julia object survives across rounds within one process.
"""

import threading
import time
from logging import INFO

import numpy as np
import NoLimitsPy as nl
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.logger import log

from nolimits_flower import task

BOOT_THREAD = threading.current_thread().name
BOOT_ON_MAIN = threading.current_thread() is threading.main_thread()
nl.seval("1")  # cold boot happens here, whatever thread later handlers run on
log(INFO, "NoLimitsPy booted on thread %r (main=%s)", BOOT_THREAD, BOOT_ON_MAIN)

app = ClientApp()

_site_dms: dict[tuple, object] = {}
_site_subjects: dict[tuple, int] = {}
# MCEM (nested EM) per-site caches: the warm-start sampler state threads across OUTER
# iterations, and one outer iteration's draws are HELD FIXED across its M-step rounds. These
# live in module globals for the same reason the DataModel does - ClientApp objects are
# rebuilt per message, so only a module global survives across rounds within one process.
_site_mcem_state: dict[tuple, object] = {}
_site_mcem_draws: dict[tuple, object] = {}  # (site_key, outer_iter) -> draws
_site_mcem_parts: dict[tuple, tuple] = {}   # site_key -> (q1_names, q2_names)


def _site_key(context: Context) -> tuple:
    return (
        int(context.node_config["partition-id"]),
        int(context.node_config["num-partitions"]),
        str(context.run_config["model"]),
        str(context.run_config["data-source"]),
        int(context.run_config["data-seed"]),
    )


def _site_dm(context: Context):
    """This site's DataModel, built once per (partition, model, data source) per process."""
    key = _site_key(context)
    if key not in _site_dms:
        pid, num, model, source, seed = key
        pid_col = task.spec(model).primary_id
        df = task.partition(task.dataset(model, source, seed, nl), num, pid_col)[pid]
        _site_dms[key] = task.build_data_model(nl, model, df)
        _site_subjects[key] = int(df[pid_col].nunique())
    return _site_dms[key]


def _dp_contribution(context: Context, config, theta):
    """This site's NOISED release for one dp round: (vector, its clipping bound).

    Nothing un-noised leaves this function, and nothing here is logged: the per-subject
    values and gradients it computes are the raw material the clipping bounds.

    MAP under DP: the data part is the per-subject MLE (clipped and noised like any site); the
    prior is PUBLIC and data-independent, so the carrier (site 0) adds it as an un-clipped,
    un-noised offset AFTER the data aggregation. The prior never touches the accountant, so
    eps(map+dp) == eps(mle+dp) at matched knobs.
    """
    dm = _site_dm(context)
    estimator = str(config["estimator"])
    site_id = int(context.node_config["partition-id"])
    is_prior_carrier = estimator == "map" and site_id == 0
    values, gradients, _ = task.dp_batch_contributions(
        nl, dm, theta, estimator, int(config["ghq-level"])
    )
    sigma, sites = float(config["dp-noise-multiplier"]), int(config["dp-sites"])
    if str(config["dp-release"]) == "value":
        # The final objective, under its own per-subject clipping bound and budget charge.
        bound = float(config["dp-value-clip"])
        total = float(np.clip(values, -bound, bound).sum())
        release = np.array([total]) + task.dp_noise(1, bound, sigma, sites)
        if is_prior_carrier:
            release = release + task.map_prior(nl, dm, theta)[0]  # public log-prior value
        return release, bound
    # Clip in the coordinate the server's Adam steps in: transformed axes times the
    # preconditioning scale s (public, model-derived), so the noise is calibrated against
    # exactly the vector the optimizer uses.
    precond = np.asarray(config["dp-precond"], dtype=float)
    gradients = gradients * precond[None, :]
    if str(config.get("dp-clip-mode", "per-group")) == "per-group":
        # per-group clipping; bound is C_total = sqrt(sum C_g^2), noise isotropic at
        # sigma*C_total, so the accounting is identical to joint at C_total.
        group_ids = list(config["dp-group-ids"])
        group_clips = list(config["dp-group-clips"])
        bound = task.dp_clip_total(group_clips)
        summed = task.dp_clip_sum_grouped(gradients, group_ids, group_clips)
    else:
        bound = float(config["dp-clip"])
        summed = task.dp_clip_sum(gradients, bound)
    noisy = summed + task.dp_noise(gradients.shape[1], bound, sigma, sites)
    if is_prior_carrier:
        # Public prior gradient, in the same preconditioned coordinate the data was clipped in.
        noisy = noisy + task.map_prior(nl, dm, theta)[1] * precond
    return noisy, bound


def _mcem_parts(context: Context):
    """This site's (q1_names, q2_names) M-step partition, cached per site."""
    key = _site_key(context)
    if key not in _site_mcem_parts:
        parts = nl.seval("nlf_mcem_parts")(_site_dm(context))
        _site_mcem_parts[key] = ([str(s) for s in parts.q1], [str(s) for s in parts.q2])
    return _site_mcem_parts[key]


def _mcem_estep(context: Context, config, theta) -> None:
    """One LOCAL E-step: sample this site's posteriors at theta, cache the FIXED draws + the
    threaded warm-start state. outer-iter 1 resets (fresh state, prior-mean seeding)."""
    key = _site_key(context)
    site_id = int(context.node_config["partition-id"])
    outer = int(config["mcem-outer-iter"])
    if outer == 1:  # new run: drop any stale state/draws from a previous run in this actor
        _site_mcem_state.pop(key, None)
        for k in [k for k in _site_mcem_draws if k[0] == key]:
            del _site_mcem_draws[k]
    state = None if outer == 1 else _site_mcem_state.get(key)
    draws, new_state = nl.seval("nlf_mcem_estep")(
        _site_dm(context), theta,
        int(config["mcem-sample-schedule"]), int(config["mcem-maxiters"]),
        int(config["mcem-seed"]) + site_id, state,
    )
    _site_mcem_state[key] = new_state
    _site_mcem_draws[(key, outer)] = draws


def _mcem_mstep(context: Context, config, theta):
    """One M-step Q (value, part-gradient) at the FIXED draws of this outer iteration."""
    key = _site_key(context)
    outer = int(config["mcem-outer-iter"])
    part = str(config["mcem-part"])
    draws = _site_mcem_draws[(key, outer)]
    q1_names, q2_names = _mcem_parts(context)
    fnames = q1_names if part == "q1" else q2_names
    Q, g = nl.seval("nlf_mcem_q")(_site_dm(context), theta, draws, part, fnames)
    return float(Q), np.asarray(g, dtype=float)


@app.query("prepare")
def prepare(msg: Message, context: Context) -> Message:
    """Warm this site: build the DataModel, burn one objective call, report theta0/names."""
    config = msg.content["config"]
    t0 = time.perf_counter()
    dm = _site_dm(context)
    # The FitContext (batch infos + caches) that every later round evaluates through.
    nl.seval("nlf_ctx")(dm)
    theta0 = np.asarray(nl.seval("nlf_theta0")(dm), dtype=float)
    log_mask = np.asarray(nl.seval("nlf_logmask")(dm), dtype=float)
    # Warm up the round path so its first-call compilation is paid here, not in round 1.
    # Under dp that is the per-batch path, and it also reports whether the clip unit really
    # is the subject (max batch ids == 1).
    extra = {}
    reply_extra = {}
    if str(config["estimator"]) == "mcem":
        # Warm the nested EM path (E-step + both M-step Q parts) so their first-call
        # compilation is paid here, and report the q1/q2 partition the server optimizes. The
        # warm draws are discarded: the real run's outer iteration 1 samples fresh.
        q1_names, q2_names = _mcem_parts(context)
        warm_draws, _ = nl.seval("nlf_mcem_estep")(
            dm, theta0, int(config["mcem-sample-schedule"]), int(config["mcem-maxiters"]),
            int(config["mcem-seed"]), None,
        )
        for part, fn in (("q1", q1_names), ("q2", q2_names)):
            if fn:
                nl.seval("nlf_mcem_q")(dm, theta0, warm_draws, part, fn)
        reply_extra["mcem"] = ConfigRecord({"q1": q1_names, "q2": q2_names})
    elif bool(config["dp"]):
        _, _, max_batch_ids = task.dp_batch_contributions(
            nl, dm, theta0, str(config["estimator"]), int(config["ghq-level"])
        )
        extra["max-batch-ids"] = max_batch_ids
    else:
        task.objective_and_gradient(
            nl, dm, theta0, str(config["estimator"]), int(config["ghq-level"])
        )
    setup_seconds = time.perf_counter() - t0
    key = _site_key(context)
    log(INFO, "prepare: site %d ready in %.1fs", key[0], setup_seconds)
    return Message(
        content=RecordDict({
            "theta0": ArrayRecord([theta0]),
            "log_mask": ArrayRecord([log_mask]),
            "names": ConfigRecord({"names": [str(s) for s in nl.seval("nlf_names")(dm)]}),
            "result": MetricRecord({
                "ready": 1,
                "site-id": key[0],
                "subjects": _site_subjects[key],
                "setup-seconds": setup_seconds,
                **extra,
            }),
            **reply_extra,
        }),
        reply_to=msg,
    )


@app.query()
def site_objective(msg: Message, context: Context) -> Message:
    config = msg.content["config"]
    site_id = int(context.node_config["partition-id"])
    # TESTING ONLY: run-config `fail-site=<id>` makes that site raise, to verify the
    # server aborts the whole fit instead of summing the surviving sites.
    if site_id == int(context.run_config["fail-site"]):
        raise RuntimeError(f"fault injection: site {site_id} refuses to answer")
    theta = msg.content["theta"].to_numpy_ndarrays()[0]
    # MCEM (nested EM): the outer iteration's phase is on the wire. The E-step is LOCAL - it
    # samples this site's posteriors and caches the draws, releasing NOTHING. Each M-step
    # round returns this site's per-part Q (value, gradient) over those cached draws, which
    # the server sums (or, under dp, the clipped+noised gradient release).
    phase = str(config.get("mcem-phase", "")) if str(config["estimator"]) == "mcem" else ""
    if phase == "estep":
        _mcem_estep(context, config, theta)
        return Message(
            content=RecordDict({"result": MetricRecord({"site-id": site_id, "ready": 1})}),
            reply_to=msg,
        )
    if phase == "mstep":
        if bool(config.get("dp", False)):
            release, _ = _mcem_dp_contribution(context, config, theta)
            return Message(
                content=RecordDict({
                    "release": ArrayRecord([release]),
                    "result": MetricRecord({"site-id": site_id}),
                }),
                reply_to=msg,
            )
        value, gradient = _mcem_mstep(context, config, theta)
        return Message(
            content=RecordDict({
                "gradient": ArrayRecord([gradient]),
                "result": MetricRecord({"value": value, "site-id": site_id}),
            }),
            reply_to=msg,
        )
    if bool(config.get("dp", False)):
        # Under dp the ONLY thing this site releases is the clipped, noised vector: no value
        # and no per-site log-likelihood.
        release, _ = _dp_contribution(context, config, theta)
        return Message(
            content=RecordDict({
                "release": ArrayRecord([release]),
                "result": MetricRecord({"site-id": site_id}),
            }),
            reply_to=msg,
        )
    # MAP prior-carrier rule: site 0 runs MAP (loglik + the shared prior), every other site
    # runs MLE (loglik only), so the server's naive sum is the pooled MAP objective. MLE and
    # the RE estimators are unaffected (site_estimator returns them unchanged).
    estimator = task.site_estimator(str(config["estimator"]), site_id)
    # require_finite=False: a non-finite marginal at an optimizer probe theta is a legitimate
    # estimator result, so reply successfully with it and let the server backtrack on a finite
    # penalty. A genuine site error (a Julia solve that throws) still propagates and aborts.
    value, gradient = task.objective_and_gradient(
        nl,
        _site_dm(context),
        theta,
        estimator,
        int(config["ghq-level"]),
        require_finite=False,
    )
    metrics = MetricRecord({
        "value": value,
        "site-id": int(context.node_config["partition-id"]),
        "boot-on-main": int(BOOT_ON_MAIN),
        "handler-on-main": int(threading.current_thread() is threading.main_thread()),
    })
    return Message(
        content=RecordDict({"gradient": ArrayRecord([gradient]), "result": metrics}),
        reply_to=msg,
    )
