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


def _site_key(context: Context) -> tuple:
    return (
        int(context.node_config["partition-id"]),
        int(context.node_config["num-partitions"]),
        str(context.run_config["data-source"]),
        int(context.run_config["data-seed"]),
    )


def _site_dm(context: Context):
    """This site's DataModel, built once per (partition, data source) per process."""
    key = _site_key(context)
    if key not in _site_dms:
        pid, num, source, seed = key
        df = task.partition(task.dataset(source, seed, nl), num)[pid]
        _site_dms[key] = task.build_data_model(nl, df)
        _site_subjects[key] = int(df["ID"].nunique())
    return _site_dms[key]


@app.query("prepare")
def prepare(msg: Message, context: Context) -> Message:
    """Warm this site: build the DataModel, burn one objective call, report theta0/names."""
    config = msg.content["config"]
    t0 = time.perf_counter()
    dm = _site_dm(context)
    # The FitContext (batch infos + caches) that every later round evaluates through.
    nl.seval("nlf_ctx")(dm)
    theta0 = np.asarray(nl.seval("nlf_theta0")(dm), dtype=float)
    # Discarded: its only job is to pay the first-call compilation cost here.
    task.objective_and_gradient(nl, dm, theta0, str(config["estimator"]), int(config["ghq-level"]))
    setup_seconds = time.perf_counter() - t0
    key = _site_key(context)
    log(INFO, "prepare: site %d ready in %.1fs", key[0], setup_seconds)
    return Message(
        content=RecordDict({
            "theta0": ArrayRecord([theta0]),
            "names": ConfigRecord({"names": [str(s) for s in nl.seval("nlf_names")(dm)]}),
            "result": MetricRecord({
                "ready": 1,
                "site-id": key[0],
                "subjects": _site_subjects[key],
                "setup-seconds": setup_seconds,
            }),
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
    value, gradient = task.objective_and_gradient(
        nl,
        _site_dm(context),
        theta,
        str(config["estimator"]),
        int(config["ghq-level"]),
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
