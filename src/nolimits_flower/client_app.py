"""ClientApp: answers a transformed-scale theta with this site's (value, gradient).

Julia is booted at MODULE IMPORT, i.e. at client process start, because juliacall
cannot cold-boot Julia from a non-main thread (NoLimitsPy raises instead of
hanging). The site DataModel is cached in a module global: `context.state` only
holds records, and ClientApp objects are rebuilt per message, so a module global
is the only place a live Julia object survives across rounds within one process.
"""

import threading
from logging import INFO

import NoLimitsPy as nl
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.logger import log

from nolimits_flower import task

BOOT_THREAD = threading.current_thread().name
BOOT_ON_MAIN = threading.current_thread() is threading.main_thread()
nl.seval("1")  # cold boot happens here, whatever thread later handlers run on
log(INFO, "NoLimitsPy booted on thread %r (main=%s)", BOOT_THREAD, BOOT_ON_MAIN)

app = ClientApp()

_site_dms: dict[tuple[int, int, int], object] = {}


def _site_dm(context: Context):
    """This site's DataModel, built once per (partition, data seed) per process."""
    key = (
        int(context.node_config["partition-id"]),
        int(context.node_config["num-partitions"]),
        int(context.run_config["data-seed"]),
    )
    if key not in _site_dms:
        pid, num, seed = key
        df = task.partition(task.simulate(seed=seed), num)[pid]
        _site_dms[key] = task.build_data_model(nl, df)
    return _site_dms[key]


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
