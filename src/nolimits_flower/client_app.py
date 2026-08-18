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

_site_dms: dict[tuple[int, int], object] = {}


def _site_dm(context: Context):
    key = (int(context.node_config["partition-id"]), int(context.node_config["num-partitions"]))
    if key not in _site_dms:
        pid, num = key
        _site_dms[key] = task.build_data_model(nl, task.partition(task.simulate(), num)[pid])
    return _site_dms[key]


@app.query()
def site_objective(msg: Message, context: Context) -> Message:
    log(
        INFO,
        "site %s: handler on thread %r (main=%s), booted on %r",
        context.node_config["partition-id"],
        threading.current_thread().name,
        threading.current_thread() is threading.main_thread(),
        BOOT_THREAD,
    )
    config = msg.content["config"]
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
