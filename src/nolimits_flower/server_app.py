"""ServerApp: one verification round - summed site (value, gradient) vs pooled.

Phase 2 only checks the federation identity at the TRUE simulation theta; the
optimizer loop is Phase 3.

The ServerApp itself never touches Julia: in the simulation runtime `@app.main`
runs on a worker thread (observed: `Thread-9 (server_th_with_start_checks)`), and
juliacall can only cold-boot Julia from a process's main thread. The pooled
reference therefore runs in a child process, whose main thread is free.
"""

import json
import subprocess
import sys
from logging import INFO

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp

from nolimits_flower import task

app = ServerApp()


def pooled_reference(estimator: str, ghq_level: int):
    """Pooled (theta, value, gradient) from a child process with its own Julia.

    Its output is captured rather than inherited: Julia's chatter into the
    simulation process's log pipe blocked the child indefinitely.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "nolimits_flower.task", estimator, str(ghq_level)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pooled reference failed:\n{proc.stderr[-4000:]}")
    line = next(l for l in proc.stdout.splitlines() if l.startswith("POOLED_JSON "))
    ref = json.loads(line[len("POOLED_JSON "):])
    return np.asarray(ref["theta"]), float(ref["value"]), np.asarray(ref["gradient"])


def broadcast(grid: Grid, theta: np.ndarray, config: ConfigRecord, rnd: int):
    """Send theta to every node; return [(site_id, value, gradient)]. Raises on any failure."""
    node_ids = list(grid.get_node_ids())
    messages = [
        Message(
            content=RecordDict({"theta": ArrayRecord([theta]), "config": config}),
            message_type="query",
            dst_node_id=nid,
            group_id=str(rnd),
        )
        for nid in node_ids
    ]
    replies = list(grid.send_and_receive(messages))
    if len(replies) != len(node_ids):
        raise RuntimeError(f"round {rnd}: {len(replies)}/{len(node_ids)} sites replied")
    out = []
    for reply in replies:
        if not reply.has_content():
            raise RuntimeError(f"round {rnd}: site error: {reply.error}")
        metrics = reply.content["result"]
        value = float(metrics["value"])
        gradient = reply.content["gradient"].to_numpy_ndarrays()[0]
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise RuntimeError(f"round {rnd}: non-finite contribution from site {metrics['site-id']}")
        log(
            INFO,
            "  site %s: value=%.10f handler-on-main=%s boot-on-main=%s",
            metrics["site-id"], value, metrics["handler-on-main"], metrics["boot-on-main"],
        )
        out.append((int(metrics["site-id"]), value, gradient))
    return out


@app.main()
def main(grid: Grid, context: Context) -> None:
    estimator = str(context.run_config["estimator"])
    ghq_level = int(context.run_config["ghq-level"])
    config = ConfigRecord({"estimator": estimator, "ghq-level": ghq_level})

    theta, pooled_value, pooled_grad = pooled_reference(estimator, ghq_level)

    log(INFO, "verification round: estimator=%s theta(transformed)=%s", estimator, theta)
    sites = broadcast(grid, theta, config, rnd=1)
    summed_value = sum(v for _, v, _ in sites)
    summed_grad = np.sum([g for _, _, g in sites], axis=0)

    value_rel = abs(summed_value - pooled_value) / abs(pooled_value)
    grad_rel = np.abs(summed_grad - pooled_grad) / np.maximum(np.abs(pooled_grad), 1e-12)
    log(INFO, "sites=%d summed value=%.10f pooled value=%.10f rel=%.3e",
        len(sites), summed_value, pooled_value, value_rel)
    log(INFO, "summed gradient=%s", summed_grad)
    log(INFO, "pooled gradient=%s", pooled_grad)
    log(INFO, "gradient rel-diffs=%s (worst %.3e)", grad_rel, grad_rel.max())
    if value_rel >= 1e-8 or grad_rel.max() >= 1e-8:
        raise RuntimeError(f"federation identity violated: value {value_rel:.3e}, gradient {grad_rel.max():.3e}")
    log(INFO, "PASS: summed site value and gradient match the pooled call within 1e-8 relative")
