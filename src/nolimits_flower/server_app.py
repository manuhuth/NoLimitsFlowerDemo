"""ServerApp: broadcasts theta, sums the site replies (Phase 1 echo loop)."""

from logging import INFO

import numpy as np
from flwr.app import ArrayRecord, Context, Message, RecordDict
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp

app = ServerApp()


def broadcast(grid: Grid, theta: np.ndarray, rnd: int) -> list[float]:
    """Send theta to every node, return the site values (raises if any site fails)."""
    node_ids = list(grid.get_node_ids())
    messages = [
        Message(
            content=RecordDict({"theta": ArrayRecord([theta])}),
            message_type="query",
            dst_node_id=nid,
            group_id=str(rnd),
        )
        for nid in node_ids
    ]
    replies = list(grid.send_and_receive(messages))
    if len(replies) != len(node_ids):
        raise RuntimeError(f"round {rnd}: {len(replies)}/{len(node_ids)} sites replied")
    values = []
    for reply in replies:
        if not reply.has_content():
            raise RuntimeError(f"round {rnd}: site error: {reply.error}")
        values.append(float(reply.content["result"]["value"]))
    return values


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config["num-server-rounds"])
    for rnd in range(1, num_rounds + 1):
        theta = np.arange(1.0, 5.0) * rnd
        values = broadcast(grid, theta, rnd)
        total = sum(values)
        # site_id runs 1..S, so the closed form is S(S+1)/2 * sum(theta).
        n = len(values)
        expected = n * (n + 1) / 2 * float(np.sum(theta))
        log(
            INFO,
            "round %d: sites=%d total=%.6f expected=%.6f match=%s",
            rnd, n, total, expected, total == expected,
        )
