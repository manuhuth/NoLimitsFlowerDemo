"""ClientApp: echo skeleton (Phase 1, no NoLimits yet)."""

import numpy as np
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

app = ClientApp()


@app.query()
def site_objective(msg: Message, context: Context) -> Message:
    theta = msg.content["theta"].to_numpy_ndarrays()[0]
    site_id = int(context.node_config["partition-id"]) + 1
    value = site_id * float(np.sum(theta))
    return Message(
        content=RecordDict({"result": MetricRecord({"value": value, "site-id": site_id})}),
        reply_to=msg,
    )
