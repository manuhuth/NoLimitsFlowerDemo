"""ServerApp: the federated fit, plus the pooled-fit acceptance comparison.

One L-BFGS-B objective evaluation == one federated round: broadcast the
transformed-scale theta, collect each site's (value, gradient), sum them. The sum
is the pooled marginal log-likelihood and its gradient exactly (disjoint subjects),
so the optimum is the pooled `fit_model` optimum.

The ServerApp itself never touches Julia: in the simulation runtime `@app.main`
runs on a worker thread and juliacall can only cold-boot Julia from a process's
main thread. Anything needing Julia (the pooled reference fit, the model's default
theta, the parameter names) comes from a child process whose main thread is free.
"""

import json
import subprocess
import sys
import time
from logging import ERROR, INFO

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp
from scipy.optimize import minimize

from nolimits_flower import task

app = ServerApp()


def pooled_fit(estimator: str, ghq_level: int, seed: int) -> dict:
    """Pooled `fit_model` reference from a child process with its own Julia.

    Its output is captured rather than inherited: Julia's chatter into the
    simulation process's log pipe blocked the child indefinitely.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "nolimits_flower.task", "fit", estimator, str(ghq_level), str(seed)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pooled fit failed:\n{proc.stderr[-4000:]}")
    line = next(l for l in proc.stdout.splitlines() if l.startswith("POOLED_JSON "))
    return json.loads(line[len("POOLED_JSON "):])


class SiteFailure(RuntimeError):
    """One site failed, so the federated sum is incomplete and the fit must abort."""


# node id -> site id, learned from successful replies: an error reply carries no
# content, so this is the only way to name the site rather than just the node.
_SITE_OF_NODE: dict[int, int] = {}


def _short_reason(error) -> str:
    """The site's own exception message out of the framework's nested traceback dump."""
    reason = str(getattr(error, "reason", error) or "")
    marker = "Message: "
    if marker in reason:
        reason = reason.rsplit(marker, 1)[1]
    lines = [l.strip() for l in reason.splitlines() if l.strip()]
    text = (lines[0] if lines else "no reason reported").rstrip("'\">").strip()
    code = getattr(error, "code", None)
    return f"{text[:300]} (error code {code})" if code is not None else text[:300]


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
        missing = set(node_ids) - {r.metadata.src_node_id for r in replies}
        raise SiteFailure(
            f"round {rnd}: only {len(replies)}/{len(node_ids)} sites replied; no reply from "
            f"{[(_SITE_OF_NODE.get(n, '?'), n) for n in sorted(missing)]} (site, node) - "
            "aborting the federated fit rather than summing a subset of the sites"
        )
    out = []
    for reply in replies:
        node = reply.metadata.src_node_id
        if not reply.has_content():
            raise SiteFailure(
                f"round {rnd}: site {_SITE_OF_NODE.get(node, 'unknown (first round)')} "
                f"(node {node}) failed: {_short_reason(reply.error)} - aborting the federated "
                "fit rather than summing the remaining sites; see that node's ClientApp log"
            )
        metrics = reply.content["result"]
        site_id = int(metrics["site-id"])
        _SITE_OF_NODE[node] = site_id
        value = float(metrics["value"])
        gradient = reply.content["gradient"].to_numpy_ndarrays()[0]
        # A -Inf / NaN site contribution (failed solve) must never be summed.
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise SiteFailure(
                f"round {rnd}: non-finite contribution from site {site_id} (node {node}, "
                f"value={value}) - aborting the federated fit"
            )
        out.append((site_id, value, gradient))
    return out


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Wrapper: a site failure aborts with one actionable line, not a nested traceback."""
    try:
        _fit(grid, context)
    except SiteFailure as exc:
        log(ERROR, "FEDERATED FIT ABORTED: %s", exc)
        raise SiteFailure(str(exc)) from None


def _fit(grid: Grid, context: Context) -> None:
    estimator = str(context.run_config["estimator"])
    ghq_level = int(context.run_config["ghq-level"])
    seed = int(context.run_config["data-seed"])
    max_rounds = int(context.run_config["max-rounds"])
    fail_site = int(context.run_config["fail-site"])
    if fail_site >= 0:
        log(INFO, "fault injection active (testing only): site %d will raise", fail_site)
    config = ConfigRecord({"estimator": estimator, "ghq-level": ghq_level})

    ref = pooled_fit(estimator, ghq_level, seed)
    names = ref["names"]
    x0 = np.asarray(ref["theta0"], dtype=float)  # the model's default theta
    log(INFO, "estimator=%s data-seed=%d params=%s start(natural)=%s",
        estimator, seed, names, task.to_natural(x0))

    rounds = 0
    t0 = time.perf_counter()

    def federated(x: np.ndarray):
        nonlocal rounds
        rounds += 1
        sites = broadcast(grid, np.asarray(x, dtype=float), config, rnd=rounds)
        value = sum(v for _, v, _ in sites)
        grad = np.sum([g for _, _, g in sites], axis=0)
        log(INFO, "round %d: loglik=%.10f |grad|=%.3e sites=%d", rounds, value, np.linalg.norm(grad), len(sites))
        return -value, -grad  # L-BFGS-B minimizes; the sites report a log-likelihood

    # maxfun caps function evaluations, i.e. federated rounds - the round guard. maxiter
    # alone would not: line searches spend extra evaluations per iteration. A truncated
    # run leaves res.success False and fails the acceptance below, rather than passing
    # off a half-optimized theta as the optimum.
    res = minimize(
        federated, x0, method="L-BFGS-B", jac=True,
        options={"maxiter": max_rounds, "maxfun": max_rounds},
    )
    wall = time.perf_counter() - t0

    # Final round at the optimum: also gives the per-site contributions to report.
    final = broadcast(grid, res.x, config, rnd=rounds + 1)
    rounds += 1
    fed_value = sum(v for _, v, _ in final)
    fed_natural = task.to_natural(res.x)

    log(INFO, "converged=%s (%s)", res.success, res.message)
    log(INFO, "federated theta*(natural) = %s", dict(zip(names, fed_natural.tolist())))
    log(INFO, "federated loglik=%.10f rounds=%d wall=%.1fs", fed_value, rounds, wall)
    for site_id, value, _ in sorted(final):
        log(INFO, "  site %d contribution: %.10f", site_id, value)

    pooled_natural = np.asarray(ref["theta_natural"], dtype=float)
    pooled_value = float(ref["value"])
    value_rel = abs(fed_value - pooled_value) / abs(pooled_value)
    theta_rel = np.abs(fed_natural - pooled_natural) / np.abs(pooled_natural)

    log(INFO, "ACCEPTANCE (federated vs pooled fit_model, data-seed=%d)", seed)
    log(INFO, "  %-8s %14s %14s %10s", "param", "federated", "pooled", "rel.diff")
    for name, f, p, r in zip(names, fed_natural, pooled_natural, theta_rel):
        log(INFO, "  %-8s %14.8f %14.8f %10.2e", name, f, p, r)
    log(INFO, "  %-8s %14.8f %14.8f %10.2e", "loglik", fed_value, pooled_value, value_rel)

    if estimator != "laplace":
        log(INFO, "estimator=%s: smoke run only, no acceptance assertion", estimator)
        return
    if value_rel >= 1e-6 or theta_rel.max() >= 1e-3:
        raise RuntimeError(
            f"acceptance failed: objective rel {value_rel:.3e} (tol 1e-6), "
            f"worst parameter rel {theta_rel.max():.3e} (tol 1e-3)"
        )
    log(INFO, "PASS: federated optimum matches the pooled fit (objective %.2e, worst param %.2e)",
        value_rel, theta_rel.max())
