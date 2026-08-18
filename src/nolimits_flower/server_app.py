"""ServerApp: the federated fit, plus the pooled-fit acceptance comparison.

A run is one PREPARE round followed by pure evaluation rounds:

- prepare: every site builds its DataModel and burns one warm-up objective call, so
  the ~85 s of Julia boot + codegen + DataModel build is visible as its own round
  instead of hiding inside optimization round 1. The sites also report the parameter
  names and the model-default transformed theta0 - that is where the fit's start
  point comes from, so the optimizer needs no Julia anywhere on the server side.
- each subsequent L-BFGS-B objective evaluation == one federated round: broadcast the
  transformed-scale theta, collect each site's (value, gradient), sum them. The sum
  is the pooled marginal log-likelihood and its gradient exactly (disjoint subjects),
  so the optimum is the pooled `fit_model` optimum.

The ServerApp itself never touches Julia: in the simulation runtime `@app.main`
runs on a worker thread and juliacall can only cold-boot Julia from a process's
main thread. The only Julia-in-a-child-process left is the pooled reference fit,
which runs AFTER convergence purely for the demo acceptance comparison.
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


def pooled_fit(estimator: str, ghq_level: int, seed: int, source: str) -> dict:
    """Pooled `fit_model` reference from a child process with its own Julia.

    DEMO ONLY: this is the acceptance comparison for the simulated demo, run after the
    federated fit has converged. Production deployments do not run it - there is no
    pooled dataset, and the federated fit needs nothing from it (theta0 and the
    parameter names come from the prepare round).

    Its output is captured rather than inherited: Julia's chatter into the
    simulation process's log pipe blocked the child indefinitely.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "nolimits_flower.task", "fit", estimator, str(ghq_level),
         str(seed), source],
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


def _send_all(grid: Grid, message_type: str, records: dict, label: str):
    """Send the same content to every node; return the replies. Raises on any failure."""
    node_ids = list(grid.get_node_ids())
    messages = [
        Message(
            content=RecordDict(dict(records)),
            message_type=message_type,
            dst_node_id=nid,
            group_id=label,
        )
        for nid in node_ids
    ]

    # Actual information is sent and received back
    replies = list(grid.send_and_receive(messages))

    # Received information is processed
    if len(replies) != len(node_ids):
        missing = set(node_ids) - {r.metadata.src_node_id for r in replies}
        raise SiteFailure(
            f"{label}: only {len(replies)}/{len(node_ids)} sites replied; no reply from "
            f"{[(_SITE_OF_NODE.get(n, '?'), n) for n in sorted(missing)]} (site, node) - "
            "aborting the federated fit rather than summing a subset of the sites"
        )
    for reply in replies:
        if not reply.has_content():
            node = reply.metadata.src_node_id
            raise SiteFailure(
                f"{label}: site {_SITE_OF_NODE.get(node, 'unknown (first round)')} "
                f"(node {node}) failed: {_short_reason(reply.error)} - aborting the federated "
                "fit rather than summing the remaining sites; see that node's ClientApp log"
            )
    return replies


def agree(sites: list[tuple[int, list[str], np.ndarray]]) -> tuple[list[str], np.ndarray]:
    """Collapse the sites' prepare replies to the one (names, theta0) they must all share.

    The sites run the same model, so a disagreement means they are not fitting the same
    thing and the summed objective would be meaningless. Pure function: unit-tested.
    """
    if not sites:
        raise SiteFailure("prepare round: no sites reported")
    ref_id, names, theta0 = sites[0]
    for site_id, other_names, other_theta0 in sites[1:]:
        if other_names != names:
            raise SiteFailure(
                f"prepare round: site {site_id} reports parameter names {other_names} but "
                f"site {ref_id} reports {names} - the sites are not running the same model"
            )
        if not np.array_equal(other_theta0, theta0):
            raise SiteFailure(
                f"prepare round: site {site_id} reports theta0 {list(other_theta0)} but site "
                f"{ref_id} reports {list(theta0)} - the sites are not running the same model"
            )
    return list(names), np.asarray(theta0, dtype=float)


def prepare(grid: Grid, config: ConfigRecord) -> tuple[list[str], np.ndarray]:
    """The prepare round: warm every site, log the setup table, source names/theta0.

    ponytail: one message per node, no retries. In the SIMULATION runtime Ray actors are
    not pinned to a node, so an actor serving several partitions still builds the ones it
    has not seen inside round 1; give each site its own actor to avoid that. In deployment
    (one process per site) this round absorbs the whole setup cost.
    """
    replies = _send_all(grid, "query.prepare", {"config": config}, "prepare round")
    sites = []
    log(INFO, "PREPARE ROUND (%d sites)", len(replies))
    log(INFO, "  %-6s %9s %14s", "site", "subjects", "setup (s)")
    for reply in sorted(replies, key=lambda r: int(r.content["result"]["site-id"])):
        metrics = reply.content["result"]
        site_id = int(metrics["site-id"])
        _SITE_OF_NODE[reply.metadata.src_node_id] = site_id
        if not int(metrics["ready"]):
            raise SiteFailure(f"prepare round: site {site_id} did not report ready")
        log(INFO, "  %-6d %9d %14.1f", site_id, int(metrics["subjects"]),
            float(metrics["setup-seconds"]))
        sites.append((
            site_id,
            [str(n) for n in reply.content["names"]["names"]],
            reply.content["theta0"].to_numpy_ndarrays()[0],
        ))
    return agree(sites)


def broadcast(grid: Grid, theta: np.ndarray, config: ConfigRecord, rnd: int):
    """Send theta to every node; return [(site_id, value, gradient)]. Raises on any failure."""
    replies = _send_all(
        grid, "query", {"theta": ArrayRecord([theta]), "config": config}, f"round {rnd}"
    )
    out = []
    for reply in replies:
        node = reply.metadata.src_node_id
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
    source = str(context.run_config["data-source"])
    max_rounds = int(context.run_config["max-rounds"])
    fail_site = int(context.run_config["fail-site"])
    if fail_site >= 0:
        log(INFO, "fault injection active (testing only): site %d will raise", fail_site)
    config = ConfigRecord({"estimator": estimator, "ghq-level": ghq_level})

    # Prepare round: sites warm up and hand over the shared start point. No Julia on the
    # server, and every later round is a pure warm evaluation.
    t_prep = time.perf_counter()
    names, x0 = prepare(grid, config)
    log(INFO, "prepare round wall=%.1fs", time.perf_counter() - t_prep)
    log(INFO, "estimator=%s data-source=%s data-seed=%d params=%s start(natural)=%s",
        estimator, source, seed, names, task.to_natural(x0, names))

    rounds = 0
    t0 = time.perf_counter()

    def federated(x: np.ndarray):
        nonlocal rounds
        rounds += 1
        t_round = time.perf_counter()
        sites = broadcast(grid, np.asarray(x, dtype=float), config, rnd=rounds)
        value = sum(v for _, v, _ in sites)
        grad = np.sum([g for _, _, g in sites], axis=0)
        log(INFO, "round %d: loglik=%.10f |grad|=%.3e sites=%d wall=%.2fs", rounds, value,
            np.linalg.norm(grad), len(sites), time.perf_counter() - t_round)
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
    fed_natural = task.to_natural(res.x, names)

    log(INFO, "converged=%s (%s)", res.success, res.message)
    log(INFO, "federated theta*(natural) = %s", dict(zip(names, fed_natural.tolist())))
    log(INFO, "federated loglik=%.10f evaluation rounds=%d wall=%.1fs (%.2fs/round)",
        fed_value, rounds, wall, wall / max(rounds, 1))
    for site_id, value, _ in sorted(final):
        log(INFO, "  site %d contribution: %.10f", site_id, value)

    # DEMO ONLY, and only now that the fit is done: the pooled reference the acceptance
    # table compares against. Production deployments delete this call.
    ref = pooled_fit(estimator, ghq_level, seed, source)
    if ref["names"] != names:
        raise RuntimeError(f"pooled reference parameter order {ref['names']} != sites' {names}")
    pooled_natural = np.asarray(ref["theta_natural"], dtype=float)
    pooled_value = float(ref["value"])
    value_rel = abs(fed_value - pooled_value) / abs(pooled_value)
    theta_rel = np.abs(fed_natural - pooled_natural) / np.abs(pooled_natural)

    log(INFO, "ACCEPTANCE (federated vs pooled fit_model, data-source=%s data-seed=%d)",
        source, seed)
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
