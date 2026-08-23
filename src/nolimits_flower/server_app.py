"""ServerApp: the federated fit, plus the pooled-fit acceptance comparison.

A run is one PREPARE round followed by pure evaluation rounds:

- prepare: every site builds its DataModel and burns one warm-up objective call, so
  the ~85 s of Julia boot + codegen + DataModel build is visible as its own round
  instead of hiding inside optimization round 1. The sites also report the parameter
  names and the model-default transformed theta0 - that is where the fit's start
  point comes from, so the optimizer needs no Julia anywhere on the server side.
- each subsequent L-BFGS-B objective evaluation == one federated round: broadcast the
  transformed-scale theta, collect each site's (value, gradient), sum them. Every wired
  estimator (laplace, focei, ghq, pooled) is a sum of per-subject terms, so with disjoint
  subjects the sum IS the pooled-data objective and gradient exactly, and the optimum is
  the pooled `fit_model` optimum. `PARAM_TOL` and the ghq branch below record where that
  identity survives the *optimizer* and where it only survives in the objective.

The ServerApp itself never touches Julia: in the simulation runtime `@app.main`
runs on a worker thread and juliacall can only cold-boot Julia from a process's
main thread. The only Julia-in-a-child-process left is the pooled reference fit,
which runs AFTER convergence purely for the demo acceptance comparison.
"""

import json
import subprocess
import sys
import time
from logging import ERROR, INFO, WARNING
from pathlib import Path

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp
from scipy.optimize import minimize

from nolimits_flower import task

app = ServerApp()

# All four estimators sum over subjects, so the summed site contributions ARE the
# pooled objective and its gradient - measured to <=1e-15 relative for every one of them
# by `python -m nolimits_flower.task probe`. Pooled's exactness is model-conditional (see
# the README estimator table), hence the log line below. Parameter-wise acceptance uses
# the per-model tolerance (task.ModelSpec.param_tol, 1e-3 for the PK/growth models);
# `ghq` is gated one-sidedly and `pooled` at 1e-2 (its plug-in eta makes the objective
# nearly flat in the omegas). The neural model is gated on additivity, not parameters.
POOLED_CAVEAT = (
    "estimator=pooled: exact here because every random effect is LogNormal, so the "
    "plug-in eta strategy resolves to :mean, a function of theta alone. A model whose "
    "plug-in resolution depends on the DATA (a normalizing-flow RE, or a strategy "
    "demoted for ForwardDiff-safety on one site's data only) would calibrate per site "
    "and break additivity - re-run `python -m nolimits_flower.task probe` after changing "
    "the model."
)


def _child(mode: str, model: str, estimator: str, ghq_level: int, seed: int, source: str) -> dict:
    """Run a task CLI mode ({fit|ref|probe}) in a child process with its own Julia.

    DEMO ONLY. Julia cannot boot on the ServerApp's worker thread, so the pooled
    reference fit and the neural model's additivity gate run as child processes with
    output captured (Julia's chatter into the simulation log pipe deadlocked the child).
    Production deployments do not run any of this - the federated fit needs nothing from
    the pooled data (theta0 and names come from the prepare round).
    """
    proc = subprocess.run(
        [sys.executable, "-m", "nolimits_flower.task", mode, model, estimator,
         str(ghq_level), str(seed), source],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"child {mode} failed:\n{proc.stderr[-4000:]}")
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


def agree(sites: list[tuple]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Collapse the sites' prepare replies to the (names, theta0, log_mask) they must share.

    The sites run the same model, so a disagreement means they are not fitting the same
    thing and the summed objective would be meaningless (for the neural model this is what
    catches an unpinned FFNN seed). Pure function: unit-tested.
    """
    if not sites:
        raise SiteFailure("prepare round: no sites reported")
    ref_id, names, theta0, mask = sites[0]
    for site_id, other_names, other_theta0, other_mask in sites[1:]:
        if other_names != names:
            raise SiteFailure(
                f"prepare round: site {site_id} reports parameter names {other_names} but "
                f"site {ref_id} reports {names} - the sites are not running the same model"
            )
        if not np.array_equal(other_theta0, theta0):
            raise SiteFailure(
                f"prepare round: site {site_id} reports theta0 {list(other_theta0)} but site "
                f"{ref_id} reports {list(theta0)} - the sites are not running the same model "
                "(for the neural model, an unpinned FFNN seed)"
            )
    return list(names), np.asarray(theta0, dtype=float), np.asarray(mask, dtype=float)


def prepare(grid: Grid, config: ConfigRecord):
    """The prepare round: warm every site, log the setup table, source names/theta0.

    Returns (names, theta0, log_mask, num_sites, biggest_batch); biggest_batch is the
    largest number of subjects in any random-effect batch over all sites (dp only, 1
    otherwise), i.e. whether the clipping unit really is the subject.

    ponytail: one message per node, no retries. In the SIMULATION runtime Ray actors are
    not pinned to a node, so an actor serving several partitions still builds the ones it
    has not seen inside round 1; give each site its own actor to avoid that. In deployment
    (one process per site) this round absorbs the whole setup cost.
    """
    replies = _send_all(grid, "query.prepare", {"config": config}, "prepare round")
    sites = []
    biggest_batch = 1
    log(INFO, "PREPARE ROUND (%d sites)", len(replies))
    log(INFO, "  %-6s %9s %14s", "site", "subjects", "setup (s)")
    for reply in sorted(replies, key=lambda r: int(r.content["result"]["site-id"])):
        metrics = reply.content["result"]
        site_id = int(metrics["site-id"])
        _SITE_OF_NODE[reply.metadata.src_node_id] = site_id
        if not int(metrics["ready"]):
            raise SiteFailure(f"prepare round: site {site_id} did not report ready")
        biggest_batch = max(biggest_batch, int(dict(metrics).get("max-batch-ids", 1)))
        log(INFO, "  %-6d %9d %14.1f", site_id, int(metrics["subjects"]),
            float(metrics["setup-seconds"]))
        sites.append((
            site_id,
            [str(n) for n in reply.content["names"]["names"]],
            reply.content["theta0"].to_numpy_ndarrays()[0],
            reply.content["log_mask"].to_numpy_ndarrays()[0],
        ))
    names, x0, mask = agree(sites)
    return names, x0, mask, len(sites), biggest_batch


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


def _dp_options(run_config, estimator: str):
    """The validated dp knobs, or None when `dp=false`. Pure function: unit-tested."""
    if not bool(run_config.get("dp", False)):
        return None
    if estimator == "pooled":
        raise ValueError(
            "dp=true cannot use estimator='pooled': the naive-pooled objective calibrates "
            "its plug-in random effects on the whole data set, so it has no per-subject term "
            "to clip and no bounded sensitivity. Use laplace, focei or ghq."
        )
    dp = {
        "clip": float(run_config.get("dp-clip", 1.0)),
        "noise-multiplier": float(run_config.get("dp-noise-multiplier", 1.0)),
        "rounds": int(run_config.get("dp-rounds", 50)),
        "lr": float(run_config.get("dp-lr", 0.05)),
        "delta": float(run_config.get("dp-delta", 1.0e-5)),
        "final-value": bool(run_config.get("dp-final-value", False)),
        "value-clip": float(run_config.get("dp-value-clip", 100.0)),
        "clip-mode": str(run_config.get("dp-clip-mode", "joint")),
    }
    bad = [k for k in ("clip", "noise-multiplier", "lr", "value-clip") if dp[k] <= 0]
    if dp["rounds"] < 1:
        bad.append("rounds")
    if not 0.0 < dp["delta"] < 1.0:
        bad.append("delta")
    if bad:
        raise ValueError(
            f"invalid dp settings {sorted('dp-' + k for k in bad)}: dp-clip, "
            "dp-noise-multiplier, dp-lr and dp-value-clip must be positive, dp-rounds at "
            "least 1, and dp-delta strictly between 0 and 1"
        )
    if dp["clip-mode"] not in ("joint", "per-group"):
        raise ValueError(
            f"invalid dp-clip-mode {dp['clip-mode']!r}: expected 'joint' or 'per-group'"
        )
    # Group resolution needs the parameter names, which only exist after prepare; here we
    # only parse and validate the two string-encoded overrides.
    dp["groups-override"] = task.parse_group_mapping(run_config.get("dp-groups", ""))
    per_group = task.parse_group_mapping(run_config.get("dp-clip-per-group", ""))
    dp["clip-per-group"] = {g: float(c) for g, c in per_group.items()}
    if any(c <= 0 for c in dp["clip-per-group"].values()):
        raise ValueError("dp-clip-per-group values must all be positive")
    if dp["clip-mode"] == "joint" and (dp["groups-override"] or dp["clip-per-group"]):
        raise ValueError(
            "dp-groups and dp-clip-per-group only apply when dp-clip-mode='per-group'"
        )
    return dp


def _run_dp(grid, context, config, dp, names, x0, mask, num_sites, biggest_batch):
    """The DP fit: fixed-schedule Adam on per-subject-clipped, Gaussian-noised gradients.

    Not L-BFGS: a line search re-evaluates the objective to test a step, which on a noisy
    gradient is meaningless and a fresh budget charge, and the objective is not released
    under dp at all. Adam spends exactly `dp-rounds` releases, which is what makes the
    round count the budget knob; there is no convergence test since every gate-able
    quantity is noisy. Writes results.json with the fit and the spent (eps, delta); nothing
    un-noised is reported - no per-site contribution, no objective trajectory, and no
    objective unless dp-final-value asked for one.
    """
    estimator = str(context.run_config["estimator"])
    ghq_level = int(context.run_config["ghq-level"])
    results_path = Path(str(context.run_config.get("results-path", "results.json"))).resolve()
    s = task.precondition_scale(x0, mask)

    # Per-group DP: resolve the coordinate->group split now that prepare reported the names,
    # and log it once (group membership is model structure, not data). Everything downstream
    # reduces to joint clipping at C_total = sqrt(sum C_g^2), the release's L2 sensitivity, so
    # the accountant needs no per-group special-casing.
    group_ids = group_names = group_clips = clip_total = None
    if dp["clip-mode"] == "per-group":
        group_ids, group_names = task.dp_resolve_groups(names, dp["groups-override"])
        group_clips = task.dp_group_clips(group_names, dp["clip"], dp["clip-per-group"])
        clip_total = task.dp_clip_total(group_clips)
        members = {g: [n for n, i in zip(names, group_ids) if group_names[i] == g]
                   for g in group_names}
        log(INFO, "dp per-group clipping: %d groups, C_total=%.4g", len(group_names), clip_total)
        for g, clip in zip(group_names, group_clips):
            log(INFO, "  group %-10s clip=%.4g  params=%s", g, clip, members[g])
        unmatched = task.dp_unmatched_group_names(names, dp["groups-override"])
        if unmatched:
            log(WARNING, "dp per-group: %s did not match a variance marker and defaulted to "
                "'location'; set dp-groups to reclassify if wrong (does not affect the "
                "privacy bound)", unmatched)

    dp_unit = "subject" if biggest_batch <= 1 else (
        f"random-effect batch (the grouping level; the largest batch holds {biggest_batch} "
        "subjects, so add/remove-one applies to the batch, not the subject)"
    )
    log(INFO, "DIFFERENTIAL PRIVACY: %s estimator, %d sites, unit %s, clip-mode %s",
        estimator, num_sites, dp_unit, dp["clip-mode"])

    rounds = 0
    slowest = 0.0
    t0 = time.perf_counter()

    def dp_release(theta: np.ndarray, release: str) -> np.ndarray:
        """One dp round: the sum of the sites' CLIPPED, NOISED vectors, and nothing else."""
        nonlocal rounds, slowest
        rounds += 1
        rnd = rounds
        t_round = time.perf_counter()
        group_cfg = ({"dp-group-ids": group_ids, "dp-group-clips": group_clips}
                     if dp["clip-mode"] == "per-group" and release == "gradient" else {})
        cfg = ConfigRecord({**dict(config.items()), "dp-sites": num_sites,
                            "dp-release": release, "dp-precond": s.tolist(), **group_cfg})
        replies = _send_all(grid, "query", {"theta": ArrayRecord([theta]), "config": cfg},
                            f"round {rnd}")
        vec = np.sum([r.content["release"].to_numpy_ndarrays()[0] for r in replies], axis=0)
        if not np.all(np.isfinite(vec)):
            raise SiteFailure(f"round {rnd}: non-finite dp release")
        slowest = max(slowest, time.perf_counter() - t_round)
        return vec

    def dp_optimize() -> np.ndarray:
        """Fixed-schedule Adam ascent in the preconditioned coordinate z (theta = x0 + s*z)."""
        z = np.zeros_like(x0)
        m = np.zeros_like(x0)
        v = np.zeros_like(x0)
        for t in range(1, dp["rounds"] + 1):
            g = dp_release(x0 + s * z, "gradient")
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            z = z + dp["lr"] * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1.0e-8)
            log(INFO, "dp round %d/%d: |noisy grad|=%.3e", t, dp["rounds"], np.linalg.norm(g))
        return z

    z_star = dp_optimize()
    theta_star = x0 + s * z_star
    natural = task.to_natural(theta_star, mask)  # server has the log mask, needs no Julia
    objective = float(dp_release(theta_star, "value")[0]) if dp["final-value"] else None
    wall = time.perf_counter() - t0

    releases = dp["rounds"] + (1 if dp["final-value"] else 0)
    dp_block = {
        "enabled": True,
        "adjacency": task.DP_ADJACENCY,
        "unit": dp_unit,
        # eps depends only on the noise multiplier and round count: per-group clipping is
        # exactly as private as joint at C_total, so the accountant is unchanged.
        "epsilon": task.dp_epsilon(releases, dp["noise-multiplier"], dp["delta"]),
        "delta": dp["delta"],
        "releases": releases,
        "sites": num_sites,
        "clip-mode": dp["clip-mode"],
        "noise": "distributed: each site adds N(0, (sigma*clip)^2 / sites)",
        "clip": dp["clip"],
        "noise-multiplier": dp["noise-multiplier"],
        "rounds": dp["rounds"],
        "lr": dp["lr"],
        "final-value": dp["final-value"],
        "value-clip": dp["value-clip"],
        # Without SecAgg the server also sees each site's OWN noised release, carrying only
        # its 1/S noise share: for that site's subjects, against the server, the budget is
        # sqrt(S) larger. SecAgg (deployment only, see the DP docs) would close this.
        "epsilon-per-site-vs-server": task.dp_epsilon(
            releases, dp["noise-multiplier"] / np.sqrt(num_sites), dp["delta"]),
    }
    if dp["clip-mode"] == "per-group":
        dp_block["groups"] = dict(zip(names, [group_names[i] for i in group_ids]))
        dp_block["group-clips"] = dict(zip(group_names, group_clips))
        dp_block["clip-total"] = clip_total

    results = {
        "converged": None,
        "message": (f"fixed-schedule Adam, {dp['rounds']} dp rounds; no convergence test is "
                    "possible on noisy gradients"),
        "objective": objective,
        "names": names,
        "theta_natural": dict(zip(names, natural.tolist())),
        "theta_transformed": dict(zip(names, theta_star.tolist())),
        "config": {"model": str(context.run_config["model"]), "estimator": estimator,
                   "ghq-level": ghq_level},
        "rounds": rounds,
        "timings": {"total-seconds": round(wall, 3), "slowest-round-seconds": round(slowest, 3)},
        "dp": dp_block,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    log(INFO, "DP FEDERATED FIT (%s, %d sites, %d dp rounds)", estimator, num_sites, rounds)
    if objective is None:
        log(INFO, "  loglik=not released (dp)  wall=%.1fs", wall)
    else:
        log(INFO, "  loglik=%.10f (noised)  wall=%.1fs", objective, wall)
    log(INFO, "  %-14s %16s %16s", "parameter", "natural", "transformed")
    for name, nat, tr in zip(names, natural, theta_star):
        log(INFO, "  %-14s %16.8g %16.8g", name, nat, tr)
    log(INFO, "  DIFFERENTIAL PRIVACY ACTIVE: (eps=%.4g, delta=%.3g) spent over %d releases, "
        "adjacency %s, unit %s", dp_block["epsilon"], dp["delta"], releases,
        task.DP_ADJACENCY, dp_unit)
    log(INFO, "  no per-site contribution, objective trajectory or un-noised quantity is "
        "reported under dp")
    log(INFO, "  results written to %s", results_path)
    log(INFO, "PASS: DP federated fit complete (eps=%.4g at delta=%.3g)", dp_block["epsilon"],
        dp["delta"])


def _fit(grid: Grid, context: Context) -> None:
    model = str(context.run_config["model"])
    estimator = str(context.run_config["estimator"])
    ghq_level = int(context.run_config["ghq-level"])
    seed = int(context.run_config["data-seed"])
    source = str(context.run_config["data-source"])
    max_rounds = int(context.run_config["max-rounds"])
    fail_site = int(context.run_config["fail-site"])
    acceptance = task.spec(model).acceptance
    dp = _dp_options(context.run_config, estimator)
    if fail_site >= 0:
        log(INFO, "fault injection active (testing only): site %d will raise", fail_site)
    # The "dp" flag and the scalar dp knobs travel in the wire config; the override dicts
    # stay server-side (they only feed the group resolution below).
    config = ConfigRecord({
        "model": model, "estimator": estimator, "ghq-level": ghq_level,
        "dp": dp is not None,
        **({} if dp is None else {f"dp-{k}": v for k, v in dp.items()
                                  if not isinstance(v, dict)}),
    })

    # Prepare round: sites warm up and hand over the shared start point, the log mask (which
    # coordinates are log-scaled, so the server reports natural-scale numbers without Julia)
    # and the parameter names. No Julia on the server; every later round is a warm eval.
    t_prep = time.perf_counter()
    names, x0, mask, num_sites, biggest_batch = prepare(grid, config)
    log(INFO, "prepare round wall=%.1fs", time.perf_counter() - t_prep)

    if dp is not None:
        _run_dp(grid, context, config, dp, names, x0, mask, num_sites, biggest_batch)
        return

    # Neural model: the additivity of the site (value, gradient) IS the acceptance gate (the
    # headline exact-FL property), checked at theta0 by a self-contained child probe.
    if acceptance == "nn":
        log(INFO, "NN additivity gate: sum over sites vs pooled-data call at theta0")
        probe = _child("probe", model, estimator, ghq_level, seed, source)["probes"]["laplace"]
        log(INFO, "  value_rel=%.3e gradient_rel=%.3e", probe["value_rel"], probe["gradient_rel"])
        if probe["value_rel"] >= 1e-8 or probe["gradient_rel"] >= 1e-8:
            raise RuntimeError(
                f"NN additivity gate failed: value_rel={probe['value_rel']:.3e} "
                f"gradient_rel={probe['gradient_rel']:.3e} (tol 1e-8) - the summed site "
                "contributions are not the pooled-data value/gradient"
            )
        log(INFO, "PASS: NN site contributions are additive (value %.1e, gradient %.1e)",
            probe["value_rel"], probe["gradient_rel"])

    log(INFO, "model=%s estimator=%s data-source=%s params=%d start(natural[:5])=%s",
        model, estimator, source, len(names), task.to_natural(x0, mask)[:5])

    # DEMO ONLY: the pooled fit_model reference. For the PK/growth models it is the
    # acceptance comparison run AFTER convergence. For the neural model, whose ~86 weights
    # are non-identifiable (permutation/sign symmetries), the federated fit is warm-started
    # from the pooled optimum so the objective comparison is meaningful; a real deployment
    # has no pooled dataset and would warm-start from a federated naive-pooled pass instead
    # (naive-pooled is itself a per-subject sum, so it federates). Production deletes this.
    ref = None
    if acceptance == "nn":
        ref = _child("fit", model, estimator, ghq_level, seed, source)
        if ref["names"] != names:
            raise RuntimeError(f"pooled reference order {ref['names']} != sites' {names}")
        x0 = np.asarray(ref["theta_transformed"], dtype=float)  # warm start (demo-only)
        log(INFO, "NN federated fit warm-started from the pooled optimum (demo-only)")

    rounds = 0
    max_round_wall = 0.0
    t0 = time.perf_counter()

    # Preconditioning, NoLimits' own rule (see task.precondition_scale): optimize z with
    # theta = x0 + s * z, so grad_z = s * grad_theta. Raw transformed coordinates mix a
    # volume of ~8 with unit-size log-parameters, which costs L-BFGS-B extra evaluations.
    s = task.precondition_scale(x0, mask)

    def federated(z: np.ndarray):
        nonlocal rounds, max_round_wall
        rounds += 1
        t_round = time.perf_counter()
        x = x0 + s * np.asarray(z, dtype=float)
        sites = broadcast(grid, x, config, rnd=rounds)
        value = sum(v for _, v, _ in sites)
        grad = np.sum([g for _, _, g in sites], axis=0)
        round_wall = time.perf_counter() - t_round
        max_round_wall = max(max_round_wall, round_wall)
        log(INFO, "round %d: loglik=%.10f |grad|=%.3e sites=%d wall=%.2fs", rounds, value,
            np.linalg.norm(grad), len(sites), round_wall)
        return -value, -(s * grad)  # L-BFGS-B minimizes; the sites report a log-likelihood

    # maxfun caps function evaluations, i.e. federated rounds - the round guard. maxiter
    # alone would not: line searches spend extra evaluations per iteration. A truncated
    # run leaves res.success False and fails the acceptance below, rather than passing
    # off a half-optimized theta as the optimum.
    res = minimize(
        federated, np.zeros_like(x0), method="L-BFGS-B", jac=True,
        options={"maxiter": max_rounds, "maxfun": max_rounds},
    )
    wall = time.perf_counter() - t0
    theta_star = x0 + s * res.x

    # Final round at the optimum: also gives the per-site contributions to report.
    final = broadcast(grid, theta_star, config, rnd=rounds + 1)
    rounds += 1
    fed_value = sum(v for _, v, _ in final)
    fed_natural = task.to_natural(theta_star, mask)

    log(INFO, "converged=%s (%s)", res.success, res.message)
    log(INFO, "federated loglik=%.10f evaluation rounds=%d wall=%.1fs (%.2fs/round, "
        "slowest round %.2fs)", fed_value, rounds, wall, wall / max(rounds, 1),
        max_round_wall)
    for site_id, value, _ in sorted(final):
        log(INFO, "  site %d contribution: %.10f", site_id, value)

    # DEMO ONLY: the pooled reference the acceptance compares against (already fetched for
    # the neural model's warm start). Production deployments delete this call.
    if ref is None:
        ref = _child("fit", model, estimator, ghq_level, seed, source)
        if ref["names"] != names:
            raise RuntimeError(f"pooled reference order {ref['names']} != sites' {names}")
    pooled_natural = np.asarray(ref["theta_natural"], dtype=float)
    pooled_value = float(ref["value"])
    value_rel = abs(fed_value - pooled_value) / abs(pooled_value)
    theta_rel = np.abs(fed_natural - pooled_natural) / np.abs(pooled_natural)

    # Neural model: ~86 weights are non-identifiable (permutation/sign symmetries), so valid
    # fits agree in objective/predictions while differing in weights. The additivity gate
    # above is the acceptance; here we only REPORT the objective agreement.
    if acceptance == "nn":
        log(INFO, "NN objective agreement (federated fit vs pooled fit_model): "
            "federated=%.8f pooled=%.8f rel.diff=%.3e", fed_value, pooled_value, value_rel)
        log(INFO, "PASS: NN federation is exact (additivity gated); objective agreement "
            "%.3e reported, parameter-wise agreement not claimed (weights non-identifiable)",
            value_rel)
        return

    log(INFO, "federated theta*(natural) = %s", dict(zip(names, fed_natural.tolist())))
    log(INFO, "ACCEPTANCE (federated vs pooled fit_model, model=%s data-source=%s)",
        model, source)
    log(INFO, "  %-8s %14s %14s %10s", "param", "federated", "pooled", "rel.diff")
    for name, f, p, r in zip(names, fed_natural, pooled_natural, theta_rel):
        log(INFO, "  %-8s %14.8f %14.8f %10.2e", name, f, p, r)
    log(INFO, "  %-8s %14.8f %14.8f %10.2e", "loglik", fed_value, pooled_value, value_rel)

    if estimator == "pooled":
        log(INFO, "%s", POOLED_CAVEAT)
    if estimator == "ghq":
        # GHQ's quadrature objective is ROUGH on this model - NoLimits itself warns that
        # levels above 3 can cancel in the signed logsumexp, and a batch can fall back to
        # the level-1 rule - so scipy's L-BFGS-B and fit_model's Optim LBFGS settle in
        # different local optima and a parameter-wise gate is unreachable in either
        # direction (measured: at level 3 the federated optimum is 3.1e-02 BETTER than
        # fit_model's, at level 5 2.3e-02 worse). What federation has to guarantee is that
        # the summed site objective IS the pooled objective (the additivity probe: 2e-16)
        # and that optimizing it loses nothing, so the gate here is one-sided.
        if fed_value < pooled_value - 1.0e-6 * abs(pooled_value):
            raise RuntimeError(
                f"acceptance failed: federated GHQ optimum {fed_value:.8f} is worse than the "
                f"pooled fit_model optimum {pooled_value:.8f} (rel {value_rel:.3e}); the "
                "objective is exactly additive, so this is an optimizer-path loss - lower "
                "ghq-level (levels 1-3 are the numerically stable range) or raise max-rounds"
            )
        log(INFO, "PASS: federated GHQ optimum is no worse than the pooled fit (%.8f vs "
            "%.8f); parameter-wise agreement is not claimed for GHQ, see the README",
            fed_value, pooled_value)
        return
    # pooled's plug-in eta makes the objective nearly flat in the omegas (1e-2); otherwise
    # the strict model tolerance (1e-3 for the PK and growth models).
    theta_tol = 1.0e-2 if estimator == "pooled" else task.spec(model).param_tol
    if value_rel >= 1e-6 or theta_rel.max() >= theta_tol:
        raise RuntimeError(
            f"acceptance failed: objective rel {value_rel:.3e} (tol 1e-6), "
            f"worst parameter rel {theta_rel.max():.3e} (tol {theta_tol:.0e})"
        )
    log(INFO, "PASS: federated optimum matches the pooled fit (objective %.2e, worst param %.2e)",
        value_rel, theta_rel.max())
