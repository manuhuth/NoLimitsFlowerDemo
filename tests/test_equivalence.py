"""Slow acceptance tests: federated == pooled, and additivity, across the 4-model catalog.

Marked slow because each boots Julia per simulated site and compiles the model. The
neural model is the heaviest (87 parameters, plus a child additivity probe and a child
pooled fit), so its federated fit is round-capped and marked `veryslow` for CI subsetting.

The assertions live in the ServerApp (the acceptance gate per model), so a failing run
fails there and these tests only establish that the run reached PASS. Additivity is
checked directly through `task.additivity_probe` (one Julia boot, no federation).
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nolimits_flower import task

REPO = Path(__file__).resolve().parents[1]
# Equal client and init CPUs size the Ray actor pool to ONE actor (floor(init/client)),
# the only way flwr 1.33 pins partitions to a process: that one actor builds every site in
# the prepare round, so no later round re-pays a DataModel build.
FEDERATION = "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
FLWR = str(Path(sys.executable).parent / "flwr")

warfarin_cached = pytest.mark.skipif(
    not task.WARFARIN_CACHE.exists(),
    reason="needs data/warfarin.csv; one online run or `task fit warfarin` writes it",
)


def _flwr(*args: str, timeout: float) -> str:
    # flwr shells out to flower-superlink/flower-superexec by bare name, so the venv bin
    # has to be on PATH even when the venv is not activated.
    env = {**os.environ, "PATH": f"{Path(FLWR).parent}{os.pathsep}{os.environ['PATH']}"}
    proc = subprocess.run([FLWR, *args], cwd=REPO, capture_output=True, text=True,
                          timeout=timeout, env=env)
    return proc.stdout + proc.stderr


def run_federated(run_config: str, timeout: float = 1800.0) -> str:
    """Submit a run, poll its log until it finishes, return the log text."""
    submit = _flwr("run", ".", "--format", "json", "--run-config", run_config,
                   "--federation-config", FEDERATION, timeout=120.0)
    payload = json.loads(re.search(r"\{.*\}", submit, re.S).group(0))
    assert payload.get("success"), f"flwr run refused the app: {submit[-2000:]}"
    run_id = payload["run-id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        log = _flwr("log", str(run_id), "--show", timeout=120.0)
        # Only TERMINAL outcomes end the poll. The NN additivity gate logs an early
        # "PASS: NN site contributions are additive" mid-run; matching a bare "PASS:"
        # would return before the fit's terminal "PASS: NN federation is exact".
        if re.search(r"PASS: (?:federated|NN federation|DP federated)|ABORTED|"
                     r"acceptance failed|gate failed|Exit Code", log):
            return log
        time.sleep(10.0)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


def _probe(model: str, timeout: float = 2400) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "nolimits_flower.task", "probe", model],
        cwd=REPO, capture_output=True, text=True, timeout=timeout,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("POOLED_JSON "))
    return json.loads(line[len("POOLED_JSON "):])


# --- additivity, ALWAYS, all four models (the exact-FL property) -------------------

@pytest.mark.slow
@pytest.mark.parametrize("model", ["theophylline", "orange", "theoph-pooled",
                                   pytest.param("warfarin", marks=warfarin_cached),
                                   pytest.param("warfarin-nn", marks=warfarin_cached)])
def test_site_contributions_add_up(model):
    """Sum over sites of (value, gradient) == the pooled-data call, to 1e-8, per estimator.

    All estimators are per-subject sums, so this is exact regardless of the model (PK, ODE,
    neural, or algebraic growth). The neural model checks `laplace` only; the naive-pooled
    model checks `mle` and `map` (map summed through the site-0 prior-carrier rule)."""
    probes = _probe(model)["probes"]
    for name, p in probes.items():
        print(f"{model}/{name}: value_rel={p['value_rel']:.3e} gradient_rel={p['gradient_rel']:.3e}")
        assert p["value_rel"] < 1e-8, (model, name, p)
        assert p["gradient_rel"] < 1e-8, (model, name, p)


# --- full federated fit == pooled fit, PK + growth models -------------------------

@pytest.mark.slow
@warfarin_cached
def test_warfarin_federated_fit_matches_pooled():
    log = run_federated('model="warfarin" data-source="warfarin"')
    assert "PASS:" in log, log[-4000:]
    # One-actor pool: every post-prepare round is warm (~0.1 s). A tens-of-seconds round
    # means the pool grew and re-paid a build.
    slowest = float(re.search(r"slowest round ([\d.]+)s", log).group(1))
    print(f"warfarin slowest post-prepare round: {slowest:.2f}s")
    assert slowest < 5.0, f"slowest round {slowest}s: actors are not pinned"


@pytest.mark.slow
def test_theophylline_federated_fit_matches_pooled():
    log = run_federated('model="theophylline"')
    assert "PASS:" in log, log[-4000:]


@pytest.mark.slow
def test_orange_federated_fit_matches_pooled():
    log = run_federated('model="orange"')
    assert "PASS:" in log, log[-4000:]


# --- naive-pooled model: fixed-effects MLE and MAP (no random effects) -------------

@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["mle", "map"])
def test_theoph_pooled_federated_fit_matches_pooled(estimator):
    """Federated MLE/MAP optimum == the single-process fit_model(dm, MLE()/MAP()) on the
    naive-pooled theoph model. MAP sums through the site-0 prior carrier; the strict gate
    (objective 1e-6, parameters 1e-3) lives in the ServerApp."""
    log = run_federated(f'model="theoph-pooled" estimator="{estimator}"')
    assert "PASS: federated optimum matches the pooled fit" in log, log[-4000:]


# --- neural model: additivity gate + reported objective agreement (round-capped) ---

@pytest.mark.veryslow
@warfarin_cached
def test_warfarin_nn_additivity_gate_and_objective_agreement():
    """The NN federated fit gates on additivity (1e-8) and reports objective agreement vs
    the pooled fit; the ~86 weights are non-identifiable so parameters are not compared."""
    log = run_federated('model="warfarin-nn" max-rounds=40', timeout=3000.0)
    assert "PASS: NN federation is exact" in log, log[-4000:]
    assert "NN objective agreement" in log


def _dp_run(clip_mode: str, tmp_path, sigma=0.5, rounds=15) -> dict:
    out = tmp_path / f"dp_{clip_mode}.json"
    log = run_federated(
        f'model="theophylline" dp=true dp-noise-multiplier={sigma} dp-rounds={rounds} '
        f'dp-clip-mode="{clip_mode}" results-path="{out}"'
    )
    assert "PASS: DP federated fit complete" in log, log[-4000:]
    return json.loads(out.read_text())


@pytest.mark.slow
def test_dp_fit_runs_and_reports_finite_epsilon(tmp_path):
    """A DP run completes, writes a dp block with finite eps, and finite theta."""
    res = _dp_run("joint", tmp_path)
    dp = res["dp"]
    assert dp["enabled"] and dp["unit"] == "subject" and dp["clip-mode"] == "joint"
    import math
    assert math.isfinite(dp["epsilon"]) and dp["epsilon"] > 0
    assert dp["releases"] == 15 and dp["sites"] == 3
    # The reported eps IS the accountant on (releases, sigma, delta).
    assert abs(dp["epsilon"] - task.dp_epsilon(15, 0.5, 1e-5)) < 1e-9
    assert all(math.isfinite(v) for v in res["theta_natural"].values())
    # Nothing un-noised leaked: no objective (dp-final-value default off), no per-site block.
    assert res["objective"] is None and "sites" not in res


@pytest.mark.slow
def test_dp_per_group_epsilon_equals_the_joint_equivalent(tmp_path):
    """per-group at C_g with isotropic C_total noise == joint at C_total: same (eps, delta)."""
    joint = _dp_run("joint", tmp_path)["dp"]
    per_group = _dp_run("per-group", tmp_path)["dp"]
    assert per_group["clip-mode"] == "per-group"
    assert set(per_group["groups"].values()) == {"location", "variance"}
    # Same sigma, rounds, delta -> the accountant returns the identical eps for both modes.
    assert per_group["epsilon"] == joint["epsilon"]
    assert per_group["epsilon"] == task.dp_epsilon(15, 0.5, 1e-5)


def _dp_run_pooled(estimator: str, tmp_path, sigma=0.5, rounds=15) -> dict:
    out = tmp_path / f"dp_pooled_{estimator}.json"
    log = run_federated(
        f'model="theoph-pooled" estimator="{estimator}" dp=true dp-noise-multiplier={sigma} '
        f'dp-rounds={rounds} dp-clip-mode="joint" results-path="{out}"'
    )
    assert "PASS: DP federated fit complete" in log, log[-4000:]
    return json.loads(out.read_text())


@pytest.mark.slow
def test_mle_map_dp_run_and_share_epsilon(tmp_path):
    """mle+dp and map+dp both complete with the subject as the clipping unit, and spend the
    SAME (eps, delta): the MAP prior is public, so it never enters the accountant."""
    import math
    mle = _dp_run_pooled("mle", tmp_path)["dp"]
    mp = _dp_run_pooled("map", tmp_path)["dp"]
    for dp in (mle, mp):
        assert dp["enabled"] and dp["unit"] == "subject"
        assert math.isfinite(dp["epsilon"]) and dp["epsilon"] > 0
        assert dp["releases"] == 15 and dp["sites"] == 3
    # eps(map+dp) == eps(mle+dp) at matched knobs, and both == the accountant's value.
    assert mp["epsilon"] == mle["epsilon"] == task.dp_epsilon(15, 0.5, 1e-5)


@pytest.mark.slow
def test_dp_per_subject_leave_one_out_bounded_by_clip():
    """The DP data part's add/remove-one-subject sensitivity is <= the clip bound: dropping
    any one subject's per-individual gradient moves the clipped site sum by at most `clip`.
    This is the MLE/MAP data path (MAP's prior is a separate public offset, added once)."""
    import numpy as np
    import NoLimitsPy as nl
    dm = task.build_data_model(nl, "theoph-pooled",
                               task.dataset("theoph-pooled", nl=nl))
    theta0 = np.asarray(nl.seval("nlf_theta0")(dm), dtype=float)
    mask = np.asarray(nl.seval("nlf_logmask")(dm), dtype=float)
    s = task.precondition_scale(theta0, mask)
    _, grads, maxids = task.dp_batch_contributions(nl, dm, theta0, "mle", 1)
    assert maxids == 1, "each subject must be its own clipping unit for a no-RE model"
    grads = grads * s[None, :]  # preconditioned coordinate the server clips in
    clip = 20.0
    full = task.dp_clip_sum(grads, clip)
    for i in range(grads.shape[0]):
        loo = task.dp_clip_sum(np.delete(grads, i, axis=0), clip)
        assert np.linalg.norm(full - loo) <= clip + 1e-9, i


# --- MCEM: nested federated EM (local E-step, federated M-step) --------------------

def _mcem_probe(model: str, source: str = "simulated", timeout: float = 2400) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "nolimits_flower.task", "mcem-probe", model, "mcem", "5",
         "20260818", source],
        cwd=REPO, capture_output=True, text=True, timeout=timeout,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("POOLED_JSON "))
    return json.loads(line[len("POOLED_JSON "):])


@pytest.mark.slow
def test_mcem_q_additivity_at_fixed_draws():
    """The EXACTNESS proof: at FIXED posterior draws, sum over subjects of the per-subject
    M-step Q (value AND gradient) == the population Q, to machine precision, for both parts
    (q1 observation-side, q2 random-effect distribution). This is why the federated M-step IS
    the pooled M-step."""
    probes = _mcem_probe("warfarin")["probes"]
    assert set(probes) == {"q1", "q2"}, probes
    for part, p in probes.items():
        print(f"warfarin/{part}: value_rel={p['value_rel']:.3e} gradient_rel={p['gradient_rel']:.3e}")
        assert p["value_rel"] < 1e-10, (part, p)
        assert p["gradient_rel"] < 1e-10, (part, p)


@pytest.mark.slow
def test_mcem_federated_fit_matches_pooled():
    """Nested federated MCEM (local E-step + federated M-step over q1 then q2) reproduces
    fit_model(dm, MCEM()) to a Monte-Carlo tolerance; the ServerApp holds the gate."""
    log = run_federated('model="warfarin" data-source="simulated" estimator="mcem"')
    assert "PASS: federated MCEM matches the pooled fit" in log, log[-4000:]


@pytest.mark.slow
def test_one_failing_site_aborts_the_run():
    """A site raising must abort, never yield a partial sum."""
    log = run_federated('model="warfarin" data-source="simulated" fail-site=1 max-rounds=3',
                        timeout=600.0)
    assert "PASS:" not in log
    assert "FEDERATED FIT ABORTED" in log, log[-4000:]
    assert "aborting the federated fit rather than summing the remaining sites" in log
