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
        if re.search(r"PASS: (?:federated|NN federation)|ABORTED|acceptance failed|"
                     r"gate failed|Exit Code", log):
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
@pytest.mark.parametrize("model", ["theophylline", "orange",
                                   pytest.param("warfarin", marks=warfarin_cached),
                                   pytest.param("warfarin-nn", marks=warfarin_cached)])
def test_site_contributions_add_up(model):
    """Sum over sites of (value, gradient) == the pooled-data call, to 1e-8, per estimator.

    All estimators are per-subject sums, so this is exact regardless of the model (PK, ODE,
    neural, or algebraic growth). The neural model checks `laplace` only."""
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


# --- neural model: additivity gate + reported objective agreement (round-capped) ---

@pytest.mark.veryslow
@warfarin_cached
def test_warfarin_nn_additivity_gate_and_objective_agreement():
    """The NN federated fit gates on additivity (1e-8) and reports objective agreement vs
    the pooled fit; the ~86 weights are non-identifiable so parameters are not compared."""
    log = run_federated('model="warfarin-nn" max-rounds=40', timeout=3000.0)
    assert "PASS: NN federation is exact" in log, log[-4000:]
    assert "NN objective agreement" in log


@pytest.mark.slow
def test_one_failing_site_aborts_the_run():
    """A site raising must abort, never yield a partial sum."""
    log = run_federated('model="warfarin" data-source="simulated" fail-site=1 max-rounds=3',
                        timeout=600.0)
    assert "PASS:" not in log
    assert "FEDERATED FIT ABORTED" in log, log[-4000:]
    assert "aborting the federated fit rather than summing the remaining sites" in log
