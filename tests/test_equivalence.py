"""The slow acceptance test: the federated fit must reach the pooled fit_model optimum.

Marked slow (`pytest -m slow`) because it boots one Julia per simulated site and
compiles the model in each of them, ~4 minutes for the default seed.

The assertion itself lives in the ServerApp (objective within 1e-6 relative, every
natural-scale parameter within 1e-3), so a failing acceptance fails the run and this
test only has to establish that the run reached PASS. The federated/pooled table is
echoed into the test output for the record.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FEDERATION = "num-supernodes=3 client-resources-num-cpus=1 init-args-num-cpus=2"


FLWR = str(Path(sys.executable).parent / "flwr")


def _flwr(*args: str, timeout: float) -> str:
    # flwr shells out to `flower-superlink`/`flower-superexec` by bare name, so the venv's
    # bin directory has to be on PATH even when the venv is not activated.
    env = {**os.environ, "PATH": f"{Path(FLWR).parent}{os.pathsep}{os.environ['PATH']}"}
    proc = subprocess.run(
        [FLWR, *args],
        cwd=REPO, capture_output=True, text=True, timeout=timeout, env=env,
    )
    return proc.stdout + proc.stderr


def run_federated(run_config: str, timeout: float = 900.0) -> str:
    """Submit a run, then poll its log until it finishes. Returns the log text.

    `flwr run --stream` can return before the run ends, so the run is submitted with
    `--format json` (which reports the run id immediately) and polled instead.
    """
    submit = _flwr("run", ".", "--format", "json", "--run-config", run_config,
                   "--federation-config", FEDERATION, timeout=120.0)
    payload = json.loads(re.search(r"\{.*\}", submit, re.S).group(0))
    assert payload.get("success"), f"flwr run refused the app: {submit[-2000:]}"
    run_id = payload["run-id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        log = _flwr("log", str(run_id), "--show", timeout=120.0)
        if re.search(r"PASS:|ABORTED|acceptance failed|Exit Code", log):
            return log
        time.sleep(10.0)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


@pytest.mark.slow
def test_federated_fit_matches_the_pooled_fit():
    log = run_federated("data-seed=20260818")
    table = [l for l in log.splitlines() if "ACCEPTANCE" in l or re.search(r"\d\.\d{8}", l)]
    print("\n".join(table))
    assert "PASS:" in log, log[-4000:]


@pytest.mark.slow
def test_one_failing_site_aborts_the_run():
    """The fault-injection knob: a site raising must abort, never yield a partial sum."""
    log = run_federated("fail-site=1 max-rounds=3", timeout=600.0)
    assert "PASS:" not in log
    assert "FEDERATED FIT ABORTED" in log, log[-4000:]
    assert "aborting the federated fit rather than summing the remaining sites" in log
