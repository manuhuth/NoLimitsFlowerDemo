"""Fast tests: no Julia, no Flower runtime, no federation. Seconds."""

import numpy as np
import pytest

from nolimits_flower import server_app, task


def test_partition_sizes_and_disjointness():
    df = task.simulate()
    sites = task.partition(df, 3)
    assert [s["ID"].nunique() for s in sites] == [8, 8, 8]
    assert sum(len(s) for s in sites) == len(df)
    ids = [set(s["ID"]) for s in sites]
    assert set.union(*ids) == set(df["ID"])
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            assert not a & b


def test_partition_keeps_all_rows_of_a_subject_together():
    sites = task.partition(task.simulate(), 5)  # 24 subjects, uneven split
    assert sorted(s["ID"].nunique() for s in sites) == [4, 5, 5, 5, 5]
    for s in sites:
        assert (s.groupby("ID").size() == len(task.TIMES)).all()


def test_partition_rejects_more_sites_than_subjects():
    with pytest.raises(ValueError, match="cannot fill"):
        task.partition(task.simulate(n_subjects=2), 3)


def test_simulate_is_deterministic_per_seed():
    a, b = task.simulate(seed=1), task.simulate(seed=1)
    assert a.equals(b)
    assert not np.allclose(a["conc"], task.simulate(seed=2)["conc"])
    assert task.partition(a, 3)[0].equals(task.partition(b, 3)[0])


def test_simulated_concentrations_are_plausible():
    df = task.simulate()
    assert (df["Dose"] == task.DOSE).all()
    peak = df.groupby("ID")["conc"].max()
    # 100 mg into ~8 L, so peaks land around 10 mg/L, and absorption is fast enough that
    # the profile has already peaked before the first design time.
    assert peak.between(4.0, 25.0).all()
    late = df[df["t"] == max(task.TIMES)]["conc"]
    assert (late < peak.to_numpy()).all()


def test_theta_scale_round_trip():
    # Mixed scales: exp/log for the omegas and sigma, identity for ka, cl, v.
    names = list(task.PARAM_NAMES)
    natural = np.array([task.TRUE_THETA[n] for n in names])
    transformed = np.where([n in task.LOG_SCALED for n in names], np.log(natural), natural)
    assert np.allclose(task.to_natural(transformed, names), natural)
    assert not np.allclose(transformed, natural)  # the scales really do differ


def test_log_scaled_matches_the_model_string():
    """LOG_SCALED is the Julia-free copy of the model's transform; keep them in sync."""
    declared = {
        line.split("=")[0].strip()
        for line in task.MODEL.splitlines()
        if "RealNumber(" in line and "scale=:log" in line
    }
    assert declared == set(task.LOG_SCALED)
    plain = {
        line.split("=")[0].strip()
        for line in task.MODEL.splitlines()
        if "RealNumber(" in line and "scale=:log" not in line
    }
    assert declared | plain == set(task.PARAM_NAMES)


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="unknown estimator"):
        task._method(nl=None, estimator="saem", ghq_level=5)


def test_known_estimators_reach_the_wrapper():
    class FakeNl:
        Laplace = staticmethod(lambda: "laplace-method")
        GHQuadrature = staticmethod(lambda level: f"ghq-{level}")

    assert task._method(FakeNl, "laplace", 5) == "laplace-method"
    assert task._method(FakeNl, "ghq", 7) == "ghq-7"


def test_agree_returns_the_shared_names_and_theta0():
    theta0 = np.array([1.0, -0.9])
    names, out = server_app.agree([
        (0, ["ka", "omega_ka"], theta0),
        (1, ["ka", "omega_ka"], theta0.copy()),
    ])
    assert names == ["ka", "omega_ka"]
    assert np.array_equal(out, theta0)


@pytest.mark.parametrize("bad", [
    (1, ["ka", "omega_cl"], np.array([1.0, -0.9])),   # different names
    (1, ["ka", "omega_ka"], np.array([1.0, -0.8])),    # different theta0
])
def test_agree_rejects_a_site_running_another_model(bad):
    with pytest.raises(server_app.SiteFailure, match="not running the same model"):
        server_app.agree([(0, ["ka", "omega_ka"], np.array([1.0, -0.9])), bad])


def test_agree_rejects_an_empty_prepare_round():
    with pytest.raises(server_app.SiteFailure, match="no sites reported"):
        server_app.agree([])


def test_short_reason_extracts_the_site_exception():
    class Error:
        code = 2
        reason = (
            "<class 'ray.exceptions.RayTaskError(ClientAppException)'>:<'ray::ClientAppActor"
            ".run()\n  File \"client_app.py\", line 51\nRuntimeError: boom\n\n"
            "Exception ClientAppException occurred. Message: fault injection: site 1 refuses"
            " to answer'>"
        )

    assert server_app._short_reason(Error()) == (
        "fault injection: site 1 refuses to answer (error code 2)"
    )


def test_short_reason_survives_an_empty_error():
    class Error:
        code = 1
        reason = ""

    assert "no reason reported" in server_app._short_reason(Error())
