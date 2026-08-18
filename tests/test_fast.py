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
    assert not np.allclose(a["y"], task.simulate(seed=2)["y"])
    assert task.partition(a, 3)[0].equals(task.partition(b, 3)[0])


def test_theta_scale_round_trip():
    # Every fixed effect is scale=:log, so the wire (transformed) scale is log/exp.
    natural = np.array([task.TRUE_THETA[n] for n in task.PARAM_NAMES])
    assert np.allclose(task.to_natural(np.log(natural)), natural)


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="unknown estimator"):
        task._method(nl=None, estimator="saem", ghq_level=5)


def test_known_estimators_reach_the_wrapper():
    class FakeNl:
        Laplace = staticmethod(lambda: "laplace-method")
        GHQuadrature = staticmethod(lambda level: f"ghq-{level}")

    assert task._method(FakeNl, "laplace", 5) == "laplace-method"
    assert task._method(FakeNl, "ghq", 7) == "ghq-7"


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
