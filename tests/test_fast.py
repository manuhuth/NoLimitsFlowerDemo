"""Fast tests: no Julia, no Flower runtime, no federation. Seconds."""

import numpy as np
import pytest

from nolimits_flower import server_app, task


# --- the 4-model catalog ----------------------------------------------------------

def test_catalog_has_the_four_models():
    assert set(task.CATALOG) == {"warfarin", "theophylline", "warfarin-nn", "orange"}


@pytest.mark.parametrize("model", list(task.CATALOG))
def test_spec_returns_a_usable_entry(model):
    sp = task.spec(model)
    assert sp.model.strip() and callable(sp.loader)
    assert sp.primary_id and sp.time_col
    assert sp.num_sites >= 1
    assert sp.acceptance in ("strict", "nn")


def test_spec_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        task.spec("bogus")


def test_dataset_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        task.dataset("bogus")


def test_per_model_primary_id_and_time_col():
    assert (task.spec("warfarin").primary_id, task.spec("warfarin").time_col) == ("ID", "t")
    assert (task.spec("theophylline").primary_id, task.spec("theophylline").time_col) == ("id", "t")
    assert (task.spec("warfarin-nn").primary_id, task.spec("warfarin-nn").time_col) == ("id", "t")
    assert (task.spec("orange").primary_id, task.spec("orange").time_col) == ("Tree", "age")


def test_per_model_column_maps():
    assert task.spec("warfarin").columns == {"id": "ID", "t": "t", "d": "Dose", "C": "conc"}
    assert task.spec("theophylline").columns["Subject"] == "id"
    assert task.spec("theophylline").columns["Time"] == "t"
    assert task.spec("orange").columns == {"Tree": "Tree", "age": "age", "circumference": "circumference"}


def test_nn_is_the_only_additivity_only_model():
    assert task.spec("warfarin-nn").acceptance == "nn"
    assert all(task.spec(m).acceptance == "strict" for m in task.CATALOG if m != "warfarin-nn")


def test_nn_model_pins_the_ffnn_seed():
    """Every site must build IDENTICAL initial weights, or theta0 disagrees and the
    prepare-round agreement fails. The pin lives in the model string."""
    model = task.spec("warfarin-nn").model
    assert "FFNNParameters(" in model
    assert "seed=1234" in model.replace(" ", "")
    # The NN pooled fit is seeded and warm-started (its weights are non-identifiable).
    assert task.spec("warfarin-nn").fit_seed == 1234
    assert task.spec("warfarin-nn").pooled_init


# --- data loaders that need no Julia (vendored/committed CSVs) ---------------------

def test_theophylline_loads_12_subjects():
    df = task.dataset("theophylline")
    assert list(df.columns) == ["id", "t", "Dose", "conc"]
    assert df["id"].nunique() == 12
    assert (df.groupby("id")["Dose"].nunique() == 1).all()  # Dose is a ConstantCovariate


def test_orange_loads_5_trees():
    df = task.dataset("orange")
    assert list(df.columns) == ["Tree", "age", "circumference"]
    assert df["Tree"].nunique() == 5
    assert (df["age"] > 0).all() and (df["circumference"] > 0).all()


def test_theophylline_partitions_into_three_sites():
    sites = task.partition(task.dataset("theophylline"), 3, "id")
    assert [s["id"].nunique() for s in sites] == [4, 4, 4]


def test_orange_partitions_into_three_sites():
    sites = task.partition(task.dataset("orange"), 3, "Tree")
    assert sorted(s["Tree"].nunique() for s in sites) == [1, 2, 2]


# --- warfarin partitioning / simulation (kept) ------------------------------------

def test_partition_sizes_and_disjointness():
    df = task.simulate()
    sites = task.partition(df, 3, "ID")
    assert [s["ID"].nunique() for s in sites] == [8, 8, 8]
    assert sum(len(s) for s in sites) == len(df)
    ids = [set(s["ID"]) for s in sites]
    assert set.union(*ids) == set(df["ID"])
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            assert not a & b


def test_partition_keeps_all_rows_of_a_subject_together():
    sites = task.partition(task.simulate(), 5, "ID")
    assert sorted(s["ID"].nunique() for s in sites) == [4, 5, 5, 5, 5]
    for s in sites:
        assert (s.groupby("ID").size() == len(task.TIMES)).all()


def test_partition_rejects_more_sites_than_subjects():
    with pytest.raises(ValueError, match="cannot fill"):
        task.partition(task.simulate(n_subjects=2), 3, "ID")


def test_simulate_is_deterministic_per_seed():
    a, b = task.simulate(seed=1), task.simulate(seed=1)
    assert a.equals(b)
    assert not np.allclose(a["conc"], task.simulate(seed=2)["conc"])


def test_dataset_simulated_is_the_simulation():
    assert task.dataset("warfarin", source="simulated", seed=7).equals(task.simulate(seed=7))


def test_dataset_rejects_an_unknown_source_for_warfarin():
    with pytest.raises(ValueError, match="unknown data-source"):
        task.dataset("warfarin", source="nonsense")


# --- theta scale glue via the log mask (Julia-free server side) -------------------

def test_to_natural_uses_the_log_mask():
    # warfarin order: ka, cl, v (identity), omega_ka, omega_cl, omega_v, sigma (log)
    theta = np.array([1.0, 0.13, 8.0, np.log(0.4), np.log(0.3), np.log(0.2), np.log(0.5)])
    mask = np.array([0, 0, 0, 1, 1, 1, 1], dtype=float)
    natural = task.to_natural(theta, mask)
    assert np.allclose(natural, [1.0, 0.13, 8.0, 0.4, 0.3, 0.2, 0.5])


def test_precondition_scale_matches_the_nolimits_rule():
    # identity coordinate -> max(|theta0|, 1); log coordinate -> 1.
    theta0 = np.array([1.0, 0.13, 8.0, -0.9, -1.2, -1.6, -0.7])
    mask = np.array([0, 0, 0, 1, 1, 1, 1], dtype=float)
    s = task.precondition_scale(theta0, mask)
    assert np.allclose(s, [1.0, 1.0, 8.0, 1.0, 1.0, 1.0, 1.0])
    # The reparameterization is exact at the start point and invertible.
    assert np.allclose(theta0 + s * ((theta0 * 1.5 - theta0) / s), theta0 * 1.5)


def test_nn_mask_scales_only_sigma():
    """The NN has one log coordinate (sigma) and 86 identity weights near zero, so its
    preconditioning is essentially the identity."""
    theta0 = np.concatenate([[0.0], np.full(86, 0.1)])  # sigma, then weights
    mask = np.concatenate([[1.0], np.zeros(86)])
    s = task.precondition_scale(theta0, mask)
    assert s[0] == 1.0 and np.all(s[1:] == 1.0)


# --- server pure functions --------------------------------------------------------

def test_agree_returns_shared_names_theta0_and_mask():
    theta0, mask = np.array([1.0, -0.9]), np.array([0.0, 1.0])
    names, out, m = server_app.agree([
        (0, ["ka", "omega_ka"], theta0, mask),
        (1, ["ka", "omega_ka"], theta0.copy(), mask.copy()),
    ])
    assert names == ["ka", "omega_ka"]
    assert np.array_equal(out, theta0) and np.array_equal(m, mask)


@pytest.mark.parametrize("bad", [
    (1, ["ka", "omega_cl"], np.array([1.0, -0.9]), np.array([0.0, 1.0])),   # names
    (1, ["ka", "omega_ka"], np.array([1.0, -0.8]), np.array([0.0, 1.0])),   # theta0 (unpinned seed)
])
def test_agree_rejects_a_site_running_another_model(bad):
    good = (0, ["ka", "omega_ka"], np.array([1.0, -0.9]), np.array([0.0, 1.0]))
    with pytest.raises(server_app.SiteFailure, match="not running the same model"):
        server_app.agree([good, bad])


def test_agree_rejects_an_empty_prepare_round():
    with pytest.raises(server_app.SiteFailure, match="no sites reported"):
        server_app.agree([])


# --- estimator selection / error parsing ------------------------------------------

@pytest.mark.parametrize("bad", ["saem", "mcem", "mle", "", "Laplace"])
def test_unknown_estimator_is_rejected(bad):
    with pytest.raises(ValueError, match="unknown estimator"):
        task._method(nl=None, estimator=bad, ghq_level=5)


def test_known_estimators_reach_the_wrapper():
    class FakeNl:
        Laplace = staticmethod(lambda: "laplace-method")
        FOCEI = staticmethod(lambda: "focei-method")
        GHQuadrature = staticmethod(lambda level: f"ghq-{level}")
        Pooled = staticmethod(lambda: "pooled-method")

    assert task._method(FakeNl, "laplace", 5) == "laplace-method"
    assert task._method(FakeNl, "ghq", 7) == "ghq-7"
    assert set(task.ESTIMATORS) == {"laplace", "focei", "ghq", "pooled"}


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
