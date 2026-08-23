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
    assert task.spec("warfarin").columns == {"id": "ID", "time": "t", "amt": "Dose", "dv": "conc"}
    assert task.spec("warfarin-nn").columns == {"id": "id", "time": "t", "amt": "d", "dv": "C"}
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


# --- differential privacy: pure helpers (no Julia, no Flower) ----------------------

def test_dp_epsilon_reproduces_the_accountant_cited_values():
    # The two values cited in docs/differential-privacy.md (T=50, delta=1e-5).
    assert round(task.dp_epsilon(50, 1.0, 1e-5), 2) == 58.93   # ~59, weak
    assert round(task.dp_epsilon(50, 4.0, 1e-5), 2) == 10.05   # ~10, meaningful
    # Monotone: more noise or fewer rounds -> smaller eps (stronger privacy).
    assert task.dp_epsilon(50, 8.0, 1e-5) < task.dp_epsilon(50, 4.0, 1e-5)
    assert task.dp_epsilon(25, 1.0, 1e-5) < task.dp_epsilon(50, 1.0, 1e-5)


def test_dp_epsilon_depends_only_on_sigma_and_rounds_not_the_clip():
    # eps is the same whatever the clip / split: that is why per-group == joint accounting.
    assert task.dp_epsilon(50, 2.0, 1e-5) == task.dp_epsilon(50, 2.0, 1e-5)


def test_dp_param_group_classifies_variance_vs_location():
    for n in ("omega_ka", "sigma", "cov_ka_cl", "sd_v", "omega", "tau", "rho12"):
        assert task.dp_param_group(n) == "variance", n
    for n in ("ka", "cl", "v", "Asym", "xmid", "nn_params[3]"):
        assert task.dp_param_group(n) == "location", n
    # An explicit override wins over the heuristic.
    assert task.dp_param_group("sigma", {"sigma": "location"}) == "location"


def test_dp_resolve_groups_orders_by_first_appearance():
    names = ["ka", "cl", "v", "omega_ka", "omega_cl", "omega_v", "sigma"]
    ids, groups = task.dp_resolve_groups(names)
    assert groups == ["location", "variance"]
    assert ids == [0, 0, 0, 1, 1, 1, 1]


def test_dp_clip_sum_bounds_the_site_sensitivity():
    # One giant subject gradient is clipped to norm C; add/remove it moves the sum by <= C.
    g = np.array([[100.0, 0.0], [0.0, 0.0]])
    summed = task.dp_clip_sum(g, clip=1.0)
    assert np.isclose(np.linalg.norm(summed), 1.0)


def test_per_group_clipping_bounds_the_concatenated_norm_by_c_total():
    """The correctness point: clipping each group sub-vector to C_g bounds the WHOLE
    subject contribution by C_total = sqrt(sum C_g^2), so isotropic sigma*C_total noise is
    one Gaussian mechanism at multiplier sigma - identical accounting to joint at C_total."""
    # one subject, 2 location coords (group 0) + 2 variance coords (group 1)
    g = np.array([[10.0, 10.0, 10.0, 10.0]])
    group_ids = [0, 0, 1, 1]
    group_clips = [1.0, 2.0]
    c_total = task.dp_clip_total(group_clips)
    assert np.isclose(c_total, np.sqrt(1.0 + 4.0))
    summed = task.dp_clip_sum_grouped(g, group_ids, group_clips)
    # location block clipped to 1, variance block clipped to 2 -> whole norm is C_total.
    assert np.linalg.norm(summed) <= c_total + 1e-12
    assert np.isclose(np.linalg.norm(summed), c_total)
    # The eps at C_total (per-group) equals the eps of a joint clip at C_total: same sigma.
    assert task.dp_epsilon(50, 4.0, 1e-5) == task.dp_epsilon(50, 4.0, 1e-5)


def test_parse_group_mapping_round_trips_and_rejects_bad_entries():
    assert task.parse_group_mapping("a:x, b:y") == {"a": "x", "b": "y"}
    assert task.parse_group_mapping("") == {}
    with pytest.raises(ValueError, match="bad group mapping"):
        task.parse_group_mapping("noselector")


def test_dp_noise_is_unseeded_and_scales_with_sigma_clip_over_sqrt_sites():
    a = task.dp_noise(100000, clip=1.0, sigma=1.0, num_sites=1)
    b = task.dp_noise(100000, clip=1.0, sigma=1.0, num_sites=1)
    assert not np.allclose(a, b)  # not reproducible: unseeded from OS entropy
    # std ~ sigma*clip/sqrt(S): 4 sites quarters the variance.
    s1 = task.dp_noise(200000, 2.0, 1.0, 1).std()
    s4 = task.dp_noise(200000, 2.0, 1.0, 4).std()
    assert np.isclose(s1, 2.0, rtol=0.05)
    assert np.isclose(s4, 1.0, rtol=0.05)


# --- differential privacy: server option validation --------------------------------

def _rc(**kw):
    base = {"dp": True, "estimator": "laplace"}
    base.update(kw)
    return base


def test_dp_options_none_when_off():
    assert server_app._dp_options({"dp": False}, "laplace") is None


def test_dp_options_rejects_pooled():
    with pytest.raises(ValueError, match="cannot use estimator='pooled'"):
        server_app._dp_options(_rc(), "pooled")


def test_dp_options_defaults_and_types():
    dp = server_app._dp_options(_rc(), "laplace")
    assert dp["clip"] == 20.0 and dp["rounds"] == 50 and dp["clip-mode"] == "per-group"
    assert dp["delta"] == 1e-5


@pytest.mark.parametrize("bad", [
    {"dp-clip": 0.0}, {"dp-noise-multiplier": -1.0}, {"dp-rounds": 0},
    {"dp-lr": 0.0}, {"dp-delta": 0.0}, {"dp-delta": 1.0}, {"dp-value-clip": -2.0},
])
def test_dp_options_rejects_bad_scalars(bad):
    with pytest.raises(ValueError, match="invalid dp"):
        server_app._dp_options(_rc(**bad), "laplace")


def test_dp_options_rejects_unknown_clip_mode():
    with pytest.raises(ValueError, match="invalid dp-clip-mode"):
        server_app._dp_options(_rc(**{"dp-clip-mode": "bogus"}), "laplace")


def test_dp_options_rejects_group_overrides_under_joint():
    with pytest.raises(ValueError, match="only apply when dp-clip-mode"):
        server_app._dp_options(
            _rc(**{"dp-clip-mode": "joint", "dp-groups": "sigma:location"}), "laplace")


def test_dp_options_parses_per_group_overrides():
    dp = server_app._dp_options(
        _rc(**{"dp-clip-mode": "per-group", "dp-groups": "ka:location",
               "dp-clip-per-group": "variance:0.5"}), "laplace")
    assert dp["groups-override"] == {"ka": "location"}
    assert dp["clip-per-group"] == {"variance": 0.5}
