from collections import Counter

import numpy as np
import pytest

from storygraph.config import load_yaml
from storygraph.experiments.analysis import (
    chi2_sf,
    cuped_adjust,
    guardrail_ok,
    norm_ppf,
    norm_sf,
    required_sample_size,
    srm_check,
    welch_ttest,
)
from storygraph.experiments.assignment import Experiment, Variant
from storygraph.experiments.metrics import per_listener, split


def _exp(ramp=1.0):
    return Experiment(
        name="e",
        salt="s",
        variants=(Variant("control", 0.5, "cf"), Variant("treatment", 0.5, "tt")),
        ramp=ramp,
    )


def test_assignment_is_deterministic_and_balanced():
    e = _exp()
    ids = [f"u{i}" for i in range(20000)]
    a = [e.assign(i).name for i in ids]
    assert a == [e.assign(i).name for i in ids]
    c = Counter(a)
    assert abs(c["control"] - c["treatment"]) < 600


def test_ramp_keeps_listeners_stable():
    ids = [f"u{i}" for i in range(10000)]
    ramps = [0.01, 0.05, 0.2, 0.5, 1.0]
    treated = [{i for i in ids if _exp(ramp=r).assign(i).name == "treatment"} for r in ramps]
    # monotone: everyone in treatment at a lower ramp stays in treatment at every higher ramp
    for a, b in zip(treated, treated[1:]):
        assert a <= b
    # exposure share is roughly ramp * weight(treatment) * 10000
    assert 150 < len(treated[1]) < 350
    assert 900 < len(treated[2]) < 1100
    assert 4700 < len(treated[4]) < 5300


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        Experiment(
            name="e",
            salt="s",
            variants=(Variant("control", 0.6, "a"), Variant("treatment", 0.6, "b")),
        )


def test_from_config(repo_root):
    cfg = load_yaml(repo_root / "configs/experiment.yaml")
    e = Experiment.from_config(cfg, layer="homepage_recs")
    assert {v.name for v in e.variants} == {"control", "treatment"}
    assert e.assign("listener-1").name in {"control", "treatment"}


def test_normal_helpers():
    assert norm_sf(0) == pytest.approx(0.5)
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert chi2_sf(3.841, 1) == pytest.approx(0.05, abs=0.01)
    assert chi2_sf(0, 1) == 1.0


def test_welch_detects_lift_and_cuped_reduces_variance():
    rng = np.random.default_rng(0)
    n = 4000
    pre_c, pre_t = rng.normal(0.5, 0.2, n), rng.normal(0.5, 0.2, n)
    c = pre_c * 0.8 + rng.normal(0, 0.1, n)
    t = pre_t * 0.8 + rng.normal(0, 0.1, n) + 0.02
    plain = welch_ttest(c, t, metric="ltr")
    cuped = welch_ttest(c, t, metric="ltr", control_pre=pre_c, treatment_pre=pre_t)
    assert plain.abs_lift == pytest.approx(0.02, abs=0.01)
    assert cuped.variance_reduction > 0.5
    assert (cuped.ci_high - cuped.ci_low) < (plain.ci_high - plain.ci_low)
    assert cuped.significant
    assert "CUPED" in cuped.summary()


def test_welch_null_is_not_significant_most_of_the_time():
    rng = np.random.default_rng(1)
    sig = sum(
        welch_ttest(rng.normal(size=200), rng.normal(size=200)).significant for _ in range(100)
    )
    assert sig < 15


def test_cuped_adjust_theta():
    rng = np.random.default_rng(2)
    x = rng.normal(size=1000)
    y = 2 * x + rng.normal(size=1000)
    adj, theta = cuped_adjust(y, x)
    assert theta == pytest.approx(2.0, abs=0.2)
    assert adj.var() < y.var()


def test_srm_check():
    ok, p = srm_check({"control": 5000, "treatment": 5050}, {"control": 0.5, "treatment": 0.5})
    assert ok and p > 0.1
    ok, p = srm_check({"control": 5000, "treatment": 4000}, {"control": 0.5, "treatment": 0.5})
    assert not ok and p < 1e-6


def test_required_sample_size_scales_with_mde():
    n1 = required_sample_size(0.5, 0.25, 0.01)
    n5 = required_sample_size(0.5, 0.25, 0.05)
    assert n1 > 20 * n5
    with pytest.raises(ValueError):
        required_sample_size(0.5, 0.25, 0.0)


def test_guardrail_logic():
    from storygraph.experiments.analysis import TestResult

    r = TestResult("conv", 0.1, 0.099, -0.001, -0.01, -0.004, 0.002, 0.5, 100, 100)
    assert guardrail_ok(r, "non_inferior", -0.005)
    r2 = TestResult("conv", 0.1, 0.09, -0.01, -0.1, -0.015, -0.005, 0.01, 100, 100)
    assert not guardrail_ok(r2, "non_inferior", -0.005)
    lat = TestResult("p95", 100, 110, 10, 0.1, 5, 15, 0.01, 100, 100)
    assert guardrail_ok(lat, "non_inferior", 30) and not guardrail_ok(lat, "non_inferior", 12)
    with pytest.raises(ValueError):
        guardrail_ok(r, "superior", 0)


def test_per_listener_metrics():
    rows = [
        {
            "listener_id": "a",
            "variant": "treatment",
            "started_seconds": 100,
            "listened_seconds": 50,
            "finished": 0,
            "converted": 0,
        },
        {
            "listener_id": "a",
            "variant": "treatment",
            "started_seconds": 100,
            "listened_seconds": 100,
            "finished": 1,
            "converted": 1,
        },
        {
            "listener_id": "b",
            "variant": "control",
            "started_seconds": 200,
            "listened_seconds": 20,
            "finished": 0,
            "converted": 0,
        },
    ]
    units = per_listener(rows)
    by = {u.listener_id: u for u in units}
    assert (
        by["a"].listen_through == pytest.approx(0.75)
        and by["a"].conversion == 1.0
        and by["a"].finish_rate == 0.5
        and by["a"].sessions == 2
    )
    arms = split(units, "listen_through")
    assert set(arms) == {"treatment", "control"} and arms["control"][0] == pytest.approx(0.1)
