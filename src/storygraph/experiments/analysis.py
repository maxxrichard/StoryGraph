"""Frequentist A/B analysis with CUPED variance reduction, SRM checks and power/sample-size.

No scipy dependency: normal tail probabilities use ``math.erfc``; the chi-square SRM test uses a
Wilson-Hilferty approximation, which is accurate for the 1-3 degrees of freedom we need.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation of the inverse normal CDF (abs err < 1.2e-9)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0,1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > 1 - plow:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def chi2_sf(x: float, df: int) -> float:
    """Wilson–Hilferty approximation to the chi-square survival function."""
    if x <= 0:
        return 1.0
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return norm_sf(z)


@dataclass
class TestResult:
    metric: str
    control_mean: float
    treatment_mean: float
    abs_lift: float
    rel_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    n_control: int
    n_treatment: int
    variance_reduction: float = 0.0

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        return (
            f"{self.metric}: {self.control_mean:.4f} -> {self.treatment_mean:.4f} "
            f"({self.rel_lift:+.2%}, 95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], p={self.p_value:.4f}, "
            f"n={self.n_control}/{self.n_treatment}"
            + (
                f", CUPED var. reduction {self.variance_reduction:.0%}"
                if self.variance_reduction
                else ""
            )
            + ")"
        )


def cuped_adjust(
    y: np.ndarray, x_pre: np.ndarray, theta: float | None = None
) -> tuple[np.ndarray, float]:
    """Return CUPED-adjusted outcomes and the theta used (cov(y,x)/var(x))."""
    if theta is None:
        vx = x_pre.var()
        theta = float(np.cov(y, x_pre, ddof=0)[0, 1] / vx) if vx > 0 else 0.0
    return y - theta * (x_pre - x_pre.mean()), theta


def welch_ttest(
    control: Sequence[float] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    metric: str = "metric",
    alpha: float = 0.05,
    control_pre: Sequence[float] | np.ndarray | None = None,
    treatment_pre: Sequence[float] | np.ndarray | None = None,
) -> TestResult:
    """Two-sided Welch test on listener-level outcomes, optionally CUPED-adjusted with a shared theta."""
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    if len(c) < 2 or len(t) < 2:
        raise ValueError("need at least two observations per arm")
    vr = 0.0
    if control_pre is not None and treatment_pre is not None:
        pooled_y = np.concatenate([c, t])
        pooled_x = np.concatenate(
            [np.asarray(control_pre, float), np.asarray(treatment_pre, float)]
        )
        _, theta = cuped_adjust(pooled_y, pooled_x)
        raw_var = c.var(ddof=1) + t.var(ddof=1)
        c, _ = cuped_adjust(c, np.asarray(control_pre, float), theta)
        t, _ = cuped_adjust(t, np.asarray(treatment_pre, float), theta)
        new_var = c.var(ddof=1) + t.var(ddof=1)
        vr = 1 - new_var / raw_var if raw_var > 0 else 0.0
    diff = t.mean() - c.mean()
    se = math.sqrt(c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t))
    z = diff / se if se > 0 else 0.0
    zcrit = norm_ppf(1 - alpha / 2)
    return TestResult(
        metric=metric,
        control_mean=float(c.mean()),
        treatment_mean=float(t.mean()),
        abs_lift=float(diff),
        rel_lift=float(diff / c.mean()) if c.mean() != 0 else float("nan"),
        ci_low=float(diff - zcrit * se),
        ci_high=float(diff + zcrit * se),
        p_value=float(2 * norm_sf(abs(z))),
        n_control=len(c),
        n_treatment=len(t),
        variance_reduction=float(vr),
    )


def srm_check(
    counts: dict[str, int], expected_weights: dict[str, float], alpha: float = 0.001
) -> tuple[bool, float]:
    """Sample-ratio mismatch: chi-square goodness of fit of observed arm sizes to configured weights.

    Returns (ok, p_value). An SRM means assignment or logging is broken and the test is invalid.
    """
    total = sum(counts.values())
    chi2 = 0.0
    for arm, w in expected_weights.items():
        exp = total * w
        obs = counts.get(arm, 0)
        chi2 += (obs - exp) ** 2 / exp if exp > 0 else 0.0
    p = chi2_sf(chi2, df=max(1, len(expected_weights) - 1))
    return p >= alpha, p


def required_sample_size(
    baseline_mean: float,
    baseline_std: float,
    mde_rel: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Per-arm sample size for a two-sided test detecting a relative lift ``mde_rel``."""
    delta = baseline_mean * mde_rel
    if delta == 0:
        raise ValueError("mde must be non-zero")
    z = norm_ppf(1 - alpha / 2) + norm_ppf(power)
    return int(math.ceil(2 * (z * baseline_std / delta) ** 2))


def guardrail_ok(result: TestResult, direction: str, margin: float) -> bool:
    """Non-inferiority: the treatment is acceptable if the CI excludes harm beyond ``margin``.

    ``margin`` is on the absolute scale and signed so that ``margin < 0`` means "may drop by at most".
    """
    if direction != "non_inferior":
        raise ValueError("only non_inferior guardrails are supported")
    if margin < 0:  # higher is better; CI lower bound must exceed margin
        return result.ci_low > margin
    return (
        result.ci_high < margin
    )  # lower is better (e.g. latency); CI upper bound must be below margin
