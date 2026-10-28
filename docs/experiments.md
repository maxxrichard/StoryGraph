# Experiment design

## Unit and assignment

The randomisation unit is the listener. `Experiment.assign` hashes `(name, salt, listener_id)`
into 10,000 buckets; variants own contiguous ranges inside the exposed share, so ramping from
1 % to 100 % never moves a listener between arms. Layers use separate salts and are therefore
orthogonal.

## Metrics

| Metric | Type | Role |
|--------|------|------|
| listen_through_rate | ratio (listened / started seconds per listener) | primary |
| conversion_rate | binary per listener | guardrail (non-inferior, margin -0.5 pp) |
| p95_latency_ms | continuous | guardrail (non-inferior, +30 ms) |
| finish_rate, skips_per_session | – | secondary |

All metrics are reduced to one value per listener before testing (`experiments.metrics.per_listener`)
so observations are independent.

## Analysis

* Welch two-sample test on listener-level values.
* CUPED with the same metric over the 14 days before exposure as covariate; theta is estimated on
  the pooled arms. Typical variance reduction on listen-through is 30–50 %.
* Sample-ratio mismatch: chi-square test of arm sizes against configured weights at alpha 0.001.
  An SRM invalidates the test regardless of lift.
* Guardrails are non-inferiority tests on the confidence interval, not point estimates.
* `required_sample_size` gives the per-arm size for the configured minimum detectable effect.

## Rollout

Ramp 1 % → 5 % → 20 % → 50 % → 100 % with 24 h holds. Guardrail breaches auto-roll back to the
previous step. The analysis script exits non-zero on a hold verdict so it can gate the ramp job.
