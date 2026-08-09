# Integrated TRIVAX Runtime — preliminary findings

The experimental runtime now integrates observation routing, delay estimation,
temporal credit assignment, probabilistic regime inference and adaptive control.

## Pipeline

```text
sensor
  -> ObservationRouter
  -> DelayEstimator
  -> temporal hold policy
  -> ProbabilisticRegimeController
  -> actuator action
```

## Preliminary end-to-end result

Synthetic delayed-feedback experiments indicate three regimes:

1. **No/very low delay:** the integrated runtime may add a small overhead relative
   to the raw v0.3 controller because protection and identification mechanisms are
   active even when little protection is required.
2. **Moderate delay (roughly 1–4 samples in the current benchmark):** automatic
   delay identification plus temporal hold produces a large improvement over a
   controller that updates every sample using stale feedback.
3. **Long delay:** the simple policy `hold_period = delay + 1` becomes too
   conservative for a moving optimum. Correct delay identification alone is not
   sufficient; predictive or multi-rate temporal control is required.

These are benchmark-specific observations, not universal performance claims.

## Scientific consequence

Delay estimation and delay compensation must be evaluated as separate mechanisms.
The estimator can identify a lag accurately while the chosen compensation policy
can still be suboptimal. Future work should therefore compare:

- fixed hold;
- bounded/multi-rate hold;
- predictive action credit;
- action-history attribution;
- model-free short-horizon prediction.

The current runtime remains experimental and is not promoted as a production
controller.
