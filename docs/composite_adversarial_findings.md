# Composite adversarial benchmark — preliminary findings

This note records exploratory findings for the first benchmark that combines multiple stressors in one plant rather than testing them independently.

## Scenario

The plant combines:

- a sinusoidally moving optimum;
- feedback delay changing over time (`2 -> 5 -> 1` samples);
- Gaussian observation noise (`sigma = 0.01`);
- impulsive outliers with 1% probability;
- an abrupt optimum offset at `t = 700`;
- a curvature change from `4.0` to `7.0` at the same point.

The benchmark is implemented in `benchmarks/composite_adversarial.py` and compares:

1. `TrivaxRuntimeV2` with online delay estimation;
2. `HistoricalCreditController` with oracle delay, used only as an upper-reference diagnostic;
3. `ProbabilisticRegimeController` (v0.3);
4. `PerturbAndObserve`.

## Exploratory result

Local exploratory runs show that historical temporal credit remains effective when the correct delay is known. This supports the credit-assignment mechanism itself.

The integrated Runtime V2 remains bounded and generally tracks the moving optimum, but the online delay estimator can become over-confident at spurious lags when several disturbances occur simultaneously. In particular, nonlinear response, moving optimum, outliers, and curvature change can introduce correlations that resemble a delayed causal response.

Therefore the current main bottleneck is no longer the scalar historical-credit rule. It is the reliability of **causal lag confidence** under nonstationary closed-loop operation.

## Interpretation

A high lagged correlation is not sufficient evidence that a lag is causal in a closed loop. The controller action, plant state, moving optimum, and observation noise are mutually coupled. A robust delay estimate should therefore use more than one correlation peak.

The next experimental direction should add confidence checks such as:

- persistence of the same lag across independent windows;
- sign consistency of the local action-response slope;
- minimum action excitation before accepting a lag;
- rejection of lag changes during detected outliers or abrupt plant transitions;
- hysteresis before replacing an already accepted delay;
- optional multi-window agreement (short and long horizons).

## Scientific status

These are preliminary engineering findings, not a claim of general superiority. The composite benchmark must be reproduced across seeds and configurations, and raw per-run results should be preserved before quantitative claims are promoted to project-level conclusions.
