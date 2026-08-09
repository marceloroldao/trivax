# Historical temporal credit findings

The first TRIVAX runtime used a conservative hold policy after estimating sensor delay: controller updates were applied once every `delay + 1` samples. This improved moderate-delay cases, but became inefficient at longer delays because the actuator was held while the target could continue moving.

A second approach was therefore tested: explicit historical temporal credit assignment.

For an estimated delay `d`, the current change in objective is attributed to the historical action change that most likely caused it:

\[
\Delta y_t \leftrightarrow \Delta a_{t-d}
\]

and the local response slope is estimated as

\[
s_t = \frac{y_t-y_{t-1}}{a_{t-d}-a_{t-d-1}}.
\]

The sign of `s_t` determines the current search direction while new actuator commands continue to be issued every cycle.

## Preliminary local findings

In the delayed sinusoidal benchmark with `step_size=0.01`, light Gaussian observation noise (`sigma=0.005`), 1200 steps and 20 seeds, the historical-credit prototype remained stable across delays from 0 through 10 samples. Mean tail absolute error was approximately in the 0.027-0.032 range across the tested delays.

These numbers are preliminary local measurements and must be reproduced from the repository benchmark before being used as evidence.

## Interpretation

The result suggests that long-delay performance is better addressed as a temporal attribution problem than as a pure waiting problem.

Correct delay identification and correct delay compensation are separate tasks:

1. `DelayEstimator` estimates when an action affects observations.
2. `HistoricalCreditController` attributes feedback to the corresponding historical action.
3. The actuator can continue to receive commands every cycle instead of being frozen for `d+1` samples.

## Current limitations

The current historical-credit controller is deliberately minimal:

- scalar action only;
- scalar objective only;
- fixed step size;
- no uncertainty weighting on the delay estimate;
- no filtering of nearly-zero historical action differences beyond a numerical deadband;
- no explicit model for time-varying delay during a single attribution window.

The module remains experimental until compared against the hold runtime, TRIVAX v0.3 and conventional baselines across noise, moving targets and varying delay.
