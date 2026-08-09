# Adversarial Benchmark Findings

This document records preliminary findings from the TRIVAX adversarial-control benchmark suite. These results are exploratory and must be reproduced from repository code before being treated as evidence.

## Scenarios

The suite stresses controllers with:

- slow optimum drift;
- repeated abrupt optimum changes;
- impulsive observation outliers;
- fixed observation delay;
- simple directional hysteresis;
- abrupt curvature change.

Controllers compared:

- TRIVAX v0.1;
- Coherence Adaptive;
- TRIVAX v0.2 Regime;
- TRIVAX v0.3 Probabilistic;
- Perturb & Observe.

## Preliminary qualitative result

The v0.3 probabilistic controller is promising in slow drift, repeated steps, hysteresis and curvature-shift scenarios. It does not dominate all adverse conditions.

Two weaknesses are especially important:

1. impulsive outliers can trigger false reversals and corrupt the short-horizon regime statistics;
2. observation delay breaks the assumption that the most recent observation delta describes the most recent action, causing direction reversals to be assigned to the wrong control move.

The delay result is structurally important. It should not be addressed by tuning regime thresholds alone. A future controller should explicitly model action-observation correspondence or delay.

## Engineering implication

The next experimental branch should focus on a robust observation layer rather than immediately adding more control regimes. Candidate mechanisms include:

- robust delta estimation;
- median or clipped innovation filtering;
- explicit action history;
- delay estimation;
- delayed-credit assignment.

Any new mechanism must be evaluated by ablation against v0.3 and the simple P&O baseline.
