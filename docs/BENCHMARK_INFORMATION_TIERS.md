# TRIVAX v0.2 — Benchmark Information Tiers

## Why this exists

A fair controller comparison requires more than equal simulation time. Competing methods must also receive comparable information. A PID that receives an explicit tracking error, an MPC that receives a plant model, and a black-box optimizer that receives only a scalar objective are not solving the same information problem.

TRIVAX v0.2 therefore separates results into information tiers.

## Tier A — Black-box objective only

Controllers receive only the scalar objective observation produced by the plant and their own internal state/action history.

Primary peer group:

- TRIVAX Regime Selector;
- TRIVAX Runtime V2 temporal-credit path;
- Adaptive Hill Climber;
- Perturb & Observe;
- Extremum Seeking.

This is the primary ranking tier for the current TRIVAX scientific claim.

## Tier B — Tracking-error controllers

Controllers may receive an explicit process measurement and setpoint/error signal.

Examples:

- PI/PID;
- gain-scheduled PID;
- Smith-predictor PID in later experiments.

Tier-B results are useful engineering references but are not ranked as direct black-box peers unless the experiment gives the same explicit tracking signal to every method.

## Tier C — Model-aware controllers

Controllers may receive an explicit plant model or model parameters unavailable to Tier-A methods.

Examples:

- MPC;
- LQR;
- model-based optimal control.

Model-aware results are reported as reference/upper-bound style comparisons unless all methods receive equivalent model information.

## Tuning protocol

Each algorithm family may tune hyperparameters on calibration scenarios only. Selected parameters are frozen before blind holdout evaluation. Holdout outcomes must not be used to retune the same reported trial.

## Metrics

Report at minimum:

- MAE;
- tail MAE;
- maximum absolute error;
- control effort;
- CPU/runtime proxy where available;
- worst-regime MAE;
- win rate over seeds/regimes;
- temporal duty and switch count for regime-selecting TRIVAX variants.

## Scientific reporting rule

Do not write "TRIVAX beats PID/MPC" from a cross-tier experiment. Correct phrasing distinguishes the information assumptions, for example:

> Under black-box objective-only feedback, TRIVAX was compared directly with P&O, adaptive hill climbing and extremum seeking. PID and MPC were evaluated separately as tracking/model-aware references.

Negative results and failure regions are retained.
