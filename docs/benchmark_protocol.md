# TRIVAX Benchmark Protocol

## Objective

Evaluate TRIVAX controllers under reproducible closed-loop conditions and compare them with appropriate conventional baselines without changing disturbance profiles, evaluation budgets or action bounds between methods.

## Required metadata

Every experiment should record:

- repository revision;
- Python/compiler version;
- operating system and hardware;
- random seed when randomness is used;
- controller configuration;
- plant/environment configuration;
- number of control cycles;
- action bounds;
- disturbance schedule;
- runtime;
- memory use where measurable.

## Initial scalar tracking benchmark

The reference plant exposes a scalar action `a_t` and an objective whose optimum changes over time. Each controller receives only observations permitted by its declared interface.

Primary metrics:

1. mean absolute tracking error;
2. median absolute tracking error;
3. cumulative tracking error;
4. cumulative regret relative to the instantaneous optimum;
5. convergence/recovery time after disturbances;
6. control effort, measured from action changes;
7. number of objective evaluations;
8. wall-clock runtime.

## Baselines

Candidate baselines include:

- fixed-step perturb-and-observe;
- hill climbing;
- PID-like control when a meaningful error signal is available;
- random/local search as a lower-bound reference.

A baseline must only be used when its information assumptions are compatible with the TRIVAX configuration being compared.

## Repetition

Deterministic tests should reproduce bit-for-bit when practical. Stochastic experiments should use multiple seeds and report distributions or uncertainty intervals, not only a best run.

## Ablations

When TRIVAX gains new mechanisms, benchmark at least:

- without memory;
- without coherence adaptation;
- fixed exploration step;
- without external inference;
- without external optimizer.

This is required before attributing performance improvement to a specific mechanism.

## Claim discipline

Benchmark results demonstrate behavior only for the evaluated task, configuration and budget. They do not establish general superiority over classical control, reinforcement learning, neural networks or optimization methods.
