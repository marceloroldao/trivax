# TRIVAX

**TRIVAX — Resolutive Adaptive Edge Control Runtime**

TRIVAX is an experimental research and engineering framework for compact adaptive inference, memory-aware decision making, black-box optimization, and closed-loop control on constrained hardware.

The project investigates whether a small, inspectable resolutive state and explicitly controlled computational budget can support real-time autonomous control without requiring a neural network in the control loop.

> **Status:** early research scaffold. Results must be reproduced from code before being treated as scientific or engineering evidence.

## Core idea

TRIVAX closes the loop between observation, state estimation, memory, decision and action:

```text
physical system / environment
          |
          v
      observation
          |
          v
   TRIVAX Encoder
          |
          v
 resolutive state R_t
          |
   +------+------+----------------+
   |             |                |
   v             v                v
Inference      Memory         Optimization
   |             |                |
   +-------------+----------------+
                 |
                 v
          Decision Core
                 |
                 v
              action
                 |
                 v
            actuator
                 |
                 +--------> feedback
```

A generic control cycle is represented as

\[
X_t \rightarrow \mathcal{R}_t \rightarrow A_t \rightarrow X_{t+1}.
\]

The initial computational state may include terms such as

\[
\mathcal{R}_t=(\rho_t,\phi_t,\kappa_t,\chi_t,\tau_t,\ldots),
\]

where the exact meaning of each component is application-specific and must be documented by each encoder.

## Initial scope

TRIVAX v0.1 focuses on **adaptive edge control**:

- sensor-to-state encoding;
- compact sequential inference;
- memory-assisted decisions;
- adaptive exploration/exploitation;
- black-box objective optimization;
- multi-timescale control loops;
- closed-loop simulation;
- future ESP32/STM32/RISC-V deployment.

The first reference problem is a simulated controller that must search for and maintain the maximum value of a changing scalar objective under disturbances.

## Relationship with companion projects

TRIVAX is intended to integrate, rather than duplicate, the following research lines:

- `resolutive-inference` — compact sequential inference;
- `memoria.ia` — hierarchical resolutive memory;
- `resolutive-DB` — resolutive addressing and retrieval;
- `resolutive-computing` — adaptive search and black-box optimization.

These integrations are optional. The core package must remain independently testable.

## Repository layout

```text
trivax/
├── src/trivax/
│   ├── core.py
│   ├── encoder.py
│   ├── runtime.py
│   ├── control.py
│   └── simulation.py
├── benchmarks/
│   └── scalar_tracking.py
├── examples/
│   └── scalar_control.py
├── tests/
│   └── test_core.py
├── docs/
│   ├── architecture.md
│   └── benchmark_protocol.md
├── pyproject.toml
└── README.md
```

## Scientific and engineering principles

1. Compare competing controllers under the same disturbance profile and evaluation budget.
2. Record seeds, configuration, runtime, memory use and all metrics.
3. Separate empirical results from theoretical interpretation.
4. Use ablation tests before attributing gains to a specific mechanism.
5. Do not claim general superiority over neural, classical-control or optimization methods outside evaluated tasks.
6. Prefer inspectable state and deterministic execution where possible.
7. Treat embedded deployment constraints as first-class metrics.

## Reference benchmark

The first benchmark evaluates whether a controller can locate and track a moving optimum.

Metrics include:

- convergence time;
- mean absolute tracking error;
- cumulative regret;
- overshoot;
- control effort;
- number of objective evaluations;
- runtime;
- memory footprint.

Future comparisons should include suitable classical baselines such as perturb-and-observe, hill climbing and PID-like controllers when their assumptions match the test problem.

## Quick start

```bash
python -m pip install -e .
python -m pytest
python benchmarks/scalar_tracking.py
```

## Roadmap

### v0.1 — Control scaffold

- deterministic resolutive state;
- adaptive scalar controller;
- closed-loop simulator;
- reproducible benchmark protocol;
- tests and metrics.

### v0.2 — Memory and inference

- state history;
- nearest prior experience;
- coherence estimation;
- exploration adapted by confidence/coherence.

### v0.3 — External resolutive modules

- optional `resolutive-inference` adapter;
- optional `resolutive-DB` adapter;
- optional `resolutive-computing` optimizer adapter;
- optional hierarchical memory adapter.

### v0.4 — Embedded runtime

- fixed-memory execution mode;
- ESP32 reference implementation;
- integer/fixed-point experiments;
- timing and power benchmarks.

### v0.5 — Robotics and IoT

- sensor fusion interface;
- robotics control examples;
- network/IoT adaptive control examples;
- multi-loop scheduler.

## Project position

TRIVAX is not presented as a replacement for established control theory, reinforcement learning or neural architectures. It is a research framework for testing a narrower hypothesis: whether compact state, explicit memory and adaptive search can form a useful low-cost control runtime for edge systems.

## Author

Marcelo Roldão Matos

ETBRA Tecnologias — 2026
