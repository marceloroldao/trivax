# TRIVAX Architecture

## 1. Purpose

TRIVAX is an experimental runtime for adaptive closed-loop decision and control on edge systems. Its main architectural role is to integrate perception, compact state representation, memory, inference, optimization and actuation while keeping each subsystem independently testable.

## 2. Functional pipeline

```text
observation X_t
      |
      v
Encoder E(X_t)
      |
      v
resolutive state R_t
      |
      +--> inference
      +--> memory
      +--> optimization
      |
      v
Decision Core
      |
      v
action A_t
      |
      v
plant / environment
      |
      v
observation X_(t+1)
```

The minimum loop is

\[
X_t \rightarrow \mathcal{R}_t \rightarrow A_t \rightarrow X_{t+1}.
\]

## 3. Resolutive state

The v0.1 reference implementation uses an intentionally small and inspectable state:

- current observation;
- observation delta;
- coherence estimate;
- current search direction;
- search step size.

Future encoders may expose application-specific forms such as

\[
\mathcal{R}_t=(\rho_t,\phi_t,\kappa_t,\chi_t,\tau_t,\ldots).
\]

No symbol is assumed to carry a universal physical interpretation. Every application must define its encoder and semantics explicitly.

## 4. Core modules

### Encoder

Maps raw observations into a compact state representation.

### Inference

Estimates latent regime, confidence or transition structure. Future adapters may connect `resolutive-inference`.

### Memory

Stores prior states, actions and outcomes. Future adapters may connect hierarchical resolutive memory and `resolutive-DB`.

### Optimization

Searches for improved actions under a bounded computational budget. Future adapters may connect `resolutive-computing`.

### Decision Core

Combines current state, historical evidence, confidence/coherence and candidate actions to select the next bounded action.

### Runtime

Schedules control cycles and, in future versions, multiple timescales.

### Embedded layer

Provides fixed-memory and hardware-specific execution for ESP32, STM32 and RISC-V targets.

## 5. Multi-timescale roadmap

TRIVAX is intended to support independent update rates:

```text
fast loop     -> actuator regulation
medium loop   -> system dynamics / local adaptation
slow loop     -> learning / memory consolidation / policy updates
```

Conceptually,

\[
\tau_{fast}<\tau_{medium}<\tau_{slow}.
\]

This feature is not yet implemented in v0.1 and must not be presented as benchmarked functionality.

## 6. Integration principle

TRIVAX should depend on companion resolutive projects through explicit adapters rather than copy their implementations. This keeps benchmarks attributable and allows each project to evolve independently.

## 7. Safety and bounded control

All actuator-facing components must support explicit action bounds. Real hardware examples must define safe operating ranges, watchdog behavior and fallback control before physical deployment.
