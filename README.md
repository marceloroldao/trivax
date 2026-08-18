# TRIVAX

**TRIVAX — Resolutive Adaptive Edge Control Runtime**

TRIVAX is an experimental research and engineering framework for compact adaptive inference, memory-aware decision making, black-box optimization, temporal credit assignment, and closed-loop control on constrained hardware.

> **Archived release:** v0.1.1 — first public experimental research baseline archived on Zenodo.
>
> **DOI:** `10.5281/zenodo.21989027`
>
> **Scientific status:** experimental. Results must be reproduced from the code and benchmark protocol before being treated as scientific or engineering evidence. TRIVAX does not claim general superiority over established control methods.

## v0.1 research focus

The strongest current experimental result is not a universal-controller claim. The v0.1 research line investigates control under delayed and time-varying feedback, including explicit historical temporal credit:

\[
\Delta y_t \leftrightarrow \Delta a_{t-\hat d}.
\]

The repository preserves multiple experimental runtime generations so that architectural changes, negative results, and ablations remain auditable.

Current experiments include robust observation routing, online delay estimation, historical temporal credit, causal-delay confidence, optional active identification/value-of-information probing, Core-RC simplification, and online regime selection between conventional adaptive control and temporal-credit control.

## Scope

TRIVAX investigates:

- sensor-to-state encoding;
- compact sequential inference;
- delayed-feedback estimation;
- historical temporal credit assignment;
- adaptive exploration/exploitation;
- black-box objective optimization;
- multi-timescale control loops;
- reproducible closed-loop simulation;
- future ESP32/STM32/RISC-V deployment.

## Reproducibility

Install the package and run tests:

```bash
python -m pip install -e .
python -m pytest
```

Representative scientific benchmarks live under `benchmarks/`. GitHub Actions executes the benchmark suite and stores result artifacts. Comparisons include Perturb & Observe, adaptive hill climbing, TRIVAX runtime generations, ablations, out-of-distribution holdouts, temporal-advantage mapping, and online regime selection.

Benchmark results are evidence only for the evaluated protocol. They must not be generalized to unrelated plants, noise models, control constraints, or safety-critical systems without independent validation.

## Scientific and engineering principles

1. Compare competing controllers under the same disturbance profile and evaluation budget.
2. Record seeds, configuration, runtime, memory use, and metrics.
3. Separate empirical results from theoretical interpretation.
4. Use ablation tests before attributing gains to a mechanism.
5. Preserve scientifically relevant negative results.
6. Do not claim general superiority outside evaluated tasks.
7. Prefer inspectable state and deterministic execution where possible.
8. Treat embedded deployment constraints as first-class metrics.

## Relationship with Resolutive Science

`resolutive-science` is the normative source of truth for shared Resolutive Science terminology, notation and scientific-status conventions.

- Resolutive Science repository baseline: `v0.1.1`
- RSMS compatibility: `1.0-rc.1` — candidate compatibility, subject to re-audit when RSMS 1.0 becomes stable
- Project governance baseline: `RSPS 1.0-draft`

TRIVAX follows the terminology and project-governance conventions of the Resolutive research ecosystem where applicable. Computational concepts inspired by Resolutive research are treated as engineering hypotheses; physical or philosophical interpretations are not evidence of controller performance.

Companion research repositories may include `resolutive-inference`, `memoria.ia`, `resolutive-DB`, and `resolutive-computing`. TRIVAX remains independently testable.

## Licensing

TRIVAX is **source-available for research and education**, not OSI open source.

The repository `LICENSE` permits non-commercial academic, educational, research, evaluation, and reproducibility use subject to its conditions. **Commercial use requires a separate paid/commercial license and prior written authorization.** See `COMMERCIAL-LICENSE.md`.

Public visibility of this repository does not grant commercial-use rights.

## Safety

TRIVAX v0.1 is research software. It is not validated as the sole or primary controller for safety-critical systems. Physical deployment requires independent engineering validation, appropriate fail-safes, and compliance with applicable standards and regulations.

## Citation

Archived TRIVAX v0.1.1 release:

**Marcelo Roldão Matos. TRIVAX: Resolutive Adaptive Edge Control Runtime, v0.1.1. Zenodo. 2026. DOI: 10.5281/zenodo.21989027.**

Persistent identifier: `https://doi.org/10.5281/zenodo.21989027`

`CITATION.cff` contains machine-readable author and software citation metadata.

## Version

**v0.1.1 — Experimental research release / archived public baseline**

The v0.1.x line establishes the reproducible baseline. Subsequent versions must be compared against v0.1 rather than silently replacing its architecture or results. Active development continues separately in `v0.2-dev`.

## Author

Marcelo Roldão Matos  
ORCID: 0009-0003-6075-4680  
ETBRA Tecnologias — 2026
