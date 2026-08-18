# TRIVAX v0.1.0 — Experimental Research Release

TRIVAX v0.1.0 is the first public experimental baseline of the TRIVAX adaptive edge-control research project.

## Scope

This release investigates adaptive control under delayed and time-varying feedback, with emphasis on historical temporal credit assignment, online delay estimation, robust observation handling, causal validation, optional value-of-information probing, and regime-dependent controller selection.

## Included research lines

- robust observation routing;
- online delay estimation;
- historical temporal credit assignment;
- causal-delay confidence;
- optional active identification / VOI probing;
- Runtime V2–V6 experimental generations;
- Core-RC simplification;
- online regime selector;
- Perturb & Observe and Adaptive Hill Climber baselines;
- ablation studies;
- out-of-distribution holdouts;
- temporal-advantage regime mapping;
- reproducible tests and benchmark workflows.

## Scientific interpretation

v0.1.0 does **not** claim universal superiority over conventional controllers. Experiments indicate that temporal-credit mechanisms can be advantageous in specific delayed-feedback and sufficiently dynamic regimes, while simpler adaptive controllers can be superior in other regimes.

Negative and neutral results are intentionally preserved as part of the research record.

## Reproducibility

Install and test with:

```bash
python -m pip install -e .
python -m pytest
```

Benchmark scripts are under `benchmarks/`. Results are protocol-specific and must not be generalized to unrelated physical plants or safety-critical deployments without independent validation.

## Safety

This is experimental research software. It is not validated as a sole or primary controller for safety-critical systems. Physical deployment requires independent engineering validation, fail-safe design, and applicable regulatory compliance.

## License

TRIVAX is source-available for non-commercial academic, educational, research, evaluation, and reproducibility use under the repository `LICENSE`. Commercial exploitation requires a separate commercial license and prior written authorization. See `COMMERCIAL-LICENSE.md`.

This license is not represented as OSI-approved open source.

## Citation and archival

Use `CITATION.cff` for citation metadata. A Zenodo DOI should be added to the repository after archival of this release.

## Version policy

`v0.1.0` is the historical experimental baseline. Subsequent development must occur in later development versions and must not silently rewrite the v0.1.0 benchmark record.
