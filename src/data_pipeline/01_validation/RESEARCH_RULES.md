# RESEARCH_RULES.md

# Core Philosophy

Understand → Validate → Verify → Quantify → Model

Never skip validation.

# Scientific Rules

1. Synchronization correctness is more important than model complexity.

2. Labels must always be validated before modeling.

3. Multimodal consistency is mandatory.

4. EEG should never be treated as generic time-series only.

5. Eye tracking should remain spatially grounded to ROI.

6. Mouse streams represent behavioral intention.

7. Every modality must be interpretable.

# Coding Rules

- small reusable modules
- strong logging
- proper docstrings
- no notebook-only logic
- no duplicated code
- no hidden assumptions

# Validation Rules

Every phase must include:

- automated validation
- manual inspection
- visualization
- statistical summary

# Reproducibility

- fixed seeds
- deterministic processing
- config-based paths
- report generation

# Future Compatibility

Code should remain compatible with:

- LOSOCV
- multimodal fusion
- graph learning
- transformers
- explainability
- neuro-symbolic systems