# AGENTS.md

PHASE 0 → Dataset Understanding

IMPORTANT:
Do NOT implement:
- preprocessing
- feature extraction
- machine learning
- deep learning
- transformers
- classification
- train/test splits

until Phase 0 validation is complete.

## Dataset

NeuMa Dataset
Nature Scientific Data Paper:
https://www.nature.com/articles/s41597-023-02392-9

## Modalities

1. EEG
- DSI-24
- 300 Hz
- 21 EEG channels

2. Eye Tracking
- Tobii Fusion
- 120 Hz
- gaze coordinates

3. Mouse Streams
- cursor positions
- mouse clicks

4. Marker Streams
- product/page events

5. ROI Bounding Boxes
- product coordinates

6. Questionnaire Data
- demographics
- personality
- buying behavior

## Coding Requirements

- modular code
- reusable functions
- no hardcoded paths
- proper validation
- strong logging
- scientific correctness
- reproducibility

## Important Validation Priorities

1. stream integrity
2. synchronization
3. timestamp correctness
4. modality consistency
5. label validity

## Expected Folder Structure

Follow the defined Phase 0 architecture strictly.

## Scientific Constraints

All implementations should prioritize:

- neuroscientific validity
- multimodal consistency
- interpretability
- reproducibility
- LOSOCV compatibility

Never prioritize accuracy over scientific correctness.