# PROJECT_CONTEXT.md

## Research Domain

Multimodal Neuromarketing

## Dataset

NeuMa Dataset

## Core Scientific Problem

Predict consumer purchase intention using synchronized:

- EEG
- Eye Tracking
- Mouse Interaction
- Product Attention

## Long-Term Architecture Goal

Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking Cross-Modal Transformer

## Current Phase

PHASE 0 → Dataset Understanding

## Current Tasks

- inspect raw XDF streams
- validate modality existence
- understand synchronization
- inspect EEG structure
- inspect ET structure
- inspect Mouse streams
- inspect markers
- validate ROI coordinates
- validate subject structure

## Current Restrictions

DO NOT:
- preprocess EEG
- interpolate ET
- train models
- extract features
- create train/test splits
- optimize accuracy

## Expected Outputs

- dataset reports
- stream inspection summaries
- synchronization visualizations
- ROI visualizations
- modality statistics

## Dataset Understanding Goals

The system must understand:

1. how modalities are stored
2. how timestamps align
3. how labels are generated
4. how products are mapped
5. how gaze relates to products
6. how clicks relate to labels
7. how EEG segments may later be created