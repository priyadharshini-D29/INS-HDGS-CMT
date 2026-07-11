# Reference figures — REAL DATA ONLY

Every figure here traces back to the trained model running on real preprocessed
data. Synthetic / `np.random` placeholder figures were deliberately excluded.

## Verified-real sources
- `analysis/figure_panels_S24.py` — loads `NeumaGraphDataset` (subject S24),
  the real 5-member trained ensemble (`repro_focal_g3p0_effective_num_37`,
  fold-22), runs eval-mode inference. Panels A1–D2. Docstring: "using REAL
  model intermediates — not synthetic placeholders."
- `analysis/case_study_high_low.py` — same real dataset + ensemble
  (`fig_regions`, `fig_gaze_pred`).
- `explainability/saliency.py` + `visualization/plot_attention.py` — real
  explainability outputs (`band_saliency`, `gradcam_graph`, `roi_gate`).

## Topic → figure map (topics that HAVE a real figure)
| Topic | File(s) |
|-------|---------|
| EEG Band-Power Features | EEG_1 · raw_eeg_epoch, band_saliency |
| Functional Connectivity Graph | EEG_2 · connectivity_matrix |
| DynamicGAT | EEG_3 · dynamic_brain_graph, gradcam_graph |
| LIF Spiking Encoder | EEG_4 · spike_raster_original, spike_raster_dark |
| Gaze and Pupil Sequence | ET_1 · gaze_trajectory, fixation_map |
| ROI Attention Distribution | ET_3 · roi_attention, roi_gate |
| EEG→Graph Cross-Attention | FUSION_5 · xattn_eeg_graph |
| EEG→ET Cross-Attention | FUSION_6 · xattn_eeg_et |
| Gated Fusion / Joint Representation | FUSION_7_8 · fused_representation |
| Differentiable Rules (×8) | NS_10 · rule_activations |
| Consumer Engagement Prediction | OUT_12 · prediction |

## Topics with NO real figure (nothing added — do NOT substitute synthetic)
- EEG Representation
- Eye-Tracking Representation
- ET "Transformer Attention Encoder" (real cross-attn C2 is EEG←ET, not the ET self-attention encoder)
- EEG Embedding
- Self-Attention over the three modality tokens
- Fuzzy Membership
- Constraint Aggregation

## Excluded as synthetic (`np.random`) — DO NOT use in the manuscript
- `cognitive_trace/S24/**`  → `analysis/full_subject_cognitive_trace.py` (100% random)
- `case_study/S24/**`       → `analysis/run_single_subject_case_study.py` (random + schematic)
- `fig_roi_timecourse.*`    → no generator script found anywhere in the repo (unverifiable)
