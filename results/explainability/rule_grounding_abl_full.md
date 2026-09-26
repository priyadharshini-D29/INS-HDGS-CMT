# Grounded soft rules — abl_full (347 epochs, 37 checkpoint(s), ig)

Premise terms are (electrode, band) node features ranked by |integrated-gradient attribution| of the rule activation a_r; ↑ means higher band power at that electrode increases the rule's activation. The conclusion is the sign of the rule head l_r on the epochs where the rule dominates.

| rule | conclusion | dominant in | EEG share | grounded premise (top-k electrode–band) | ET term |
|---|---|---|---|---|---|
| 1 | LOW | 0% | 92% | O2-alpha↑ ∧ T6-alpha↑ ∧ Fz-alpha↑ | pupil↑ |
| 2 | LOW | 57% | 82% | O2-alpha↓ ∧ Fz-alpha↓ ∧ P4-alpha↓ | gaze_x↑ |
| 3 | LOW | 0% | 80% | P4-alpha↑ ∧ C4-alpha↑ ∧ T6-alpha↑ | gaze_x↓ |
| 4 | LOW | 0% | 73% | O2-alpha↑ ∧ Fz-theta↓ ∧ C3-gamma↑ | gaze_x↑ |
| 5 | LOW | 0% | 72% | C4-alpha↑ ∧ T4-beta↑ ∧ Fz-alpha↑ | gaze_x↑ |
| 6 | HIGH | 0% | 60% | C4-alpha↓ ∧ C3-gamma↓ ∧ O1-gamma↑ | gaze_x↓ |
| 7 | LOW | 5% | 88% | Fz-alpha↓ ∧ F3-alpha↓ ∧ T6-alpha↓ | gaze_x↓ |
| 8 | LOW | 37% | 78% | P4-alpha↓ ∧ C4-alpha↓ ∧ T6-alpha↓ | gaze_x↓ |
