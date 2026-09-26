# Grounded soft rules — abl_full_S01 (15 epochs, 1 checkpoint(s), ig)

Premise terms are (electrode, band) node features ranked by |integrated-gradient attribution| of the rule activation a_r; ↑ means higher band power at that electrode increases the rule's activation. The conclusion is the sign of the rule head l_r on the epochs where the rule dominates.

| rule | conclusion | dominant in | EEG share | grounded premise (top-k electrode–band) | ET term |
|---|---|---|---|---|---|
| 1 | LOW | 0% | 26% | T3-beta↑ ∧ F7-beta↓ ∧ T6-beta↓ | pupil↑ |
| 2 | LOW | 0% | 73% | T3-beta↓ ∧ T4-beta↓ ∧ F4-theta↑ | gaze_x↑ |
| 3 | LOW | 0% | 73% | T4-gamma↓ ∧ T3-gamma↓ ∧ Cz-theta↑ | pupil↑ |
| 4 | HIGH | 0% | 32% | Cz-theta↓ ∧ T3-beta↑ ∧ T4-beta↑ | gaze_x↑ |
| 5 | LOW | 0% | 51% | T4-beta↑ ∧ T3-beta↑ ∧ P3-alpha↑ | gaze_x↓ |
| 6 | HIGH | 0% | 44% | T4-beta↓ ∧ P3-alpha↓ ∧ T3-beta↓ | gaze_x↑ |
| 7 | HIGH | 0% | 50% | T4-beta↑ ∧ T3-beta↑ ∧ T4-gamma↑ | gaze_x↓ |
| 8 | LOW | 100% | 43% | F8-beta↑ ∧ Fp1-beta↑ ∧ O2-beta↑ | pupil↓ |
