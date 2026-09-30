# Grounded soft rules — abl_ns_rule_only (347 epochs, 37 checkpoint(s), ig)

Premise terms are (electrode, band) node features ranked by |integrated-gradient attribution| of the rule activation a_r; ↑ means higher band power at that electrode increases the rule's activation. The conclusion is the sign of the rule head l_r on the epochs where the rule dominates.

| rule | conclusion | dominant in | EEG share | grounded premise (top-k electrode–band) | ET term |
|---|---|---|---|---|---|
| 1 | HIGH | 1% | 50% | O1-gamma↑ ∧ C3-gamma↑ ∧ Fp2-gamma↑ | gaze_x↓ |
| 2 | LOW | 29% | 75% | O1-alpha↑ ∧ C3-gamma↓ ∧ O2-alpha↑ | gaze_x↓ |
| 3 | LOW | 13% | 85% | O1-alpha↑ ∧ C3-gamma↓ ∧ Pz-alpha↑ | gaze_x↓ |
| 4 | LOW | 14% | 66% | O1-gamma↓ ∧ T4-theta↑ ∧ O1-alpha↓ | gaze_y↑ |
| 5 | LOW | 8% | 47% | T6-alpha↑ ∧ Pz-alpha↑ ∧ P4-alpha↑ | gaze_x↓ |
| 6 | HIGH | 2% | 66% | O1-alpha↓ ∧ C3-gamma↑ ∧ O2-alpha↓ | gaze_x↑ |
| 7 | HIGH | 15% | 78% | O1-alpha↓ ∧ O2-alpha↓ ∧ Pz-alpha↓ | gaze_x↑ |
| 8 | LOW | 18% | 55% | P4-alpha↓ ∧ T6-alpha↓ ∧ Pz-alpha↓ | gaze_x↑ |
