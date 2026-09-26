# SNN spike-sparsity & energy proxy

Trained LIF encoder (`repro_focal_g3p0_effective_num_37_fold01_e0.pt`), 300 EEG epochs, 10 simulation steps.

- **Mean firing rate:** 10.59%  (sparsity 89.41% — neurons silent most steps).
- **LIF synaptic MACs / inference:** 327,680 (event-driven in the SNN).
- **Energy (LIF layers), 45 nm proxy:** SNN 31,241 pJ vs ANN 1,507,328 pJ → **2.1%** of ANN (48.2× lower).
- **Energy (whole encoder, dense input-projection counted as MAC in both):** SNN 172,553 pJ vs ANN 1,648,640 pJ → 10.5% of ANN.

Energies: E_MAC=4.6 pJ, E_AC=0.9 pJ (45 nm CMOS, Horowitz ISSCC 2014). The spiking layers compute only when a presynaptic neuron fires, so their cost scales with the firing rate; the ratio is rho·E_AC/E_MAC. This quantifies the efficiency rationale for the LIF encoder independently of its (neutral) effect on accuracy.
