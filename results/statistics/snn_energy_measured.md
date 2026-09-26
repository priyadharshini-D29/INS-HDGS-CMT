# Spiking front-end: measured cost on commodity hardware (NVIDIA A100-SXM4-80GB)

Trained LIF firing rate on 385 real epochs: 7.5% (sparsity 92.5%).

| encoder | batch | latency / inference (ms) | board power (W) | dynamic energy / inference (mJ) | total energy / inference (mJ) |
|---|---|---|---|---|---|
| spiking (LIF) | 1 | 3.985 | 85.0 | 0.0751 | 338.8980 |
| dense twin (Linear+GELU) | 1 | 0.407 | 88.3 | 1.3216 | 35.8878 |
| spiking (LIF) | 32 | 4.031 | 85.4 | 1.3204 | 344.0664 |
| dense twin (Linear+GELU) | 32 | 0.434 | 86.8 | 0.7660 | 37.6290 |

Card 1: idle board power 85.0 W at 1% utilisation before the loops; 0 other compute processes on the card. Dynamic energy = (mean power - idle power) x latency. Clean measurement.

Analytic neuromorphic estimate (event-driven AC vs. dense MAC, 45 nm): 1.5% of the dense energy for the LIF layers (327,680 synaptic operations/inference).

Interpretation: on a GPU/CPU every LIF time-step still executes dense matrix products, so the spiking encoder is not cheaper than its dense twin on this hardware (the table above measures this directly). The sparsity-based figure is a *projection* for event-driven neuromorphic processors and is reported only as such.
