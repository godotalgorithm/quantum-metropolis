# quantum Metropolis algorithm

This repository contains the software and numerical data described in the paper
"Measurement-Based Quantum Metropolis Algorithm" [https://arxiv.org/abs/1903.01451].

1. `verification.c` is a statistical verification of the algorithms in Figs. 1 and 2 of the paper
   on small, random Hamiltonians with random transition probabilities.

2. `stop-statistics.c` is a simulation of Fig. 2 applied to the single-state Hamiltonian to gather
   statistics on stopping times, which are aggregated in `simulation.txt` and summarized in the body
   of the paper.

3. `error-bound.c` is a program to verify the empirical error bound in Eq. (B6) of the paper on an
   adjustable, uniform grid of parameter values. The set of parameters over which the bound was verified
   is defined by the job script `error-bound.sh` and recorded in `error-bound.txt`.

4. `simulation.c` is a program to simulate the quantum Metropolis algorithm applied to 1D transverse-field
   Ising models with periodic boundary conditions. The Markov chain of observation data visualized in
   Fig. 4 is the result of running the job script `simulation.sh`, which is recorded in `simulation.txt`.

5. Fig. 3 of the paper was generated with Inkscape and contains embedded pdf images generated by LaTexIt,
   which are recorded here to aid in future visualizations of Gaussian-filtered quantum phase estimation
   circuits or modifications thereof.
