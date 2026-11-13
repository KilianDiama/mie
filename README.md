# OMEGA PRIME CORE 7.0

Open, model-based engine for design, calibration, and uncertainty quantification of DNA-functionalized plasmonic LSPR nanosensors.

Single-file Python monolith. Dependencies: `numpy` and `scipy` only.

## What it does

- Gold permittivity: Rakic Lorentz–Drude model with additive Kreibig size correction.
- Exact Mie theory: bare gold sphere and core–shell sphere.
- LSPR peak refinement, FWHM, bulk sensitivity, and FOM.
- Binding-induced shift: core–shell Mie + Maxwell–Garnett effective medium.
- DNA thermodynamics: SantaLucia 1998 nearest-neighbor model with configurable mismatch penalty.
- Monte-Carlo propagation of DNA ΔG uncertainty.
- Analytical thermal estimates: single particle, Kapitza interface, collective heating, transient heuristic.
- Bayesian Langmuir calibration with analytically marginalized `S_max`, posterior predictive checks, identifiability diagnostics, and probabilistic LOD.
- Optical Monte-Carlo uncertainty propagation and one-at-a-time sensitivity analysis.
- Pareto design screening, self-tests, JSON/CSV reports, SHA-256 provenance.
- Every result carries an `epistemic` status block.

## Quick start

```bash
python omega_prime_core.py selftest
python omega_prime_core.py demo
python omega_prime_core.py spectrum --radius-nm 40
python omega_prime_core.py shift --radius-nm 40 --theta 1
python omega_prime_core.py binding --probe ACGTTAGC --target GCTAACGT --ddg-sigma 0.8
python omega_prime_core.py thermal --radius-nm 40 --intensity 5e5
python omega_prime_core.py mc-optical --radius-nm 40 --radius-sigma 2 --medium-sigma 0.005
python omega_prime_core.py sensitivity --radius-nm 40
python omega_prime_core.py calibrate --data calib.csv --sigma-nm 0.05
python omega_prime_core.py predict --data calib.csv --conc-nm 50 --sigma-nm 0.05
python omega_prime_core.py design --target-nm 650
