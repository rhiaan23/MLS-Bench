# Climate Physics Emulation: Neural Network Architecture

## Research Question
Design an improved neural network architecture for emulating sub-grid atmospheric physics processes in climate models. Your architecture should achieve lower Normalized MSE (NMSE) than the default MLP baseline on the ClimSim low-resolution dataset.

## Background
Global climate models divide the atmosphere into grid cells, but many critical physical processes (radiation, convection, cloud formation) occur at scales smaller than these grid cells. Traditionally, these sub-grid processes are approximated by parameterization schemes — handcrafted physics-based approximations. Neural network emulators can learn these mappings from high-resolution simulation data, potentially improving both accuracy and computational efficiency.

ClimSim (Yu et al., "ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation", NeurIPS 2023 Datasets & Benchmarks; arXiv:2306.08754) provides data from the E3SM-MMF multi-scale climate model, where each sample maps an atmospheric column state to the corresponding sub-grid physics tendencies computed by the high-resolution physics module.

## Task
Modify the `Custom` model class in `custom_emulator.py` to implement a better neural network architecture. The model must:

- Accept `input_dim` and `output_dim` in `__init__`.
- Implement `forward(x)` where `x` has shape `(batch_size, input_dim)`.
- Return predictions of shape `(batch_size, output_dim)`.

## Interface

**Input structure** (556-dim vector per atmospheric column):
- 9 multi-level variables × 60 vertical levels = 540 features:
  temperature (`state_t`), specific humidity (`state_q0001`), cloud ice (`state_q0002`),
  cloud liquid (`state_q0003`), zonal wind (`state_u`), meridional wind (`state_v`),
  ozone (`pbuf_ozone`), methane (`pbuf_CH4`), nitrous oxide (`pbuf_N2O`).
- 16–17 single-level (surface/TOA) scalar variables:
  surface pressure, solar insolation, heat fluxes, wind stress, albedos,
  surface type fractions, snow depths.

**Output structure** (368-dim vector):
- 6 multi-level tendency variables × 60 levels = 360 features:
  temperature tendency (`ptend_t`), humidity tendencies (`ptend_q0001`–`q0003`),
  wind tendencies (`ptend_u`, `ptend_v`).
- 8 single-level diagnostic outputs:
  net shortwave, longwave down, snow/rain precipitation, direct/diffuse solar.

## Fixed Pipeline
Dataset loading, input/output normalization, train/val/test splits, optimizer choice and schedule, loss function, and the multi-budget evaluation harness are all fixed by the scaffold. Only the `Custom` architecture is editable.

## Evaluation
- **Primary metric**: Normalized MSE (NMSE = MSE / Var(target), lower is better).
- **Secondary metrics**: R² (higher is better), RMSE, plus separate `ml_nmse` (multi-level) and `sl_nmse` (single-level) breakdowns.
- **Training budgets**: 30 epochs (short), 100 epochs (medium), 200 epochs (long).
- All three training budgets are run; improvements should be consistent across all three.
