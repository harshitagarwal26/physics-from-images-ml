# Lagrangian caVAE — Full Context Handoff

## Project Overview
Online perturbation identification in pendulum dynamics from image observations using a Lagrangian constrained autoencoder VAE.

**Core novelty (paper contribution):** Hard-frozen encoder/decoder approach where only a small `delta_V_net` is trainable, learning the perturbation potential: `V_total = V_frozen + delta_V_net`. Everything else (encoder, decoder, V_net, M_net, g_net) is frozen with `requires_grad=False`.

## Architecture
- **Lagrangian caVAE**: encoder (image → latent cos/sin), Lagrangian neural ODE (V_net, M_net, g_net), decoder (latent → image)
- **Delta-V frozen training**: `V_total(cos_q, sin_q) = V_net_frozen(cos_q, sin_q) + delta_V_net(cos_q, sin_q)`
- `delta_V_net`: MLP(2→50→50→1) with tanh activation, input = (cos θ, sin θ)
- Loss: pixel reconstruction loss, NOT latent trajectory loss
- Annealing is irrelevant with frozen encoder (kl_q and norm_penalty are constants w.r.t. delta_V_net)

## Key Files
- **Trainer**: `examples/pend_delta_v_frozen_trainer.py` — DeltaVFrozenModel class
- **ODE with delta_V**: `lag_caVAE/lag_delta_v.py` — Lag_Net_DeltaV, line 53: `self.delta_V_q = self.delta_V_net(cos_q_sin_q)`
- **Base ODE**: `lag_caVAE/lag.py` — Lag_Net, V_net takes (cos_q, sin_q)
- **Analysis notebook**: `results/pend/8_deltaV_retrain.ipynb` — main analysis with perturbation_config, PySR, SINDy, polynomial regression
- **Base model trainer**: `examples/pend_lag_cavae_trainer.py` — Model class (unperturbed)

## Checkpoints
- **Unperturbed base model**: `results/pend/pend-lag-cavae-T_p=4-epoch=983-step=7871.ckpt` (Model class)
- **Polynomial perturbation**: `results/pend/poly_original.ckpt` (DeltaVFrozenModel, epoch 846, dataset: `pendulum-gym-image-dataset-train-reverse-angle-perturbed.pkl`)
- **Sin perturbation checkpoints**: `2sin6_372.ckpt`, `5sin6_372_1000_epochs.ckpt`, `10sin6_372.ckpt`, `10sin4_372.ckpt`, `10sin2_372.ckpt` (DeltaVFrozenModel)

## Training Command
```bash
python examples/pend_delta_v_frozen_trainer.py \
    --pretrained_ckpt results/pend/pend-lag-cavae-T_p=4-epoch=983-step=7871.ckpt \
    --data_path datasets/<perturbed-dataset>.pkl \
    --name pend-delta-v-frozen --T_pred 4 --max_epochs 1000
```

## Datasets
- **Unperturbed**: `pendulum-gym-image-dataset-test-reverse-angle-stable-down-2.pkl`
- **Polynomial perturbed**: `pendulum-gym-image-dataset-train-reverse-angle-perturbed.pkl` (eta1=-0.0481, eta2=0.0113, eta3=0.1125)
- **Sin perturbed**: `pendulum-gym-image-dataset-train-reverse-angle-perturbed-sin2.pkl` etc.
- User generates datasets on a separate server by modifying `myenv/pendulum.py`

## Perturbation Definitions
- **Polynomial**: Δa = -α₁θ - α₂θ² - α₃θ³ where αᵢ = 3·ηᵢ/(M·L²). ETA1=-0.0481, ETA2=0.0113, ETA3=0.1125 → ALPHA1=-0.1443, ALPHA2=0.0339, ALPHA3=0.3375
- **Trigonometric**: Δa = A·sin(B·θ), ΔV = M·(A/B)·cos(B·θ). Tested with non-integer B (2.372, 4.372, 6.372)

## Experiment Results

### Trigonometric Perturbation Study (A·sin(B·θ))

| B | A | Epochs | R²(Δa) | R²(ΔV) | PySR Δa | PySR ΔV |
|---|---|--------|--------|--------|---------|---------|
| 6.372 | 2 | 1000 | 0.13 | - | freq=6.085 | - |
| 6.372 | 5 | 4000 | 0.59 | - | 3.12·sin(6.13θ) | - |
| 6.372 | 10 | 1000 | 0.63 | 0.04 | 5.49·sin(6.157θ) | messy, cos(6.173θ) buried |
| 4.372 | 10 | 1000 | 0.83 | 0.50 | 7.69·sin(4.227θ) | sin(θ) + cos(4.117θ) |
| 2.372 | 10 | 1000 | 0.81 | 0.56 | 7.74·sin(2.202θ) | sin(1.192θ)·sin(θ)·(-4.70) |

### Polynomial Perturbation Results

| Metric | Value |
|--------|-------|
| R²(ΔV) | **0.84** |
| R²(Δa) | **0.82** |
| ΔV amplitude ratio | 103% |
| Δa amplitude ratio | 111% |

Polynomial fit for Δa (learned vs true):
- θ³ coeff: -0.317 vs -0.338 (94% recovery)
- θ² coeff: 0.295 vs -0.034 (wrong sign, 9x too large — signal too small to detect)
- θ⁰ (constant): -0.983 vs 0.000 (spurious offset)

## Key Scientific Findings

### 1. (cos θ, sin θ) representation limitation
Non-integer B perturbations sin(Bθ) are NOT 2π-periodic, but networks taking (cos θ, sin θ) can only produce 2π-periodic functions. This is a mathematical impossibility — no architecture change can fix it while keeping (cos,sin) input. Learned curves match near θ≈0 but deviate near θ≈±π.

### 2. Differentiation as high-pass filter — why R²(Δa) vs R²(ΔV) differs between trig and polynomial
The network always learns low-frequency content best (spectral bias of tanh MLPs).

**Trigonometric**: True ΔV ~ cos(Bθ) is high-frequency, small amplitude. Network learns it poorly → R²(ΔV) low. But differentiation amplifies the partial high-frequency signal while suppressing low-frequency errors (offsets, biases) → R²(Δa) much better.

**Polynomial**: True ΔV ~ θ², θ³, θ⁴ is smooth, low-frequency. Network learns it well → R²(ΔV) good. But differentiation amplifies high-frequency noise more than the low-frequency signal → R²(Δa) slightly worse.

### 3. Fourier decomposition of polynomials on [-π, π]
Counterintuitive result — even powers are low-frequency, odd powers are high-frequency:
- θ²: 3 Fourier terms for 99% energy (even, smooth, continuous at boundary)
- θ⁴: 5 terms (even, smooth, continuous)
- θ: 49 terms (odd, discontinuous jump at ±π boundary)
- θ³: 74 terms (odd, large discontinuous jump at ±π)

The dominant cubic term (α₃θ³) is the hardest to represent in Fourier sense, but it's recovered well (94%) because its amplitude (0.3375) is much larger than the quadratic term (0.0339).

### 4. Amplitude recovery
Consistently ~77% amplitude recovery for trig perturbations (not yet explained). Polynomial shows ~94% for dominant term.

### 5. PySR configuration
- `nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}}` prevents illogical nested trig like sin(sin(θ))
- Only applied when trig operators are in unary_operators list (fix applied in this session for polynomial case)

## Planned but NOT implemented
- **θ-input fix**: Change delta_V_net input from (cos θ, sin θ) to θ = atan2(sin_q, cos_q) to support non-periodic perturbations. Saved in `memory/project_theta_input_fix.md`. User said "remember this fix for later."
- Zero-mean regularization + small output init for delta_V_net (discussed, not implemented)

## Environment Notes
- Python: `.venv/bin/python3` (Python 3.9)
- PyTorch 2.8.0 — requires `weights_only=False` for torch.load
- NumPy 2.0+ — `np.Inf` removed, need `np.Inf = np.inf` patch
- sklearn not installed in .venv (use manual r2_score)
- Plots saved to `results/pend/sine_perturbations_final/`

## Notebook (8_deltaV_retrain.ipynb) Structure
- Cell 0: markdown intro
- Cell 3: CONFIGURATION — checkpoints, perturbation_config dict (switch between poly/trig), sampling params
- Cell 7: Helper functions — extract_physics(), extract_delta_v_physics(), calibrate_phase_offset(), phi_to_theta_phys(), run_pysr()
- Cell 11: Ground truth computation, R² scores
- Cell 12: Section 4 — 2x3 plot grid (Row 1: V, M, acc comparisons; Row 2: ΔV, ΔM, Δa perturbation signals)
- Cell 21: PySR on Δa
- Later cells: PySR on ΔV, SINDy, polynomial regression

## Discussion thread left off at
We were discussing why polynomial perturbation has R²(Δa) ≤ R²(ΔV) while trig has R²(Δa) >> R²(ΔV). The analysis was at the point of understanding Fourier decomposition of polynomial terms — even powers (θ², θ⁴) are low-frequency, odd powers (θ, θ³) are high-frequency due to boundary discontinuities. The dominant θ³ term is recovered well despite being "high-frequency" because its amplitude is large. The user may want to continue this analysis or move on to other experiments.
