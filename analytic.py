"""Closed-form reference trajectories and metrics. Not imported by the prediction module."""
import json
from pathlib import Path

import torch

GEOMETRIES = ("pipe", "channel")
J0_ROOTS = (2.404825557695773, 5.520078110286311, 8.653727912911013, 11.791534439014281,
            14.930917708487787, 18.071063967910924, 21.21163662987926, 24.352471530749302)
NODES = 128


def original_test_cases(geometry):
    path = Path(__file__).parent / "data" / "original_high_modes.json"
    cases = json.loads(path.read_text(encoding="utf-8"))[geometry]
    return cases


def sample_cases(geometry, count, seed, split):
    if geometry not in GEOMETRIES or split not in ("train", "validation", "test"):
        raise ValueError("Unknown geometry or split.")
    rng = torch.Generator().manual_seed(seed)
    draws = torch.rand((count, 6), dtype=torch.float64, generator=rng)
    limits = torch.tensor([.5, .25, .15, .08], dtype=torch.float64)
    amplitudes = (2 * draws[:, 2:] - 1) * limits
    if split == "train":
        amplitudes[::8] = 0
    return [
        {"id": f"{geometry}_{split}_{i}", "Ub": .005 * 6 ** row[0].item(),
         "t0": .03 * row[1].item(), "dt": (.005, .01, .02)[i % 3],
         "coefficients": amplitudes[i].tolist()}
        for i, row in enumerate(draws)
    ]


def reference(geometry, cases, frames=30):
    """Return exact total velocity [batch, frames, nodes] and its steady term [batch, nodes]."""
    if geometry not in GEOMETRIES or not cases or frames < 1:
        raise ValueError("Invalid reference request.")
    pipe = geometry == "pipe"
    x = torch.linspace(0 if pipe else -1, 1, NODES, dtype=torch.float64)
    speed = torch.tensor([c["Ub"] for c in cases], dtype=torch.float64)
    steady = speed[:, None] * ((2.0 if pipe else 1.5) * (1 - x.square()))
    amplitudes = torch.tensor([c["coefficients"] for c in cases], dtype=torch.float64)
    modes = amplitudes.size(1)
    if not 1 <= modes <= len(J0_ROOTS):
        raise ValueError("Expected one to eight analytic modes.")
    if pipe:
        roots = torch.tensor(J0_ROOTS[:modes], dtype=torch.float64)
        basis = torch.special.bessel_j0(roots[:, None] * x)
    else:
        roots = torch.arange(1, modes+1, dtype=torch.float64) * (torch.pi / 2)
        basis = torch.sin(roots[:, None] * (1 + x))
    initial_t = torch.tensor([c["t0"] for c in cases], dtype=torch.float64)
    dt = torch.tensor([c["dt"] for c in cases], dtype=torch.float64)
    time = initial_t[:, None] + dt[:, None] * torch.arange(frames, dtype=torch.float64)
    decay = torch.exp(-time[:, :, None] * roots.square())
    perturbation = (decay * amplitudes[:, None, :]) @ basis
    total = steady[:, None] + speed[:, None, None] * perturbation
    total[:, :, -1] = 0
    if not pipe:
        total[:, :, 0] = 0
    return total, steady


def trajectory_errors(geometry, prediction, truth, steady):
    """Percent, integrated over space and forecast time, one value per trajectory."""
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and target shapes must match.")
    weights = torch.ones(truth.size(-1), dtype=torch.float64, device=truth.device)
    weights[[0, -1]] = .5
    if geometry == "pipe":
        weights *= torch.linspace(0, 1, len(weights), dtype=weights.dtype, device=truth.device)
    def magnitude(field):
        return (field.square() * weights).sum(dim=(1, 2)).sqrt()
    error = magnitude(prediction.to(truth) - truth)
    disturbance = magnitude(truth-steady[:, None])
    # Relative disturbance error is undefined for exactly steady samples.
    return 100*error/magnitude(truth), 100*error/disturbance.masked_fill(disturbance == 0, torch.nan)
