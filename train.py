"""Supervised fine-tuning of OmniArch's own 1D decoder."""
from copy import deepcopy

import torch
from torch.nn import functional as F

from analytic import reference, sample_cases, trajectory_errors
from base_pipeline import HISTORY, predict


def fit_decoder(network, geometry, updates=160, seed=17, *, training_bank=None):
    if updates < 1:
        raise ValueError("Training updates must be positive.")
    offset = int(geometry == "channel")
    if training_bank is None:
        train_cases = sample_cases(geometry, 256, 100 + offset, "train")
        train_u, _ = reference(geometry, train_cases, frames=36)
    else:
        train_cases, train_u = training_bank
        if train_u.ndim != 3 or train_u.shape[:2] != (len(train_cases), 36):
            raise ValueError("Expected a prepared bank of [cases, 36, nodes].")
    val_cases = sample_cases(geometry, 12, 200 + offset, "validation")
    truth, steady = reference(geometry, val_cases)
    device = next(network.parameters()).device
    train_u = train_u.to(device=device, dtype=torch.float32)
    speeds = torch.tensor([c["Ub"] for c in train_cases], device=device)
    network.requires_grad_(False)
    decoder = network.dec["1"]
    decoder.requires_grad_(True)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-4)
    random = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    checkpoints = {0, updates // 2, updates}
    for i in range(updates + 1):
        if i in checkpoints:
            network.eval()
            prediction = predict(network, truth[:, :HISTORY]).cpu()
            vel, _ = trajectory_errors(geometry, prediction, truth[:, HISTORY:], steady)
            validation = vel.mean().item()
            if not torch.isfinite(vel).all():
                raise FloatingPointError("Nonfinite validation error.")
            if validation < best_loss:
                best_loss = validation
                best_step = i
                best = deepcopy(decoder.state_dict())
            print(f"{geometry} seed {seed}: update {i}, validation {validation:.4f}%", flush=True)
        if i == updates:
            break
        network.train()
        ids = torch.randint(len(train_cases), (8,), generator=random)
        length = (10, 15)[i % 2]
        start = torch.randint(36-length, (len(ids),), generator=random)
        times = start[:, None] + torch.arange(length)
        history = train_u[ids[:, None], times].unsqueeze(2)
        target = train_u[ids, start+length]
        pred, (mean, std) = network(history)
        next_frame = (pred[:, -1:] * std + mean)[:, 0, 0]
        loss = F.mse_loss(next_frame / speeds[ids, None], target / speeds[ids, None])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    decoder.load_state_dict(best)
    network.eval()
    return best_step
