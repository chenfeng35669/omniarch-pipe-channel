"""Timed, paired OmniArch experiments; stored results are per trajectory and per update."""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch

from analytic import GEOMETRIES, original_test_cases, reference, sample_cases, trajectory_errors
from base_pipeline import CONTEXT, HISTORY, ROOT, STEPS, load_base, predict
from train import fit_decoder

FINAL_DIR = ROOT / "results" / "native_posttraining"
# Training reruns must never replace the validated weights, even for the same seed.
OUT = Path(os.environ.get("OMNIARCH_EXPERIMENT_DIR") or ROOT / "results" / "reruns").resolve()
if OUT == FINAL_DIR.resolve() or FINAL_DIR.resolve() in OUT.parents:
    raise ValueError("Use an experiment directory outside the saved final results.")


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def write(path, obj):
    path.write_text(json.dumps(obj, allow_nan=False, indent=2), encoding="utf-8")


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def brief(v):
    return [float(x) for x in torch.quantile(torch.as_tensor(v, dtype=torch.float64),
                                              torch.tensor([0., .5, 1.], dtype=torch.float64))]


def fit_original(net, g, seed):
    dest = OUT / f"{g}_{seed}_initial"
    norms, losses, group_norms = [], [], []
    clip = torch.nn.utils.clip_grad_norm_
    from torch.nn import functional as F
    mse = F.mse_loss

    def observe_clip(params, *args, **kwargs):
        gp = []
        for part in (net.dec["1"].head, net.dec["1"].mlp):
            gp.append(torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in part.parameters()])))
        group_norms.append(torch.stack(gp).tolist())
        out = clip(params, *args, **kwargs)
        norms.append(out.item())
        return out

    def observe_loss(*args, **kw):
        loss = mse(*args, **kw)
        losses.append(loss.item())
        return loss

    t = sync()
    with patch.object(torch.nn.utils, "clip_grad_norm_", observe_clip), patch.object(F, "mse_loss", observe_loss):
        best = fit_decoder(net, g, 160, seed)
    elapsed = sync() - t
    torch.save(cpu_state(net.dec["1"]), dest.with_suffix(".pt"))
    write(dest.with_suffix(".json"), dict(seconds=elapsed, selected=best, grad=norms, head_mlp=group_norms,
                                           loss=losses, seed=seed))
    print(g, seed, "initial", elapsed, brief(norms), flush=True)


def cases_mixed(g, n, seed):
    # Independent profiles; every other draw remains in the original training domain.
    base = sample_cases(g, n, seed, "validation")
    r = torch.rand(n, 10, generator=torch.Generator().manual_seed(seed+70001), dtype=torch.float64)
    limits = torch.tensor([.8, .35, .2, .12, .07, .04, .04, .025], dtype=r.dtype)
    for i, c in enumerate(base):
        if i % 2 == 0:
            c.update(dt=(.001, .0025)[(i//2) % 2], t0=.001*float(r[i, 0]),
                     coefficients=((2*r[i, 1:9]-1)*limits).tolist())
        else:
            c["coefficients"] += [0.]*4
        if i % 8 == 7:
            c["coefficients"] = [0.]*8
    return base


def suites(g):
    offset = int(g == "channel")
    return dict(four=sample_cases(g, 48, 3110+offset, "test"), eight=original_test_cases(g))


def native_next(net, h):
    pred, (mu, sd) = net(h.unsqueeze(2))
    return (pred[:, -1:] * sd + mu)[:, 0, 0]


def centered_next(net, h, increments=False):
    """Invertible per-window velocity coordinates; only the model supplies a new increment."""
    anchor = h[:, -1:]
    diff = h.diff(dim=1) if increments else h-anchor
    safe = diff.square().mean((1, 2), keepdim=True).clamp_min(1e-18).sqrt()
    return anchor[:, 0] + safe[:, 0]*native_next(net, diff/safe)


@torch.no_grad()
def centered_predict(net, h, increments=False):
    if h.ndim != 3 or h.shape[0] == 0 or h.shape[1] != HISTORY:
        raise ValueError("Expected a nonempty [batch, 10 observed frames, nodes] history.")
    if h.shape[-1] // 2 + 1 < net.config.modes[1]:
        raise ValueError("Grid is too short for the checkpoint's Fourier encoder.")
    net.eval()
    h = h.to(next(net.parameters()).device, dtype=torch.float32)
    out = []
    for j in range(STEPS):
        nxt = centered_next(net, h, increments)
        out.append(nxt)
        h = torch.cat([h, nxt[:, None]], 1)[:, -CONTEXT:]
    return torch.stack(out, dim=1)


def predictor_for(arm):
    if arm.startswith("increment"):
        return lambda net, h: centered_predict(net, h, increments=True)
    return centered_predict if arm.startswith("center") else predict


def apply_recommended(net, geometry):
    """Install the selected decoder in an existing Base model; verify the saved weights."""
    if geometry not in GEOMETRIES:
        raise ValueError(geometry)
    bundle = json.loads((FINAL_DIR/"recommended.json").read_text(encoding="utf-8"))[geometry]
    path = FINAL_DIR/bundle["decoder"]
    if bundle["representation"] != "increment_rel_long" or hashlib.sha256(path.read_bytes()).hexdigest() != bundle["decoder_sha256"]:
        raise ValueError(f"Unexpected final decoder for {geometry}: {path}")
    net.dec["1"].load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.eval(), predictor_for(bundle["representation"])


def load_recommended(geometry, device=None, checkpoint=None):
    """Return model and velocity-history-only predictor chosen by validation across the two seeds."""
    return apply_recommended(load_base(checkpoint=checkpoint, device=device), geometry)


@torch.no_grad()
def evaluate(net, g, cases, teacher=False, prediction=predict):
    tg = sync()
    u, steady = reference(g, cases)
    target = u[:, HISTORY:]
    ud = u.float().cuda()
    t0 = sync()
    forecast = torch.cat([prediction(net, h[:, :HISTORY]) for h in ud.split(8)])
    end = sync()
    f = forecast.cpu()
    vel, pert = trajectory_errors(g, f, target, steady)
    out = dict(seconds=end-t0, prep_seconds=t0-tg, total=vel.tolist(), pert=pert.tolist())
    for j in [0, 4, 9, 19]:
        out[f"step{j+1}"] = trajectory_errors(g, f[:, j:j+1], target[:, j:j+1], steady)[0].mean().item()
    ub = torch.tensor([c["Ub"] for c in cases], dtype=u.dtype)[:, None, None]
    edges = [-1] if g == "pipe" else [0, -1]
    out["wall_rms_pct_Ub"] = ((f[:, :, edges]/ub).square().mean().sqrt()*100).item()
    if teacher:
        t1 = sync()
        allpred = []
        for group in ud.split(8):
            allpred.append(torch.stack([native_next(net, group[:, max(0, HISTORY+j-CONTEXT):HISTORY+j])
                                        for j in range(STEPS)], dim=1))
        out["teacher_seconds"] = sync()-t1
        p = torch.cat(allpred).cpu()
        out["teacher_total"] = trajectory_errors(g, p, target, steady)[0].mean().item()
        out["teacher_step20"] = trajectory_errors(g, p[:, -1:], target[:, -1:], steady)[0].mean().item()
    return out


def bench(net, g, seed, arm, teacher=False):
    predictor = predictor_for(arm)
    res = {k: evaluate(net, g, v, teacher, predictor) for k, v in suites(g).items()}
    write(OUT/f"{g}_{seed}_{arm}_test.json", res)
    print(g, seed, arm, [(k, round(statistics.mean(v["total"]), 4), round(v["seconds"], 2)) for k,v in res.items()], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["initial", "post", "probe"])
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--geometry", nargs="+", choices=GEOMETRIES, default=list(GEOMETRIES))
    p.add_argument("--arms", nargs="+", default=["same", "mix", "roll", "momentum", "energy", "both"])
    args = p.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    t = time.perf_counter()
    net = load_base(device="cuda")
    native_weights = cpu_state(net.dec["1"])
    print("setup seconds", sync()-t, flush=True)
    for g in args.geometry:
        if args.stage == "initial":
            net.dec["1"].load_state_dict(native_weights)
            if args.seed == 17:
                bench(net, g, args.seed, "base")
            fit_original(net, g, args.seed)
            bench(net, g, args.seed, "initial", teacher=args.seed==17)
        else:
            from diagnose_posttraining import run
            run(args, net, g)


if __name__ == "__main__":
    main()
