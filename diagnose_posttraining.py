"""Post-training branches from the same fitted decoder. Physics is a training loss only."""
import statistics

import torch

from analytic import original_test_cases, reference, sample_cases, trajectory_errors
from base_pipeline import CONTEXT, HISTORY, predict
from diagnose_omniarch import OUT, bench, cases_mixed, centered_next, predictor_for, cpu_state, evaluate, native_next, sync, write


def balance(g, old, new, dt):
    """Weak momentum moments (incl. total momentum) and pressure work minus viscous dissipation.

    Speeds are divided by Ub, coordinates by radius/half-spacing and time by a^2/nu.
    Use midpoint velocities and linear cell quadrature; no residual is used for prediction.
    """
    pipe = g == "pipe"
    n = old.shape[-1]
    x = torch.linspace(0 if pipe else -1, 1, n, device=old.device, dtype=old.dtype)
    dx = (1 if pipe else 2)/(n-1)
    measure = x.clone() if pipe else torch.ones_like(x)
    cell = (measure[1:]+measure[:-1])*.5*dx
    w = measure*dx
    w[[0, -1]] *= .5
    volume = w.sum()
    cell, w = cell/volume, w/volume
    force = 8. if pipe else 3.
    test = torch.stack([1-x.square(), (1-x.square())*(x.square() if pipe else x),
                        (1-x.square())**2])
    test /= ((test.square()*w).sum(-1).sqrt())[:, None]
    vel = (old+new)*.5
    deriv = vel.diff()/dx
    du = new-old
    moments = (du*w)@test.T + dt[:, None]*((deriv*cell)@(test.diff()/dx).T - force*(test*w).sum(-1))
    right = (3*vel[:, -1]-4*vel[:, -2]+vel[:, -3])/(2*dx)
    left = (-3*vel[:, 0]+4*vel[:, 1]-vel[:, 2])/(2*dx)
    boundary = (right - (0 if pipe else left))/volume
    global_m = (du*w).sum(-1)-dt*(force+boundary)
    energy = (du*vel*w).sum(-1) - dt*(force*(vel*w).sum(-1)-(deriv.square()*cell).sum(-1))
    # Bound sensitivity of the endpoint derivative in the global momentum training term.
    return moments, global_m, energy, 1+dt/dx


def val_score(net, g, validation, predictor):
    with torch.no_grad():
        scores = []
        for part in validation:
            u, steady = part
            p = torch.cat([predictor(net, b) for b in u[:, :HISTORY].split(8)]).cpu()
            scores.append(trajectory_errors(g, p, u[:, HISTORY:], steady)[0].mean().item())
        return statistics.mean(scores)


def branch(net, g, seed, arm, val):
    t = sync()
    off = int(g == "channel")
    mixed = arm != "same"
    k = 1 if arm in ("same", "mix") else 3
    increments = arm.startswith("increment")
    centered = arm.startswith("center") or increments
    momentum = arm=="momentum" or arm.endswith("both")
    energy = arm=="energy" or arm.endswith("both")
    cases = (cases_mixed(g, 256, 42011+off) if mixed else
             sample_cases(g, 256, 100+off, "train"))
    u = reference(g, cases, frames=36)[0].float().cuda()
    speed = torch.tensor([c["Ub"] for c in cases], device="cuda")
    spacing = torch.tensor([c["dt"] for c in cases], device="cuda")
    net.requires_grad_(False)
    dec = net.dec["1"].requires_grad_(True)
    extended = arm == "increment_rel_long"
    opt = torch.optim.AdamW(dec.parameters(), lr=5.e-5 if extended else 1.e-4, weight_decay=1.e-4)
    rng = torch.Generator().manual_seed(seed+(2300 if extended else 900))
    grads, errors = [], []
    val_hist = {}
    best_loss = float("inf")
    nsteps = 320 if extended else 160
    train_seconds = 0.
    validation_seconds = 0.
    for it in range(nsteps+1):
        checks = (0, 80, 160, 240, 320) if extended else (0,nsteps//2,nsteps)
        if it in checks:
            tv = sync()
            val_hist[it] = val_score(net, g, val, predictor_for(arm))
            if val_hist[it] < best_loss:
                best_loss = val_hist[it]
                best = it
                weights = cpu_state(dec)
            validation_seconds += sync()-tv
        if it == nsteps:
            break
        tt = sync()
        ids = torch.randint(len(cases), (8,), generator=rng)
        length = (10, 15)[it%2]
        # All arms see exactly the same sample indices and window starts.
        start = torch.randint(36-length-2, (8,), generator=rng)
        idx = ids.cuda()
        times = (start[:, None]+torch.arange(length)).cuda()
        h = u[idx[:, None], times]
        ub = speed[idx, None]
        dt = spacing[idx]
        # A fixed scale from the observed window; floor keeps steady profiles well-defined.
        scale_loss = ((h-h[:, -1:]).square().mean((1, 2)).sqrt()/ub[:, 0]).clamp_min(.01) if "rel" in arm else torch.ones_like(dt)
        loss = torch.zeros((), device="cuda")
        net.train()
        for j in range(k):
            gt = u[idx, times[:, -1]+1+j]
            new = centered_next(net, h, increments) if centered else native_next(net, h)
            term = (((new-gt)/ub)/scale_loss[:, None]).square().mean()
            if momentum or energy:
                m, gm, e, scale = balance(g, h[:, -1]/ub, new/ub, dt)
                if momentum:
                    term = term + .25*((m/scale_loss[:, None]).square().mean()+(gm/(scale*scale_loss)).square().mean())
                if energy:
                    term = term + .25*(e/scale_loss).square().mean()
            loss = loss + term/k
            if j+1 < k:
                h = torch.cat([h, new[:, None]], 1)[:, -CONTEXT:]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(dec.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        grads.append(grad.item())
        errors.append(loss.item())
        train_seconds += sync()-tt
    dec.load_state_dict(weights)
    net.eval()
    elapsed = sync()-t
    path = OUT/f"{g}_{seed}_{arm}"
    write(path.with_suffix(".json"), dict(seconds=elapsed, optimizer_seconds=train_seconds,
                                          validation_seconds=validation_seconds, selected=best,
                                          val=val_hist, grad=grads, loss=errors, seed=seed))
    torch.save(weights, path.with_suffix(".pt"))
    print(g, seed, arm, "selected", best, "val", val_hist, "seconds", round(elapsed,1), flush=True)


def probe(net, g, seed):
    # Same Ub, lowest four coefficients, and start-time percentile for all 2x2x2 cells.
    ext = original_test_cases(g)
    tg = sync()
    original_u = reference(g, ext)[0]
    short = [dict(c, coefficients=c["coefficients"][:4]) for c in ext]
    truncated, steady = reference(g, short)
    def field_fraction(start, end):
        return statistics.mean(trajectory_errors(g, truncated[:, start:end], original_u[:, start:end], steady)[0].tolist())
    audit = dict(high4_history=field_fraction(0,10), high4_forecast=field_fraction(10,30),
                 fp32_roundtrip=statistics.mean(trajectory_errors(g, original_u.float().double()[:, 10:], original_u[:, 10:], steady)[0].tolist()),
                 data_seconds=sync()-tg)
    for modes in (4, 8):
        for dt in ("small", "train"):
            for t0 in ("early", "train"):
                cases = [dict(c, coefficients=c["coefficients"][:modes],
                              dt=c["dt"] if dt=="small" else (.005,.01,.02)[i%3],
                              t0=c["t0"] if t0=="early" else c["t0"]*30) for i,c in enumerate(ext)]
                key = f"m{modes}_{dt}_{t0}"
                if key == "m8_small_early":
                    import json
                    audit[key] = json.loads((OUT/f"{g}_{seed}_initial_test.json").read_text())["eight"]
                else:
                    audit[key] = evaluate(net, g, cases)
    write(OUT/f"{g}_{seed}_probe.json", audit)


def run(args, net, g):
    dest = OUT / f"{g}_{args.seed}_initial.pt"
    initial = torch.load(dest, weights_only=True)
    if args.stage=="probe":
        net.dec["1"].load_state_dict(initial)
        probe(net, g, args.seed)
        return
    off = int(g=="channel")
    mixed_val = cases_mixed(g, 32, 53261+off)[::2]
    val = [reference(g, sample_cases(g, 16, 52131+off, "validation")), reference(g, mixed_val)]
    for arm in args.arms:
        if arm not in ("same", "mix", "roll", "momentum", "energy", "both", "center", "center_both", "center_rel", "center_rel_both","increment_rel","increment_rel_both","increment_rel_long"):
            raise ValueError(arm)
        net.dec["1"].load_state_dict(torch.load(OUT/f"{g}_{args.seed}_increment_rel.pt", weights_only=True)
                                     if arm == "increment_rel_long" else initial)
        branch(net, g, args.seed, arm, val)
        bench(net, g, args.seed, arm)
