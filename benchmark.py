"""Reproducible, unmodified public OmniArch benchmark for pipe/channel flow.

Run baseline first, then model --device cuda and model --device cpu.
No source changes in vendor/, no training, no boundary projection, no similarity head.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import get_lapack_funcs, solve_banded
from scipy.special import j0, jn_zeros
import psutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CFG = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
JROOTS = jn_zeros(0, 2)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_rows(name, rows):
    write_json(OUT / (name + ".json"), rows)
    if rows:
        with (OUT / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def grid(geometry, n):
    return np.linspace(0.0, 1.0, n) if geometry == "pipe" else np.linspace(-1.0, 1.0, n)


def exact(geometry, n, times, Ub=1.0, transient=False):
    x = grid(geometry, n)
    base = (2.0 if geometry == "pipe" else 1.5) * (1.0 - x*x)
    u = np.broadcast_to(base, (len(times), n)).copy()
    if transient:
        for k, amp in enumerate(CFG["transient_mode_amplitudes"]):
            lam = JROOTS[k] if geometry == "pipe" else (2*k+1)*np.pi/2
            phi = j0(lam*x) if geometry == "pipe" else np.cos(lam*x)
            u += amp * np.exp(-lam*lam*np.asarray(times)[:, None]) * phi
    # Analytic no-slip boundaries are represented exactly in stored input/reference arrays.
    u[:, -1] = 0.0
    if geometry == "channel":
        u[:, 0] = 0.0
    return Ub * u


def coeffs(geometry, n, dtype=np.float64):
    x = grid(geometry, n)
    h = x[1]-x[0]
    if geometry == "channel":
        m = n-2
        lower = np.full(m-1, 1/h**2, dtype=dtype)
        diag = np.full(m, -2/h**2, dtype=dtype)
        upper = lower.copy()
    else:
        m = n-1
        r = x[:m]
        lower = (1/h**2 - 1/(2*r[1:]*h)).astype(dtype)
        diag = np.full(m, -2/h**2, dtype=dtype)
        diag[0] = -4/h**2
        upper = np.empty(m-1, dtype=dtype)
        upper[0] = 4/h**2
        upper[1:] = (1/h**2 + 1/(2*r[1:-1]*h)).astype(dtype)
    return lower, diag, upper


def unknowns(geometry, u):
    return u[..., :-1] if geometry == "pipe" else u[..., 1:-1]


def restore(geometry, v, n):
    out = np.zeros((*v.shape[:-1], n), dtype=v.dtype)
    if geometry == "pipe":
        out[..., :-1] = v
    else:
        out[..., 1:-1] = v
    return out


def tri_apply(lower, diag, upper, v):
    z = diag*v
    z[..., 1:] += lower*v[..., :-1]
    z[..., :-1] += upper*v[..., 1:]
    return z


def lap_full(geometry, u):
    """Includes the actual predicted wall values, without correcting the model."""
    n = u.shape[-1]
    x = grid(geometry, n)
    h = x[1]-x[0]
    if geometry == "channel":
        return (u[..., 2:]-2*u[..., 1:-1]+u[..., :-2])/h**2
    interior = (u[..., 2:]-2*u[..., 1:-1]+u[..., :-2])/h**2
    interior += (u[..., 2:]-u[..., :-2])/(2*h*x[1:-1])
    axis = 4*(u[..., 1:2]-u[..., :1])/h**2
    return np.concatenate([axis, interior], axis=-1)


def source(geometry, Ub):
    return (8.0 if geometry == "pipe" else 3.0)*Ub


def steady_fd(geometry, n, Ub=1.0, dtype=np.float64):
    lower, diag, upper = coeffs(geometry, n, dtype)
    band = np.zeros((3, len(diag)), dtype=dtype)
    band[0, 1:] = upper
    band[1] = diag
    band[2, :-1] = lower
    rhs = np.full(len(diag), -source(geometry, Ub), dtype=dtype)
    return restore(geometry, solve_banded((1, 1), band, rhs, check_finite=False), n)


def forecast_fd(geometry, initial, dt, steps, Ub=1.0, dtype=np.float64):
    """Second-order centered space, Crank-Nicolson; LU factored once per rollout."""
    n = len(initial)
    lower, diag, upper = coeffs(geometry, n, dtype)
    factor, solve = get_lapack_funcs(("gttrf", "gttrs"), dtype=dtype)
    dl, d, du, du2, ipiv, info = factor(-0.5*dt*lower, 1-0.5*dt*diag, -0.5*dt*upper)
    if info != 0:
        raise RuntimeError(f"gttrf info={info}")
    v = unknowns(geometry, np.asarray(initial, dtype=dtype)).copy()
    result = []
    for _ in range(steps):
        rhs = v + np.asarray(0.5*dt, dtype=dtype)*tri_apply(lower, diag, upper, v)
        rhs += np.asarray(dt*source(geometry, Ub), dtype=dtype)
        sol, info = solve(dl, d, du, du2, ipiv, rhs[:, None])
        if info != 0:
            raise RuntimeError(f"gttrs info={info}")
        v = sol[:, 0]
        result.append(restore(geometry, v, n))
    return np.stack(result)


def weights(geometry, x):
    w = np.ones(len(x))
    w[[0, -1]] = 0.5
    if geometry == "pipe":
        w *= x
    return w/w.sum()


def metrics(geometry, pred, true, Ub, initial=None, dt=None):
    pred = np.atleast_2d(pred).astype(np.float64)
    true = np.atleast_2d(true).astype(np.float64)
    x = grid(geometry, pred.shape[-1])
    w = weights(geometry, x)
    diff = pred-true
    err = np.sqrt(np.sum(diff*diff*w)/np.sum(true*true*w))
    wall = np.abs(pred[..., -1]).max()
    if geometry == "channel":
        wall = max(wall, np.abs(pred[..., 0]).max())
        symmetry = np.abs(pred-pred[..., ::-1]).max()/Ub
    else:
        h = x[1]-x[0]
        symmetry = np.abs((-3*pred[..., 0]+4*pred[..., 1]-pred[..., 2])/(2*h)).max()/Ub
    if initial is None:
        res = lap_full(geometry, pred) + source(geometry, Ub)
    else:
        prev = np.concatenate([np.asarray(initial)[None], pred[:-1]])
        res = unknowns(geometry, (pred-prev)/dt) - lap_full(geometry, (pred+prev)/2) - source(geometry, Ub)
    return {
        "rel_l2_weighted": float(err),
        "nrmse_unweighted": float(np.linalg.norm(diff)/np.linalg.norm(true)),
        "max_abs_m_s": float(np.abs(diff).max()),
        "max_abs_over_Ub": float(np.abs(diff).max()/Ub),
        "wall_max_over_Ub": float(wall/Ub),
        "symmetry_or_axis_gradient": float(symmetry),
        "discrete_residual_rms_over_source": float(np.sqrt(np.mean(res*res))/source(geometry, Ub)),
        "final_rel_l2": float(np.sqrt(np.sum(diff[-1]**2*w)/np.sum(true[-1]**2*w))),
    }


def time_calls(fn, repeats=7, inner=1, warmups=2, sync=None):
    for _ in range(warmups):
        fn()
    timings, outputs = [], []
    for _ in range(repeats):
        if sync:
            sync()
        start = time.perf_counter_ns()
        for _ in range(inner):
            result = fn()
        if sync:
            sync()
        timings.append((time.perf_counter_ns()-start)*1e-6/inner)
        outputs.append(np.asarray(result).copy())
    return {
        "median_ms": statistics.median(timings),
        "min_ms": min(timings), "max_ms": max(timings),
        "mean_ms": statistics.mean(timings), "std_ms": statistics.stdev(timings),
        "repeats": repeats, "inner_calls": inner,
        "repeat_max_abs_difference": float(max(np.max(np.abs(z-outputs[0])) for z in outputs)),
        "raw_ms": timings,
    }


def environment():
    mi = psutil.Process().memory_info()
    return {
        "time_Asia_Shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
        "logical_cpus": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
        "process_rss_bytes": mi.rss, "process_peak_working_set_bytes": getattr(mi, "peak_wset", mi.rss),
    }


def data_case(geometry, n, case, test):
    k, steps, dt = CFG["history_frames"], CFG["forecast_steps"], CFG["dimensionless_dt"]
    all_u = exact(geometry, n, np.arange(k+steps)*dt, case["Ub"], test == "transient")
    return all_u[:k].astype(np.float32), all_u[k:]


def baseline():
    rows, convergence, temporal, timings = [], [], [], []
    for geo in CFG["geometries"]:
        for n in [16, 32, 64, 128, 256, 512, 1024]:
            u = steady_fd(geo, n)
            truth = exact(geo, n, [0])[0]
            dense_x = grid(geo, 8193)
            interp = np.interp(dense_x, grid(geo, n), u)
            dense_true = exact(geo, 8193, [0])[0]
            m = metrics(geo, u, truth, 1.0)
            w = weights(geo, dense_x)
            convergence.append({"geometry":geo, "N":n,
                "node_rel_l2":m["rel_l2_weighted"],
                "linear_interpolation_rel_l2":float(np.sqrt(np.sum((interp-dense_true)**2*w)/np.sum(dense_true**2*w)))})
        for n in [32, 64, 128, 256, 512]:
            # Small time step isolates space convergence for the transient control.
            initial = exact(geo, n, [0.09], transient=True)[0]
            target = exact(geo, n, [0.29], transient=True)[0]
            pred = forecast_fd(geo, initial, 0.00025, 800)[-1]
            temporal.append({"geometry":geo,"kind":"space","N":n,"dt":0.00025,
                             "rel_l2":metrics(geo,pred,target,1.0)["rel_l2_weighted"]})
        for dt in [0.02, 0.01, 0.005, 0.0025]:
            n=1025
            initial=exact(geo,n,[0.09],transient=True)[0]
            target=exact(geo,n,[0.29],transient=True)[0]
            pred=forecast_fd(geo,initial,dt,round(0.2/dt))[-1]
            temporal.append({"geometry":geo,"kind":"time","N":n,"dt":dt,
                             "rel_l2":metrics(geo,pred,target,1.0)["rel_l2_weighted"]})
        for n in CFG["grids"]:
            for case in CFG["physical_cases"]:
                u = steady_fd(geo,n,case["Ub"])
                rows.append({"method":"FD-steady-f64","geometry":geo,"N":n,"case":case["id"],"test":"steady_BVP",
                             **metrics(geo,u,exact(geo,n,[0],case["Ub"]),case["Ub"])})
                for test in ["steady","transient"]:
                    hist, target = data_case(geo,n,case,test)
                    for dtype, label in [(np.float64,"f64"),(np.float32,"f32")]:
                        pred=forecast_fd(geo,hist[-1],CFG["dimensionless_dt"],CFG["forecast_steps"],case["Ub"],dtype)
                        rows.append({"method":"FD-CN-"+label,"geometry":geo,"N":n,"case":case["id"],"test":test,
                            **metrics(geo,pred,target,case["Ub"],hist[-1],CFG["dimensionless_dt"])})
                    pred=np.repeat(hist[-1:],[CFG["forecast_steps"]],axis=0)
                    rows.append({"method":"persistence","geometry":geo,"N":n,"case":case["id"],"test":test,
                        **metrics(geo,pred,target,case["Ub"],hist[-1],CFG["dimensionless_dt"])})
            # End-to-end timing including operator assembly and factorization per solve/rollout.
            case=CFG["physical_cases"][0]
            for test in ["steady","transient"]:
                hist,target=data_case(geo,n,case,test)
                for dtype,label in [(np.float64,"f64"),(np.float32,"f32")]:
                    fn=lambda:forecast_fd(geo,hist[-1],0.01,20,case["Ub"],dtype)
                    timings.append({"method":"FD-CN-"+label,"geometry":geo,"N":n,"test":test,"batch":1,
                                    **time_calls(fn,inner=20)})
                timings.append({"method":"persistence","geometry":geo,"N":n,"test":test,"batch":1,
                                **time_calls(lambda:np.repeat(hist[-1:],20,axis=0),inner=500)})
            timings.append({"method":"FD-steady-f64","geometry":geo,"N":n,"test":"steady_BVP","batch":1,
                            **time_calls(lambda:steady_fd(geo,n,case["Ub"]),inner=200)})
    save_rows("baseline_metrics",rows)
    save_rows("steady_convergence",convergence)
    save_rows("transient_convergence",temporal)
    save_rows("baseline_timings",timings)
    write_json(OUT/"environment_baseline.json",environment())
    print("Baseline complete",len(rows),"metric records",flush=True)
    print(json.dumps(convergence[:3],indent=2),flush=True)


def model_run(device, ckpt):
    import torch
    sys.path.insert(0,str(ROOT/"vendor"/"omniarch_pt"))
    from omniarch import OmniArchPredictor
    torch.manual_seed(20261107)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter()
    p=OmniArchPredictor(str(ckpt),device=device)
    load_seconds=time.perf_counter()-start
    sync=torch.cuda.synchronize if device=="cuda" else None
    if sync: sync()
    info={"config":p.config.to_dict(),"checkpoint_info":p.info,"n_params":p.n_params,
          "load_seconds":load_seconds,"device":str(p.device),"amp":p.amp,
          "torch":torch.__version__,"cuda_runtime":torch.version.cuda,
          "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          "torch_threads":torch.get_num_threads(),"checkpoint_path":str(ckpt)}
    write_json(OUT/f"model_info_{device}.json",info)
    print("Loaded",p.n_params,"parameters on",device,"in",round(load_seconds,3),"s",flush=True)
    rows,timings=[],[]
    grids=CFG["grids"] if device=="cuda" else [128]
    cases=CFG["physical_cases"] if device=="cuda" else CFG["physical_cases"][:1]
    for geo in CFG["geometries"]:
        for n in grids:
            for test in ["steady","transient"]:
                histories=[]; targets=[]
                for case in cases:
                    hist,target=data_case(geo,n,case,test)
                    histories.append(hist[:,None,:]); targets.append(target)
                batch=np.stack(histories)
                start=time.perf_counter()
                pred=p.predict(batch,20,batched=True)
                elapsed=time.perf_counter()-start
                for j,case in enumerate(cases):
                    met=metrics(geo,pred[j,:,0],targets[j],case["Ub"],histories[j][-1,0],0.01)
                    rows.append({"method":"OmniArch-"+device,"geometry":geo,"N":n,"case":case["id"],"test":test,**met})
                np.savez_compressed(OUT/f"predictions_{device}_{geo}_{n}_{test}.npz",
                    history=batch,prediction=pred,truth=np.stack(targets)[:,:,None,:],x=grid(geo,n))
                print(device,geo,n,test,"relL2",rows[-len(cases)]["rel_l2_weighted"],"batch_seconds",elapsed,flush=True)
                save_rows(f"model_metrics_{device}",rows)
                if n==128:
                    h=histories[0]
                    if sync: torch.cuda.reset_peak_memory_stats()
                    result=time_calls(lambda:p.predict(h,20),sync=sync)
                    peak=int(torch.cuda.max_memory_allocated()) if sync else None
                    timings.append({"method":"OmniArch-"+device,"geometry":geo,"N":n,"test":test,
                                    "batch":1,"cuda_peak_allocated_bytes":peak,**result})
                    if device=="cuda":
                        timings.append({"method":"OmniArch-"+device,"geometry":geo,"N":n,"test":test,
                            "batch":len(cases),"cuda_peak_allocated_bytes":None,
                            **time_calls(lambda:p.predict(batch,20,batched=True),sync=sync)})
                    save_rows(f"model_timings_{device}",timings)
    if device=="cuda":
        z=np.zeros((10,1,128),dtype=np.float32)
        zp=p.predict(z,20)
        np.save(OUT/"zero_history_prediction.npy",zp)
        write_json(OUT/"zero_history_diagnostic.json",{
            "identical_history_for_all_pressure_gradients":True,
            "output_max_abs_m_s":float(np.max(np.abs(zp))),
            "meaning":"The public API has no pressure-gradient, geometry, viscosity or wall-BC arguments. This is an interface-identifiability diagnostic, not a valid BVP accuracy competition."})
    save_rows(f"model_metrics_{device}",rows)
    save_rows(f"model_timings_{device}",timings)
    write_json(OUT/f"environment_{device}.json",environment())
    print("Model phase complete",device,flush=True)


def validate():
    checks=[]
    for geo in ["pipe","channel"]:
        for n in [17,64,129]:
            u=steady_fd(geo,n)
            ref=exact(geo,n,[0])[0]
            e=np.max(np.abs(u-ref))
            assert e<1e-9,(geo,n,e)
            assert np.max(np.abs(lap_full(geo,ref)+source(geo,1)))<1e-8
            assert abs(u[-1])==0
            if geo=="channel": assert abs(u[0])==0
            checks.append({"geometry":geo,"N":n,"max_abs_error":float(e)})
        # Temporal refinement on exact non-steady solution, not a code-mirroring fixture.
        n=1025
        init=exact(geo,n,[0.09],transient=True)[0]
        truth=exact(geo,n,[0.29],transient=True)[0]
        errors=[metrics(geo,forecast_fd(geo,init,dt,round(0.2/dt))[-1],truth,1)["rel_l2_weighted"] for dt in [0.02,0.01,0.005]]
        assert errors[0]>3*errors[1] and errors[1]>2.5*errors[2],(geo,errors)
        checks.append({"geometry":geo,"time_errors":errors})
    write_json(OUT/"validation.json",checks)
    print("Analytic and refinement checks passed",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("phase",choices=["validate","baseline","model"])
    ap.add_argument("--device",default="cuda",choices=["cuda","cpu"])
    ap.add_argument("--checkpoint",type=Path,default=Path(os.environ.get("OMNIARCH_CKPT", ROOT/"weights"/"omniarch-base.pt")))
    args=ap.parse_args()
    if args.phase=="validate": validate()
    elif args.phase=="baseline": baseline()
    else: model_run(args.device,args.checkpoint)
