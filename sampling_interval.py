"""Paired decoder continuation: change only the training sampling interval.

Fixed-horizon scores use the same physical profiles, forecast origin, target times
and duration across intervals (in nondimensional time a^2/nu). No dt enters the network.
"""
import csv
import hashlib
import json
import statistics as st

import torch

from analytic import GEOMETRIES, original_test_cases, reference, sample_cases, trajectory_errors
from base_pipeline import CONTEXT, HISTORY, ROOT, load_base
from diagnose_omniarch import FINAL_DIR, OUT, cases_mixed, centered_next, cpu_state, sync

DEST = OUT/"sampling_interval"
SEEDS = (17,29)
CHECKS = (80,160,240,320)
VAL_DT = (.0015,.003,.006,.012)
TEST_DT = (.00125,.00375,.0075,.015)
SEEN_DT = (.001,.0025,.005)  # all divide the test comparison grid (.015)
ETOL = 1.e-6  # percentage points; roundoff only
START = {"pipe":"11f23e1b03c4f07e10709d1f3085e188ccde69c606597ddd31e3d0dc8b1495c2",
         "channel":"2e2a2dcf57df82de7e21ced3dbb9d421542f6749ad457df6dd7445d6e415a345"}


def bank(g):
    """Change half of each time-step/profile stratum; retain coefficients, Ub and initial time."""
    off=int(g=="channel")
    old=cases_mixed(g,256,42011+off)
    new=[dict(c) for c in old]
    rng=torch.Generator().manual_seed(77311+off)
    strata={}
    for i,c in enumerate(old):
        key=(i%2, c["dt"], any(c["coefficients"]))
        strata.setdefault(key,[]).append(i)
    picked=[]; leftovers=[]
    for group in strata.values():
        order=[group[j] for j in torch.randperm(len(group),generator=rng).tolist()]
        picked.extend(order[:len(group)//2])
        if len(group)%2: leftovers.append(order[len(group)//2])
    picked.extend(leftovers[:128-len(picked)])
    grid0={c["dt"] for c in old}
    disallowed=grid0|set(VAL_DT+TEST_DT)|{.0016,.00375,.0125}
    for idx in picked:
        # Log-uniform stratification covers a continuous range within the original extrema.
        p=float(torch.rand((),generator=rng,dtype=torch.float64))
        dt=.001*(20.**p)
        assert min(abs(dt-v) for v in disallowed)>1e-12
        new[idx]["dt"]=dt
    assert sum(a["dt"]!=b["dt"] for a,b in zip(old,new))==128
    assert all(dict(a,dt=b["dt"])==b for a,b in zip(old,new))
    return {arm:reference(g,cs,36)[0].float().cuda() for arm,cs in (("original_dt",old),("varied_dt",new))},torch.tensor([c["Ub"] for c in old],device="cuda"),[new[i]["dt"] for i in picked]


def make_set(g,cs,steps=20,keep=None):
    u,ss=reference(g,cs,HISTORY+steps)
    keep=list(range(1,steps+1)) if keep is None else keep
    assert keep[-1]==steps
    return dict(h=u[:,:HISTORY].float().cuda(),target=u[:,[HISTORY+i-1 for i in keep]],
                steady=ss,steps=steps,keep=keep)


def fixed(g,n,seed,spacings,grid):
    profiles=cases_mixed(g,2*n,seed)[::2]  # independent, dynamic eight-mode profiles
    answer={}
    base_truth=None
    for dt in spacings:
        # Last observation is always at 0.135 + the same profile-specific small offset.
        cs=[dict(c,t0=max(0.,.135+c["t0"]-(HISTORY-1)*dt),dt=dt) for c in profiles]
        keep=[round(t/dt) for t in grid]
        assert all(abs(k*dt-t)<1e-12 for k,t in zip(keep,grid))
        item=make_set(g,cs,keep[-1],keep)
        # Confirm equality before casting the history to FP32; compare all reference targets.
        if base_truth is not None:
            torch.testing.assert_close(item["target"],base_truth,rtol=1e-12,atol=1e-14)
        base_truth=item["target"]
        answer[str(dt)]=item
    return answer


@torch.no_grad()
def score(net,g,item):
    net.eval()
    result=[]
    t=sync()
    for h in item["h"].split(8):
        out=[]
        for i in range(1,item["steps"]+1):
            nxt=centered_next(net,h,increments=True)
            if i in item["keep"]: out.append(nxt)
            h=torch.cat((h,nxt[:,None]),1)[:,-CONTEXT:]
        result.append(torch.stack(out,1))
    pred=torch.cat(result)
    elapsed=sync()-t
    vel,pert=trajectory_errors(g,pred.cpu(),item["target"],item["steady"])
    assert torch.isfinite(vel).all() and torch.isfinite(pert).all()
    return dict(field=vel.mean().item(),pert=pert.mean().item(),seconds=elapsed,
                n=len(vel),steps=item["steps"],field_per_case=vel.tolist(),pert_per_case=pert.tolist())


def eval_sets(net,g,sets):
    return {name:score(net,g,item) for name,item in sets.items()}


def mean_metric(scores,keys,metric="field"):
    return st.mean(scores[k][metric] for k in keys)


def validation(net,g,sets):
    s=eval_sets(net,g,sets)
    return {"unseen":mean_metric(s,map(str,VAL_DT)), "four":s["four"]["field"], "eight":s["eight"]["field"],
            "unseen_pert":mean_metric(s,map(str,VAL_DT),"pert"), "seconds":sum(v["seconds"] for v in s.values())}


def guards(row,base):
    return all(row[k]<=base[k]+ETOL for k in ("four","eight"))


def fit(net,g,initial,u,speeds,sets,base,seed,arm):
    net.dec["1"].load_state_dict(initial)
    net.requires_grad_(False)
    dec=net.dec["1"].requires_grad_(True)
    opt=torch.optim.AdamW(dec.parameters(),lr=5e-5,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(seed+45800)
    best=initial; metric=base["unseen"]; step=0; log={0:base}; train_seconds=0.
    wall=sync()
    for i in range(320):
        t=sync()
        net.train()
        ids=torch.randint(256,(8,),generator=rng).cuda()
        length=(10,15)[i%2]
        start=torch.randint(36-length-2,(8,),generator=rng).cuda()
        times=start[:,None]+torch.arange(length,device="cuda")
        h=u[ids[:,None],times]
        ub=speeds[ids,None]
        scale=((h-h[:,-1:]).square().mean((1,2)).sqrt()/ub[:,0]).clamp_min(.01)
        loss=0.
        for j in range(3):
            new=centered_next(net,h,increments=True)
            gt=u[ids,times[:,-1]+1+j]
            loss=loss+(((new-gt)/ub)/scale[:,None]).square().mean()/3
            h=torch.cat((h,new[:,None]),1)[:,-CONTEXT:]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        train_seconds+=sync()-t
        if i+1 in CHECKS:
            v=validation(net,g,sets); log[i+1]=v
            if guards(v,base) and v["unseen"]<metric-ETOL:
                best=cpu_state(dec); metric=v["unseen"]; step=i+1
            print(g,seed,arm,i+1,"val",round(v["unseen"],4),"four/eight",round(v["four"],4),round(v["eight"],4),"selected",step,flush=True)
    return best,dict(seed=seed,arm=arm,selected=step,validation=log[step],curve=log,fit_seconds=sync()-wall,
                     optimizer_seconds=train_seconds)


def run(net,g):
    off=int(g=="channel")
    path=FINAL_DIR/f"{g}_17_increment_rel_long.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==START[g]
    initial=torch.load(path,weights_only=True,map_location="cpu")
    dec=net.dec["1"];dec.load_state_dict(initial)
    t=sync()
    banks,speeds,dt=bank(g)
    val=fixed(g,16,164118+off,VAL_DT,[.012*i for i in range(1,6)])
    val["four"]=make_set(g,sample_cases(g,16,178761+off,"validation"))
    val["eight"]=make_set(g,cases_mixed(g,32,178812+off)[::2])
    base=validation(net,g,val)
    print(g,"baseline validation",base,"setup",sync()-t,flush=True)
    rows={}; weights={}
    for i,seed in enumerate(SEEDS):
        for arm in (("original_dt","varied_dt") if i==0 else ("varied_dt","original_dt")):
            w,r=fit(net,g,initial,banks[arm],speeds,val,base,seed,arm)
            rows[(seed,arm)]=r;weights[(seed,arm)]=w
    # Selection uses validation only. Both training seeds must show the same advantage.
    wins={s:rows[(s,"varied_dt")]["validation"]["unseen"]+ETOL <
            min(base["unseen"],rows[(s,"original_dt")]["validation"]["unseen"]) for s in SEEDS}
    choice={a:min(SEEDS,key=lambda s: rows[(s,a)]["validation"]["unseen"]) for a in ("original_dt","varied_dt")}
    report=dict(baseline_sha256=START[g],changed_dt=dt,validation_baseline=base,training=list(rows.values()),
                seed_wins=wins,selected_seeds=choice,primary_test_dt=TEST_DT,seen_test_dt=SEEN_DT,
                forecast_duration=.060,forecast_origin=".135 + profile t0",test={})
    # Always measure current model sensitivity on untouched profiles. Candidate confirmation
    # is opened only when the prior validation rule passes.
    test=fixed(g,48,187517+off,TEST_DT+SEEN_DT,[.015,.030,.045,.060])
    test["four"]=make_set(g,sample_cases(g,48,3110+off,"test"))
    test["eight"]=make_set(g,original_test_cases(g))
    fresh=cases_mixed(g,96,872941+off)[::2]
    test["fresh_matched"]=make_set(g,fresh)
    test["legacy_new_dt"]=make_set(g,[dict(c,dt=(.0016,.00375,.0125)[i%3]) for i,c in enumerate(fresh)])
    for arm in (("current","original_dt","varied_dt") if all(wins.values()) else ("current",)):
        dec.load_state_dict(initial if arm=="current" else weights[(choice[arm],arm)])
        report["test"][arm]=eval_sets(net,g,test)
        print(g,arm,"fixed unseen",mean_metric(report["test"][arm],map(str,TEST_DT)),"seen",mean_metric(report["test"][arm],map(str,SEEN_DT)),flush=True)
    promote=False
    if "varied_dt" in report["test"]:
        b,c,e=(report["test"][k] for k in ("current","original_dt","varied_dt"))
        promote=(mean_metric(e,map(str,TEST_DT))+ETOL<min(mean_metric(b,map(str,TEST_DT)), mean_metric(c,map(str,TEST_DT)))
                 and all(e[k]["field"]<=b[k]["field"]+ETOL for k in ("four","eight","fresh_matched")))
    report["confirmed"]=promote
    if promote:
        p=DEST/f"{g}_interval_decoder.pt"
        torch.save(weights[(choice["varied_dt"],"varied_dt")],p)
        report["decoder"]=p.name
        report["decoder_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
    (DEST/f"{g}.json").write_text(json.dumps(report,allow_nan=False),encoding="utf8")
    print(g,"validation wins",wins,"confirmed",promote,flush=True)


def export_table():
    """Values are trajectory means, then equally weighted over the specified intervals."""
    values=[]; train_rows=[]
    for g in GEOMETRIES:
        rep=json.loads((DEST/f"{g}.json").read_text(encoding="utf8"))
        for r in rep["training"]:
            v=r["validation"]
            train_rows.append((g,r["seed"],r["arm"],r["selected"],v["unseen"],v["unseen_pert"],
                               v["four"],v["eight"],r["optimizer_seconds"],r["fit_seconds"]))
        for arm,suites in rep["test"].items():
            tr=next((r for r in rep["training"] if r["arm"]==arm and r["seed"]==rep["selected_seeds"].get(arm)),None)
            for name,v in suites.items():
                values.append((g,arm,name,v["n"],v["steps"],v["field"],v["pert"],v["seconds"],
                               tr["fit_seconds"] if tr else 0))
            for name,ds in (("seen_fixed_mean",SEEN_DT),("unseen_fixed_mean",TEST_DT)):
                subs=[suites[str(dt)] for dt in ds]
                values.append((g,arm,name,subs[0]["n"],"",st.mean(x["field"] for x in subs),
                               st.mean(x["pert"] for x in subs),sum(x["seconds"] for x in subs),tr["fit_seconds"] if tr else 0))
    with (DEST/"test_metrics.csv").open("w",encoding="utf-8-sig",newline="") as f:
        c=csv.writer(f);c.writerow(("geometry","model","suite_or_dt","profiles","forecast_steps","field_pct",
                                    "disturbance_pct","rollout_seconds","fit_and_validation_seconds"));c.writerows(values)
    with (DEST/"training_metrics.csv").open("w",encoding="utf-8-sig",newline="") as f:
        c=csv.writer(f);c.writerow(("geometry","seed","arm","selected_step","unseen_field_pct","unseen_disturbance_pct",
                                    "original_four_pct","original_eight_pct","optimizer_seconds","fit_and_validation_seconds"))
        c.writerows(train_rows)


if __name__=="__main__":
    DEST.mkdir(exist_ok=True,parents=True)
    model=load_base(device="cuda")
    for geometry in GEOMETRIES: run(model,geometry)
    export_table()
