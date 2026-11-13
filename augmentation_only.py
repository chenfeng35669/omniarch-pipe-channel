"""Precompute data variants, then run the unchanged, single-stage Base decoder training."""
import json
import statistics
import time
from copy import deepcopy

import torch

from analytic import GEOMETRIES, original_test_cases, reference, sample_cases
from base_pipeline import ROOT, load_base
from diagnose_omniarch import OUT, cases_mixed
from reproduce import evaluate
from train import fit_decoder


def make_banks(g):
    """No model is used. Reuse all 256 original profiles; add one variant per profile."""
    offset = int(g=="channel")
    original = sample_cases(g,256,100+offset,"train")
    rng = torch.Generator().manual_seed(54128+offset)
    hi = (2*torch.rand(256,4,generator=rng,dtype=torch.float64)-1)*torch.tensor([.07,.04,.04,.025],dtype=torch.float64)
    dt_order = torch.randperm(256,generator=rng).tolist()
    base_fields = reference(g,original,frames=36)[0]
    banks = {"original":(original,base_fields)}
    for variant in ("modes_only","modes_and_sampling","sampling_only"):
        augmented = []
        for i,c in enumerate(original):
            extra = hi[i].tolist() if any(c["coefficients"]) else [0.]*4
            a = dict(c,id=f"{g}_{variant}_{i}",coefficients=c["coefficients"]+extra)
            if variant=="sampling_only":
                a["coefficients"] = c["coefficients"].copy()
            if variant in ("modes_and_sampling","sampling_only"):
                a.update(dt=(.001,.0025)[dt_order[i]%2],t0=c["t0"]/30)
            augmented.append(a)
        fields = reference(g,augmented,frames=36)[0]
        banks[variant] = (original+augmented,torch.cat((base_fields,fields),0))
    return banks


def verify_pairing(seed):
    # For these power-of-two bank sizes, the original underlying row and window
    # must be identical across arms, even when the augmented copy is drawn.
    a,b = (torch.Generator().manual_seed(seed) for _ in range(2))
    for i in range(160):
        assert torch.equal(torch.randint(256,(8,),generator=a),torch.randint(512,(8,),generator=b)%256)
        limit=36-(10,15)[i%2]
        assert torch.equal(torch.randint(limit,(8,),generator=a),torch.randint(limit,(8,),generator=b))


def physical_key(c):
    return (c["Ub"],c["t0"],c["dt"],*tuple(c["coefficients"]+[0.]*(8-len(c["coefficients"]))))


def main():
    torch.set_num_threads(4)
    seeds=(17,29,43,61,83)
    prepared,tests,prep_time = {},{},{}
    for seed in seeds: verify_pairing(seed)
    for off,g in enumerate(GEOMETRIES):
        start=time.perf_counter()
        prepared[g]=make_banks(g)
        prep_time[g]=time.perf_counter()-start
        tests[g]={"original_eight":original_test_cases(g),"four":sample_cases(g,48,3110+off,"test"),
                  "fresh_eight":cases_mixed(g,96,764213+off)[::2]}
        val=sample_cases(g,12,200+off,"validation")
        held={physical_key(c) for cs in [*tests[g].values(),val] for c in cs}
        assert all(not any(physical_key(c) in held for c in rows) for rows,_ in prepared[g].values())
    net=load_base(device="cuda")
    orig=deepcopy(net.dec["1"].state_dict())
    path=OUT/"offline_augmentation_comparison.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    rows=json.loads(path.read_text(encoding="utf8"))["rows"] if path.exists() else []
    known={(r["seed"],r["geometry"],r["training_data"]) for r in rows}
    assert len(known)==len(rows)
    for seed in seeds:
        for g in GEOMETRIES:
            labels=list(prepared[g])
            # Rotate order across seeds to reduce systematic timing effects.
            k=seeds.index(seed)%len(labels)
            for label in labels[k:]+labels[:k]:
                if (seed,g,label) in known:
                    continue
                net.dec["1"].load_state_dict(orig)
                net.eval()
                torch.cuda.synchronize()
                t=time.perf_counter()
                step=fit_decoder(net,g,160,seed,training_bank=prepared[g][label])
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-t
                scores={}
                for test,cases in tests[g].items():
                    start=time.perf_counter()
                    vel,pert,calls=evaluate(net,g,cases)
                    torch.cuda.synchronize()
                    scores[test]=dict(velocity_pct=vel,disturbance_pct=pert,seconds=time.perf_counter()-start,forwards=calls)
                rows.append(dict(geometry=g,training_data=label,seed=seed,selected_update=step,
                                 training_seconds=elapsed,scores=scores))
                path.write_text(json.dumps(dict(preparation_seconds=prep_time,rows=rows),indent=2),encoding="utf8")
                print("RESULT",g,seed,label,round(elapsed,2),{k:round(v["velocity_pct"],5) for k,v in scores.items()},flush=True)
    for g in GEOMETRIES:
        for label in prepared[g]:
            arms=[r for r in rows if r["training_data"]==label and r["geometry"]==g]
            print("MEAN",g,label,{k:statistics.mean(r["scores"][k]["velocity_pct"] for r in arms) for k in tests[g]},
                  "seconds",statistics.mean(r["training_seconds"] for r in arms),flush=True)


if __name__=="__main__":
    main()
