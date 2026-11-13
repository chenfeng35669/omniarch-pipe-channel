from integrated import base,fit_physics_layer,predict_pipeline,STEPS
import numpy as np
import json
import hashlib
from pathlib import Path

R=base.PROJECT / "results" / "ae_suites"
R.mkdir(parents=True, exist_ok=True)
SUITES=("original","six_nodes","middle","low","high","early_six","early_eight")
def cases(g):
    offset=int(g=="channel")
    early=base.make(g,"covered",48,439117+offset)
    gen=np.random.default_rng(439137+offset)
    for i,c in enumerate(early):
        c.update(t0=float(gen.uniform(0,.001)),dt=(.001,.0025)[i%2])
    eight=[dict(c,coefficients=c["coefficients"]+gen.uniform([-.04,-.025],[.04,.025]).tolist()) for c in early]
    low=base.make(g,"covered",48,439151+offset)
    high=[]
    for i,c in enumerate(low):
        c["dt"]=(.00065,.00075,.00090)[i%3]
        high.append(dict(c,dt=(.028,.030,.032)[i%3]))
    nodes=base.make(g,"covered",48,973451+offset)
    early_rng=np.random.default_rng(973471+offset)
    for i,c in enumerate(nodes):
        c.update(t0=float(early_rng.uniform(0,.001)),dt=float(base.KNOTS[i%8]))
    middle=[dict(c,dt=(.00375,.0125,.0225)[(i//8)%3]) for i,c in enumerate(nodes)]
    return dict(original=base.make(g,"in",48,439101+offset),six_nodes=nodes,middle=middle,
                low=low,high=high,early_six=early,early_eight=eight)

def digest(c):
    return c["Ub"],c["nu"],c["a"]

def validate_cases(g,z):
    j=int(g=="channel")
    sets=[base.REF[g]["train"],base.make(g,"aug",768,850101+j)]
    for k in (161301,161501): # the original E validation/test cases
        sets.extend(base.make(g,kind,48,k+j) for kind in ("in","covered"))
    old_ids=set(map(digest,sum(sets,[])))
    for name,cc in z.items():
        ids=set(map(digest,cc));assert len(ids)==48 and not ids.intersection(old_ids)
    for i in range(48):
        s=z["early_six"][i];e=z["early_eight"][i]
        assert s["coefficients"]==e["coefficients"][:6]
        assert all(s[k]==e[k] for k in s if k!="coefficients")
    assert not set(STEPS).intersection(c["dt"] for name in ("low","high") for c in z[name])
def save(name,z):
    f=R/name;tmp=f.with_suffix(".partial")
    tmp.write_text(json.dumps(z,allow_nan=False),encoding="utf-8");tmp.replace(f)

def run():
    p=base.base()
    for g in base.GEOS:
        z=cases(g);validate_cases(g,z);fit=fit_physics_layer(g);records={}
        raw={}
        for name in SUITES:
            data=base.prepare(z[name],g);raw[name]=data
            u=data["u"];h=u[:,:10].astype("f4");dt=data["dt"];eq=data["eq"];w=data["w"]
            numerical=np.stack([base.old.old.forecast_fd(g,profile[-1],step,20,c["Ub"]) for profile,step,c in zip(h,dt,z[name])])
            result=dict(FD=base.stats(numerical,data))
            for s in base.SEEDS:
                a,_=predict_pipeline(p,g,s,h,dt,eq,variant="A")
                e,flags=predict_pipeline(p,g,s,h,dt,eq,fit,"E")
                if not flags.any(): assert np.array_equal(a,e)
                if flags.all() and s!=17: assert np.array_equal(last_e,e)
                last_e=e
                energy=base.norm(np.concatenate([h[:,-1:],e],axis=1)-eq[:,None],w)**2
                assert np.diff(energy,axis=1).max()<1e-15
                assert np.all(e[:,:,-1]==0)
                if g=="channel":assert np.all(e[:,:,0]==0)
                result[str(s)]={"A":base.stats(a,data),"E":base.stats(e,data)}
                if "flags" in result:assert result["flags"]==flags.tolist()
                result["flags"]=flags.tolist()
            records[name]=result
            print(g,name,int(flags.sum()),[(m,base.means(result["17"][m])) for m in ("A","E")],flush=True)
        # Real mixed-batch interface: both paths in one call, independently check assignments.
        names=("original","high","early_eight","middle");cc=[raw[k]["cc"][i] for i,k in enumerate(names)]
        mix=base.prepare(cc,g);h=mix["u"][:,:10]
        y,flags=predict_pipeline(p,g,17,h,mix["dt"],mix["eq"],fit)
        assert flags.tolist()==[False,True,True,False]
        ix=~flags
        ya=base.predict_profiles(p,g,17,h[ix],mix["dt"][ix],mix["eq"][ix])
        assert np.array_equal(y[ix],ya)
        for j in np.flatnonzero(flags):
            dt=mix["dt"][j];from numpy.polynomial.chebyshev import chebval
            loglo,loghi=np.log(STEPS[[0,-1]])
            t=(2*np.log(dt)-loglo-loghi)/(loghi-loglo)
            mat=fit["projection"]@chebval(t,fit["coefficients"])
            eq=mix["eq"][j];prev=h[j,-1].astype("f4").astype(float)
            for k in range(20):
                new=prev+(prev-eq)@mat;new[-1]=0
                if g=="channel":new[0]=0
                n,b=base.norm(new-eq,mix["w"]),base.norm(prev-eq,mix["w"])
                if n>b:new=eq+(new-eq)*b/n
                assert np.max(np.abs(new-y[j,k]))<1e-12
                prev=new
        save(f"test_{g}.json",records)
    sha=hashlib.sha256()
    with (base.old.checkpoint_path()).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):sha.update(block)
    assert sha.hexdigest()==base.old.CFG["base_checkpoint_sha256"]
    save("verified.json",dict(mixed_batch=True,weight_hash=sha.hexdigest()))
if __name__=="__main__":run()
