"""Two-stage temporal correction, using frozen project OmniArch Base weights."""
import os
for variable in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[variable]="1"
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,solve
from scipy.special import j0,jn_zeros
import torch

HERE=Path(__file__).resolve().parent
PROJECT=HERE
HEADS = HERE / "artifacts" / "heads"
OLD=PROJECT/"specialization_20261107"
sys.path.insert(0,str(OLD))
import specialize as old
from omniarch import OmniArchPredictor

N=128
SEEDS=(17,29,43)
GEOS=("pipe","channel")
KNOTS=np.array([.001,.0025,.005,.0075,.01,.015,.02,.025])
LAMS=(1e-8,1e-5,1e-3,.1,1.)
REF = {g: {"train": old.make_cases(g, "train", 256, 100+i)} for i,g in enumerate(GEOS)}
torch.set_num_threads(4)



def cuda(a):return torch.as_tensor(np.ascontiguousarray(a),dtype=torch.float32,device="cuda")


def weight(g):return old.old.weights(g,old.old.grid(g,N))


def norm(v,w):return np.sqrt(np.sum(v*v*w,axis=-1))


def steady(cc,g):return np.asarray([c["Ub"] for c in cc])[:,None]*old.old.steady_fd(g,N)


def truth(cc,n):
    result=[]
    for c in cc:
        coeff=c["coefficients"]
        u=old.trajectory(dict(c,coefficients=coeff[:6]),N,n)
        if len(coeff)>6:
            g=c["geometry"];x=old.old.grid(g,N);t=c["t0"]+c["dt"]*np.arange(n)
            for k in range(6,len(coeff)):
                lam=jn_zeros(0,len(coeff))[k] if g=="pipe" else (k+1)*np.pi/2
                phi=j0(lam*x) if g=="pipe" else np.sin(lam*(x+1))
                u+=c["Ub"]*coeff[k]*np.exp(-lam**2*t[:,None])*phi
            u[:,-1]=0
            if g=="channel":u[:,0]=0
        result.append(u)
    return np.stack(result)


def make(g,kind,n,seed):
    extended=kind!="in"
    cc=old.make_cases(g,kind,n,seed,extended)
    random=np.random.default_rng(seed+7013)
    for i,c in enumerate(cc):
        if kind=="aug":
            c["dt"]=float(KNOTS[i%len(KNOTS)])
            if (i//len(KNOTS))%2==0:c["t0"]=float(random.uniform(0,.001))
        elif kind=="mid":c["dt"]=(.00375,.0125,.0225)[i%3]
        elif kind=="outside":c["dt"]=(.00075,.030)[i%2]
        elif kind=="shape":
            c["dt"]=(.001,.0025)[i%2];c["t0"]=float(random.uniform(0,.001))
            c["coefficients"]+=random.uniform([-.04,-.025],[.04,.025]).tolist()
    return cc


def base():
    p=OmniArchPredictor(str(old.checkpoint_path()),"cuda")
    p.amp=False;p.model.requires_grad_(False).eval()
    assert p.info["step"]==72000
    return p


@torch.no_grad()
def features(p,a,h,dt,eq):
    tok,skip,mean,std=old.frozen_features(p.model,cuda(h-eq[:,None])[:,:,None])
    tok=tok+a.condition(torch.log(cuda(dt)/.01)[:,None])
    return tok.cpu().numpy(),(a.decoder(tok,skip)*std[:,None]).cpu().numpy()


def rate(h,w):
    first=h[:,-2]-h[:,-3];last=h[:,-1]-h[:,-2]
    den=(first*first*w).sum(1)
    return np.clip(np.divide((first*last*w).sum(1),den,out=np.zeros_like(den),where=den>0),0,1)


def time_features(d,dt,r):
    t=dt[:,None]/.01
    return np.concatenate([(1-r[:,None])*d,t*d,t*t*d],1)


def nn_features(token,field,mu,V,amp,dt,r,scale):
    q=(token.astype(mu.dtype)-mu)@V;t=dt[:,None]/.01
    return amp[:,None]*np.concatenate([q,t*q,(1-r[:,None])*q,np.tanh(field/scale[:,None]),
                np.ones_like(t),t,1-r[:,None]],axis=1)


def ridge(x,y,lams):
    rms=np.sqrt(np.mean(x*x,axis=0));rms=np.maximum(rms,max(1e-14,rms.max()*1e-3))
    z=x/rms;gram=z.T@z/len(x);rhs=z.T@y/len(x)
    return [solve(gram+v*np.eye(len(gram)),rhs,assume_a="pos")/rms[:,None] for v in lams]


def bank(p,a,cc,g):
    u=truth(cc,45).astype("f4").astype(float);dt=np.array([c["dt"] for c in cc]);eq=steady(cc,g);w=weight(g)
    rows=[];aux=[]
    for start in (0,5):
        initial=u[:,start:start+10];s=norm(initial[:,-1],w);r=rate(initial,w)
        for offset in (0,2,6,12,19):
            n=start+offset;h=u[:,n:n+10];tok=[];field=[]
            for j in range(0,len(cc),48):
                sl=slice(j,j+48);x,f=features(p,a,h[sl],dt[sl],eq[sl]);tok.append(x);field.append(f)
            d=(h[:,-1]-h[:,-2])/s[:,None];target=(u[:,n+10]-h[:,-1]-r[:,None]*(h[:,-1]-h[:,-2]))/s[:,None]
            rows.append((d,target,dt,r,s,np.concatenate(tok),np.concatenate(field)))
    for n in (1,2,4):
        s=norm(u[:,n],w);d=(u[:,n]-u[:,n-1])/s[:,None]
        r=np.ones(len(cc)) if n==1 else rate(u[:,n-2:n+1],w)
        target=(u[:,n+1]-u[:,n]-r[:,None]*(u[:,n]-u[:,n-1]))/s[:,None]
        aux.append((d,target,dt,r))
    return [np.concatenate(v) for v in zip(*rows)],[np.concatenate(v) for v in zip(*aux)]


def prepare_nn(rows,w):
    d,y,dt,r,s,tok,f=rows
    tok=tok.astype(float);mu=tok.mean(0);cov=(tok-mu).T@(tok-mu)
    _,V=eigh(cov,subset_by_index=[cov.shape[0]-24,cov.shape[0]-1])
    return nn_features(tok,f,mu,V,norm(d,w),dt,r,s),mu,V


def local_heads(d,y,dt,r,w,penalty):
    ps=[];bs=[]
    for k in KNOTS:
        ix=dt==k;x=d[ix];target=y[ix]+r[ix,None]*x
        # Equalize small and large resolved increments, limiting amplification of FP32 roundoff.
        scales=np.maximum(norm(x,w),1e-4);x=x/scales[:,None];target=target/scales[:,None]
        gram=x.T@x;rhs=x.T@target
        reg=np.eye(N)*np.trace(gram)*penalty
        mats=solve(gram+reg,np.concatenate([gram,rhs],axis=1),assume_a="pos")
        ps.append(mats[:,:N]);bs.append(mats[:,N:])
    return np.array(ps),np.array(bs)


def interpolate(dt):
    exact=np.flatnonzero(KNOTS==dt)
    if len(exact):return [(int(exact[0]),1.)]
    if dt<KNOTS[0]:ids=np.arange(2)
    elif dt>KNOTS[-1]:ids=np.arange(len(KNOTS)-2,len(KNOTS))
    else:
        left=np.clip(np.searchsorted(KNOTS,dt)-2,0,len(KNOTS)-4)
        ids=np.arange(left,left+4)
    terms=[]
    for i in ids:
        others=ids[ids!=i]
        terms.append((i,float(np.prod((dt-KNOTS[others])/(KNOTS[i]-KNOTS[others])))))
    return terms


def plans(head,dt):
    if head["kind"]=="global":return None
    ops=[]
    for k in np.unique(dt):
        factors=interpolate(k)
        ops.append((np.flatnonzero(dt==k),sum(c*head["P"][i] for i,c in factors),sum(c*head["B"][i] for i,c in factors)))
    return ops


def predict_time(head,ops,d,dt,r):
    if ops is None:return time_features(d,dt,r)@head["W"]
    out=np.zeros_like(d)
    for ix,P,B in ops:out[ix]=d[ix]@B-r[ix,None]*(d[ix]@P)
    return out


def prepare(cc,g):
    u=truth(cc,30);w=weight(g);eq=steady(cc,g);dt=np.array([c["dt"] for c in cc])
    return dict(u=u,eq=eq,dt=dt,w=w,cc=cc,g=g)


def stats(pred,data):
    u=data["u"][:,10:];w=data["w"];eq=data["eq"]
    err=np.sum((pred-u)**2*w,axis=(1,2))
    def calculate(ref):
        return (100*np.sqrt(err/np.sum(ref*ref*w,axis=(1,2)))).tolist()
    return dict(total=calculate(u),pert=calculate(u-eq[:,None]))


def forecast(p,a,head,history,dt,eq,g,steps=20,precision="float64"):
    """Forecast from exactly ten observed frames, time steps and forced steady profiles."""
    dtype=np.dtype(precision)
    assert dtype in (np.dtype("float32"),np.dtype("float64"))
    w=weight(g).astype(dtype);h=np.asarray(history,dtype="f4").astype(dtype)
    assert h.shape==(len(dt),10,N) and eq.shape==(len(dt),N)
    ops=plans(head,dt);dt=np.asarray(dt,dtype=dtype);eq=eq.astype(dtype,copy=False)
    if dtype==np.dtype("float32"):
        head={k:(v.astype(dtype) if isinstance(v,np.ndarray) else v) for k,v in head.items()}
        if ops is not None:ops=[(ix,P.astype(dtype),B.astype(dtype)) for ix,P,B in ops]
    r=rate(h,w);scale=norm(h[:,-1],w);out=[]
    for _ in range(steps):
        diff=h[:,-1]-h[:,-2];d=diff/scale[:,None]
        y=h[:,-1]+r[:,None]*diff
        y+=scale[:,None]*predict_time(head,ops,d,dt,r)
        if "Wn" in head and head["gain"]!=0:
            tok,f=features(p,a,h,dt,eq)
            x=nn_features(tok,f,head["mu"],head["V"],norm(d,w),dt,r,scale)
            y+=scale[:,None]*((x@head["Wn"])*head["gain"])
        y[:,-1]=0
        if g=="channel":y[:,0]=0
        if head.get("cap",True):
            n=norm(y-eq,w);before=norm(h[:,-1]-eq,w);act=n>before
            y[act]=eq[act]+(y[act]-eq[act])*(before[act]/n[act])[:,None]
        if not np.isfinite(y).all() or np.any(abs(y)>scale[:,None]*1e6):raise ArithmeticError("unstable")
        assert y.dtype==dtype
        out.append(y);h=np.concatenate([h[:,1:],y[:,None]],1)
    return np.stack(out,1)


def run(p,a,head,data):
    pred=forecast(p,a,head,data["u"][:,:10],data["dt"],data["eq"],data["g"])
    return stats(pred,data)


def means(row):return {k:float(np.mean(v)) for k,v in row.items()}


def previous(p,a,g,rows,val):
    d,y,dt,r,s,tok,f=rows;w=weight(g)
    base_heads=[dict(kind="global",W=coef,cap=False) for coef in ridge(time_features(d,dt,r),y,LAMS)]
    def select(candidates):
        errs=[]
        for head in candidates:
            try:errs.append(means(run(p,a,head,val))["total"])
            except ArithmeticError:errs.append(1e100)
        return candidates[np.argmin(errs)]
    best=select(base_heads)
    x,mu,V=prepare_nn(rows,w)
    residual=y-predict_time(best,None,d,dt,r)
    heads=[dict(best,Wn=coef,gain=1.,mu=mu,V=V) for coef in ridge(x,residual,LAMS)]
    return dict(select(heads),cap=True)

GUARD = {'pipe': {'low': True, 'high': True}, 'channel': {'low': True, 'high': False}}
CONFIG = {('pipe', 17): [1e-12, 0.1, 1.0], ('pipe', 29): [1e-12, 0.1, 1.0], ('pipe', 43): [1e-12, 0.1, 1.0], ('channel', 17): [1e-12, 0.1, 0.25], ('channel', 29): [1e-12, 0.1, 0.25], ('channel', 43): [1e-12, 0.1, 0.25]}

def rebuild(p,g,seed):
    """Rebuild the validation-selected correction head from original training trajectories."""
    HEADS.mkdir(parents=True, exist_ok=True)
    reg,net_reg,gain=CONFIG[g,seed]
    adapter=old.get_adapter(p,g,"plain",seed)
    aug=make(g,"aug",768,850101+int(g=="channel"))
    old_rows,old_early=bank(p,adapter,REF[g]["train"],g)
    new_rows,new_early=bank(p,adapter,aug,g)
    if not all(GUARD[g].values()):
        val=old.make_cases(g,"select",48,718301+int(g=="channel"))
        legacy=previous(p,adapter,g,old_rows,prepare(val,g))
        np.savez(HEADS/f"legacy_{g}_{seed}.npz",**legacy)
    rows=[np.concatenate([a,b]) for a,b in zip(old_rows,new_rows)]
    early=[np.concatenate([a,b]) for a,b in zip(old_early,new_early)]
    del old_rows,old_early,new_rows,new_early
    d,y,dt,r=[np.concatenate([a,b]) for a,b in zip(rows[:4],early)]
    w=weight(g);P,B=local_heads(d,y,dt,r,w,reg)
    head=dict(kind="local",P=P,B=B,cap=True,gain=gain)
    if gain:
        x,mu,V=prepare_nn(rows,w)
        d,y,dt,r=rows[:4]
        remaining=y-predict_time(head,plans(head,dt),d,dt,r)
        head.update(mu=mu,V=V,Wn=ridge(x,remaining,[net_reg])[0])
    np.savez(HEADS/f"accepted_{g}_{seed}.npz",**head)
    return head

def predict_profiles(p,geometry,seed,history,dt,steady_velocity,precision="float64"):
    """history [batch,10,128], dt [batch], steady_velocity [batch,128]."""
    history=np.asarray(history);dt=np.asarray(dt);steady_velocity=np.asarray(steady_velocity)
    keep=((dt<KNOTS[0]) & (not GUARD[geometry]["low"])) | ((dt>KNOTS[-1]) & (not GUARD[geometry]["high"]))
    adapter=old.get_adapter(p,geometry,"plain",seed)
    out=np.empty((len(dt),20,N),dtype=precision)
    for mask,name in ((keep,"legacy"),(~keep,"accepted")):
        if mask.any():
            with np.load(HEADS/f"{name}_{geometry}_{seed}.npz") as saved:
                head={k:(saved[k].item() if saved[k].ndim==0 else saved[k].copy()) for k in saved}
            out[mask]=forecast(p,adapter,head,history[mask],dt[mask],steady_velocity[mask],geometry,precision=precision)
    return out

if __name__=="__main__":
    predictor=base()
    for geometry in GEOS:
        for seed in SEEDS:
            rebuild(predictor,geometry,seed)
