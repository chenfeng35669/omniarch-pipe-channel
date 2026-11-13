"""Train/evaluate additive task adapters without changing original experiment files."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('MKL_NUM_THREADS','8')
import argparse,copy,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import j0,jn_zeros

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
sys.path.insert(0,str(PARENT/'vendor/omniarch_pt'))
import benchmark as old
from omniarch import OmniArchPredictor

CFG=json.loads((ROOT/'protocol.json').read_text())
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
CK=ROOT/'checkpoints'; CK.mkdir(exist_ok=True)
DEVICE=torch.device('cuda')
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
JROOTS=jn_zeros(0,6)


def js(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def checkpoint_path():
    return Path(os.environ.get("OMNIARCH_CKPT", PARENT / "weights" / "omniarch-base.pt")).expanduser()


def get_base():
    t=time.perf_counter()
    p=OmniArchPredictor(str(checkpoint_path()),'cuda')
    p.amp=False
    p.model.requires_grad_(False).eval()
    print('Base loaded FP32',p.n_params,'parameters',round(time.perf_counter()-t,3),'s',flush=True)
    return p


def make_cases(geo,split,n,seed,extrap=False):
    rng=np.random.default_rng(seed)
    limits=CFG['extrapolation_mode_amplitude_limits'] if extrap else CFG['mode_amplitude_limits']
    dtlist=CFG['extrapolation_dtau'] if extrap else CFG['training_dtau']
    cases=[]
    for i in range(n):
        ub=float(np.exp(rng.uniform(np.log(.005),np.log(.03))))
        a=float(rng.uniform(.0025,.0075));nu=float(rng.uniform(.8e-6,2e-6))
        coeff=rng.uniform(-np.asarray(limits),np.asarray(limits)).tolist()
        # Steady samples are deliberately present in training, but not the dynamic test set.
        if split=='train' and i%8==0: coeff=[0.0]*len(limits)
        cases.append(dict(id=f'{geo}_{split}_{i:03}',geometry=geo,split=split,Ub=ub,a=a,nu=nu,rho=1000.0,
                          G=(8 if geo=='pipe' else 3)*1000*nu*ub/a**2,
                          Ustar=(8 if geo=='pipe' else 3)*ub,dt=float(dtlist[i%len(dtlist)]),
                          t0=float(rng.uniform(0,.03)),coefficients=coeff))
    return cases


def trajectory(case,n,frames):
    geo=case['geometry'];x=old.grid(geo,n)
    u=np.broadcast_to((2 if geo=='pipe' else 1.5)*(1-x*x),(frames,n)).copy()
    times=case['t0']+case['dt']*np.arange(frames)
    for k,c in enumerate(case['coefficients']):
        lam=JROOTS[k] if geo=='pipe' else (k+1)*np.pi/2
        phi=j0(lam*x) if geo=='pipe' else np.sin(lam*(x+1))
        u+=c*np.exp(-lam*lam*times[:,None])*phi
    u[:,-1]=0
    if geo=='channel':u[:,0]=0
    return u*case['Ub']


def prepare():
    allcases={}
    for j,geo in enumerate(['pipe','channel']):
        allcases[geo]={
          'train':make_cases(geo,'train',256,100+j),
          'val':make_cases(geo,'val',12,200+j),
          'test':make_cases(geo,'test',16,300+j),
          'extrap':make_cases(geo,'extrap',8,400+j,True),
        }
        legacy=[]
        for name,coeff in [('steady',[0,0,0,0]),('transient',[.35,-.1,0,0] if geo=='pipe' else [.35,0,.1,0])]:
            legacy.append(dict(id=f'{geo}_legacy_{name}',geometry=geo,split='legacy',Ub=.01,a=.005,nu=1e-6,rho=1000.,
                               G=3.2 if geo=='pipe' else 1.2,Ustar=.08 if geo=='pipe' else .03,dt=.01,t0=0.,coefficients=coeff))
        allcases[geo]['legacy']=legacy
    ids=[c['id'] for g in allcases.values() for ss in g.values() for c in ss]
    assert len(ids)==len(set(ids))
    js(ROOT/'cases.json',allcases)
    print('Saved fixed train/val/test cases',len(ids),flush=True)
    return allcases


@torch.no_grad()
def frozen_features(model,x):
    """Exactly the public encoder/backbone calculation, returning the last hidden token."""
    # x is (B,T,1,N); no future values enter normalization or attention.
    B,T,C,N=x.shape
    mean,std=model.stats(x)
    xn=(x-mean)/std
    flat=xn.reshape(B*T*C,N)
    tok=model.enc['1'](flat).view(B,T,C,-1)
    ty=torch.zeros((B,C),device=x.device,dtype=torch.long)
    tok=(tok+model.type_emb(ty)[:,None]).view(B,T*C,-1)
    idx=torch.arange(T,device=x.device).repeat_interleave(C)
    mask=idx[None,:]<=idx[:,None]
    for blk in model.blocks:tok=blk(tok,idx,mask)
    h=model.norm(tok).reshape(B,T,C,-1)[:,-1,0]
    return h,xn[:,-1,0],mean[:,0,0,0],std[:,0,0,0]


def project(u,geo):
    mask=torch.ones(u.shape[-1],device=u.device,dtype=u.dtype)
    mask[-1]=0
    if geo=='channel':mask[0]=0
    return u*mask


class Adapter(nn.Module):
    def __init__(self,base_decoder):
        super().__init__()
        self.decoder=copy.deepcopy(base_decoder)
        self.condition=nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,1024))
        nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
        self.requires_grad_(True)

    def forward(self,h,skip,mean,std,dt):
        hc=h+self.condition(torch.log(dt/.01)[:,None])
        delta=self.decoder(hc,skip)
        return (skip+delta)*std[:,None]+mean[:,None]


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(13,32,5,padding=2),nn.GELU(),nn.Conv1d(32,32,5,padding=2),nn.GELU(),
                               nn.Conv1d(32,32,3,padding=1),nn.GELU(),nn.Conv1d(32,1,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,dt):
        return x[:,9]+dt[:,None]*self.net(x)[:,0]


def cnn_input(hist,dt,geo):
    B,T,N=hist.shape
    x=torch.linspace(0,1,N,device=hist.device) if geo=='pipe' else torch.linspace(-1,1,N,device=hist.device)
    dx=(x[1]-x[0]).expand(B,1,N)
    return torch.cat([hist[:,-10:],x[None,None,:].expand(B,1,N),torch.log(dt/.01)[:,None,None].expand(B,1,N),dx],1)


def lap(u,geo):
    n=u.shape[-1];h=(1 if geo=='pipe' else 2)/(n-1)
    v=(u[:,2:]-2*u[:,1:-1]+u[:,:-2])/(h*h)
    if geo=='pipe':
        r=torch.arange(1,n-1,device=u.device,dtype=u.dtype)*h
        v=v+(u[:,2:]-u[:,:-2])/(2*h*r)
        v=torch.cat([4*(u[:,1:2]-u[:,:1])/(h*h),v],-1)
    return v


def physics_loss(pred,last,dt,geo):
    # In U*=Ga^2/mu and tau=nu*t/a^2 units the constant forcing is exactly one.
    temporal=(pred-last)/dt[:,None]
    temporal=temporal[:,:-1] if geo=='pipe' else temporal[:,1:-1]
    res=temporal-lap((pred+last)*.5,geo)-1
    loss=(res*res).mean()
    if geo=='pipe':
        h=1/(pred.shape[-1]-1)
        axis=(-3*pred[:,0]+4*pred[:,1]-pred[:,2])/(2*h)
        loss=loss+axis.square().mean()
    return loss


def cache_features(p,cases,geo,mode):
    chunks={k:[] for k in ['h','skip','mean','std','dt','target','last','ref','cnn']}
    for k in CFG['training_history_lengths']:
        for start in range(0,len(cases),16):
            cc=cases[start:start+16]
            raw=[trajectory(c,128,k+1) for c in cc]
            scale=np.asarray([c['Ustar'] if mode!='plain' else 1 for c in cc])
            hist=torch.tensor(np.stack([a[:k] for a in raw])/scale[:,None,None],device=DEVICE,dtype=torch.float32)
            target=torch.tensor(np.stack([a[k] for a in raw])/scale[:,None],device=DEVICE,dtype=torch.float32)
            dt=torch.tensor([c['dt'] for c in cc],device=DEVICE)
            if mode=='small':
                h=torch.empty((len(cc),0),device=DEVICE)
                skip=hist[:,-1]
                mean=torch.zeros(len(cc),device=DEVICE)
                std=torch.ones(len(cc),device=DEVICE)
            else:
                h,skip,mean,std=frozen_features(p.model,hist[:,:,None])
            vals=dict(h=h,skip=skip,mean=mean,std=std,dt=dt,target=target,last=hist[:,-1],
                      ref=torch.tensor([c['Ub']/s for c,s in zip(cc,scale)],device=DEVICE,dtype=torch.float32),
                      cnn=cnn_input(hist,dt,geo))
            for key in chunks:chunks[key].append(vals[key].detach())
    return {k:torch.cat(v) for k,v in chunks.items()}


@torch.no_grad()
def forecast(p,adapter,cases,n,mode,bc=False,histories=None):
    geo=cases[0]['geometry']
    scale=np.asarray([c['Ustar'] if mode in ['specialized','small','norm'] else 1 for c in cases])
    initial=np.stack([trajectory(c,n,30)[:10] for c in cases]) if histories is None else np.asarray(histories)
    hist=torch.tensor(initial/scale[:,None,None],device=DEVICE,dtype=torch.float32)
    dt=torch.tensor([c['dt'] for c in cases],device=DEVICE)
    out=[]
    for _ in range(20):
        if mode=='small':pred=adapter(cnn_input(hist,dt,geo),dt)
        else:
            h,skip,mean,std=frozen_features(p.model,hist[:,-15:,None])
            if adapter is None:pred=(skip+p.model.dec['1'](h,skip))*std[:,None]+mean[:,None]
            else:pred=adapter(h,skip,mean,std,dt)
        if bc or mode in ['specialized','small']:pred=project(pred,geo)
        out.append(pred)
        hist=torch.cat([hist,pred[:,None]],1)
    return torch.stack(out,1).cpu().numpy()*scale[:,None,None]


def score(p,adapter,cases,mode):
    pred=forecast(p,adapter,cases,128,mode)
    e=[]
    for i,c in enumerate(cases):
        truth=trajectory(c,128,30)[10:]
        w=old.weights(c['geometry'],old.grid(c['geometry'],128))
        e.append(np.sqrt(np.sum((pred[i]-truth)**2*w)/np.sum(truth*truth*w)))
    return float(np.mean(e))


def train_one(p,cases,geo,mode,seed,cache):
    torch.manual_seed(seed)
    model=(SmallCNN() if mode=='small' else Adapter(p.model.dec['1'])).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=1e-4)
    gen=torch.Generator(device=DEVICE);gen.manual_seed(seed)
    best=float('inf');beststep=-1;beststate=None;log=[]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();begin=time.perf_counter();validation_seconds=0
    num=len(cache['dt'])
    for step in range(CFG['training_steps']+1):
        if step in CFG['validation_steps']:
            model.eval();torch.cuda.synchronize();t=time.perf_counter()
            value=score(p,model,cases['val'],mode)
            torch.cuda.synchronize();validation_seconds+=time.perf_counter()-t
            if value<best:
                best=value;beststep=step
                beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            log.append(dict(step=step,validation_rel_l2=value,best_validation_rel_l2=best))
            print('VAL',geo,mode,seed,step,'error',value,flush=True)
            if step==CFG['training_steps']:break
            model.train()
        idx=torch.randint(num,(CFG['batch_size'],),device=DEVICE,generator=gen)
        batch={k:v[idx] for k,v in cache.items()}
        if mode=='small':pred=model(batch['cnn'],batch['dt'])
        else:pred=model(batch['h'],batch['skip'],batch['mean'],batch['std'],batch['dt'])
        if mode!='plain':pred=project(pred,geo)
        data_loss=(((pred-batch['target'])/batch['ref'][:,None])**2).mean()
        phys=physics_loss(pred,batch['last'],batch['dt'],geo) if mode!='plain' else pred.new_tensor(0.)
        weight=CFG['physics_weight']*min(1,(step+1)/CFG['physics_ramp_steps'])
        loss=data_loss+weight*phys
        assert torch.isfinite(loss),f'nonfinite {geo} {mode} {seed} {step}'
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        assert torch.isfinite(norm)
        opt.step()
        lr=CFG['learning_rate']*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/CFG['training_steps'])))
        for group in opt.param_groups:group['lr']=lr
        if step%100==0:
            log.append(dict(step=step,data_loss=float(data_loss.detach()),physics_loss=float(phys.detach()),total_loss=float(loss.detach())))
    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
    record=dict(geometry=geo,method=mode,seed=seed,best_step=beststep,best_validation_rel_l2=best,
                trainable_parameters=sum(v.numel() for v in model.parameters()),
                train_plus_validation_seconds=elapsed,validation_seconds=validation_seconds,
                optimization_seconds=elapsed-validation_seconds,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                log=log)
    torch.save(dict(state=beststate,metadata=record),CK/f'{geo}_{mode}_seed{seed}.pt')
    js(OUT/f'train_{geo}_{mode}_{seed}.json',record)
    del opt,model;torch.cuda.empty_cache()


def validate_wrapper(p,cases):
    arr=np.stack([trajectory(c,128,30)[:10] for c in cases[:2]])
    x=torch.tensor(arr[:,:,None],device=DEVICE,dtype=torch.float32)
    with torch.no_grad():
        h,skip,mean,std=frozen_features(p.model,x)
        y=(skip+p.model.dec['1'](h,skip))*std[:,None]+mean[:,None]
        pn,(m,s)=p.model(x)
        z=(pn[:,-1:]*s+m)[:,0,0]
        err=float((y-z).abs().max())
    assert err<1e-7,err
    js(OUT/'wrapper_validation.json',dict(max_abs_difference=err,passed=True))


def train(modes=('plain','specialized','small')):
    cases=prepare()
    p=get_base()
    validate_wrapper(p,cases['pipe']['val'])
    for geo in ['pipe','channel']:
        for mode in modes:
            t=time.perf_counter();cache=cache_features(p,cases[geo]['train'],geo,mode)
            torch.cuda.synchronize();cache_sec=time.perf_counter()-t
            js(OUT/f'cache_{geo}_{mode}.json',dict(seconds=cache_sec,samples=len(cache['dt']),bytes=sum(x.numel()*x.element_size() for x in cache.values())))
            for seed in CFG['training_seeds']:
                train_one(p,cases[geo],geo,mode,seed,cache)
            del cache;torch.cuda.empty_cache()
    completed = 'training_complete.json' if len(modes) == 3 else 'plain_training_complete.json'
    js(OUT/completed,dict(complete=True,torch=torch.__version__,device=torch.cuda.get_device_name(),
                                        environment=old.environment()))
    print('ALL TRAINING COMPLETE',flush=True)


def get_adapter(p,geo,mode,seed):
    saved=torch.load(CK/f'{geo}_{mode}_seed{seed}.pt',map_location='cpu',weights_only=True)
    a=(SmallCNN() if mode=='small' else Adapter(p.model.dec['1'])).to(DEVICE)
    a.load_state_dict(saved['state']);a.eval()
    return a


def evaluate():
    assert (OUT/'training_complete.json').exists(),'Complete all training before opening test evaluation'
    cases=json.loads((ROOT/'cases.json').read_text())
    p=get_base();records=[];timings=[]
    for geo in ['pipe','channel']:
        groups=[('test',128),('extrap',128),('test',64),('test',256),('legacy',128)]
        methods=[('base','raw',None,None,False),('base_norm','norm',None,None,False),('base_bc','raw',None,None,True)]
        for seed in CFG['training_seeds']:
            methods.extend([('plain','plain','plain',seed,False),('plain_bc','plain','plain',seed,True),
                            ('specialized','specialized','specialized',seed,True),('small','small','small',seed,True)])
        for label,mode,ckmode,seed,bc in methods:
            adapter=get_adapter(p,geo,ckmode,seed) if ckmode else None
            for group,n in groups:
                cc=cases[geo][group]
                pred=forecast(p,adapter,cc,n,mode,bc)
                truths=np.stack([trajectory(c,n,30)[10:] for c in cc])
                histories=np.stack([trajectory(c,n,30)[:10].astype(np.float32) for c in cc])
                np.savez_compressed(OUT/f'pred_{geo}_{label}_{seed}_{group}_{n}.npz',prediction=pred,truth=truths,history=histories)
                for i,c in enumerate(cc):
                    m=old.metrics(geo,pred[i],truths[i],c['Ub'],histories[i,-1],c['dt'])
                    records.append(dict(geometry=geo,method=label,seed=seed,group=group,N=n,case=c['id'],**m))
                print('TEST',geo,label,seed,group,n,'meanE2',np.mean([r['rel_l2_weighted'] for r in records[-len(cc):]]),flush=True)
            js(OUT/'metrics.json',records)
            # Same nominal dynamic trajectory; latency compared separately from batched accuracy.
            if seed in [None,17] and label not in ['base_norm','plain_bc']:
                cc=[cases[geo]['legacy'][1]]
                pre_hist=np.stack([trajectory(c,128,30)[:10] for c in cc])
                tm=old.time_calls(lambda:forecast(p,adapter,cc,128,mode,bc,histories=pre_hist),sync=torch.cuda.synchronize)
                timings.append(dict(geometry=geo,method=label,seed=seed,**tm))
                js(OUT/'timings.json',timings)
            del adapter;torch.cuda.empty_cache()
        # Traditional methods and persistence see the same history and physical problem.
        for group,n in groups:
            for c in cases[geo][group]:
                traj=trajectory(c,n,30);hist=traj[:10].astype(np.float32);truth=traj[10:]
                for label in ['FD-CN','persistence']:
                    pred=old.forecast_fd(geo,hist[-1],c['dt'],20,c['Ub']) if label=='FD-CN' else np.repeat(hist[-1:],20,axis=0)
                    records.append(dict(geometry=geo,method=label,seed=None,group=group,N=n,case=c['id'],
                                        **old.metrics(geo,pred,truth,c['Ub'],hist[-1],c['dt'])))
        c=cases[geo]['legacy'][1];hist=trajectory(c,128,30)[:10].astype(np.float32)
        timings.append(dict(geometry=geo,method='FD-CN',seed=None,
            **old.time_calls(lambda:old.forecast_fd(geo,hist[-1],c['dt'],20,c['Ub']),inner=20)))
        js(OUT/'metrics.json',records);js(OUT/'timings.json',timings)
    js(OUT/'evaluation_complete.json',dict(complete=True,records=len(records),environment=old.environment()))
    print('ALL EVALUATION COMPLETE',len(records),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['train','train-plain','evaluate'])
    args=ap.parse_args()
    if args.phase=='train':train()
    elif args.phase=='train-plain':train(('plain',))
    else:evaluate()
