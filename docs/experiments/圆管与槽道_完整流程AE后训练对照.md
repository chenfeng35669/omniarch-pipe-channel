# 圆管与槽道：A 与 E 的完整流程对照

2026 年 11 月 13 日。A 为上轮已验证的完整网格纠偏方案，保留槽道大步长旧版回退；E 为 A 加入物理后训练后的统一推理入口。本轮固定路由与权重选择，实际从同一历史观测分别运行 A、E，有限差分由相同 FP32 末帧启动。大步长组的圆管扰动误差从 **10.289%** 降到 **0.002969%**，槽道从 **0.1794%** 降到 **0.003342%**。

## 模块训练与接入

冻结 A。每种几何重新用相同训练种子生成 256 个由局部平滑函数叠加而成的扰动，圆管满足轴心对称，实体壁面为零。对 16 个无量纲时间步（0.00055–0.034），用 128 节点离散扩散方程、每个时间步 24 个 CN 子步生成 4,096 对参考剖面。对探针的空间主方向与对数步长的八次切比雪夫基做最小二乘，得到单步扰动更新矩阵。推理时只应用训练好的矩阵，不重算 24 个子步。

用 A 的六模态训练轨迹在不同时刻的剖面得到加权六维空间基。入口根据首个观测帧和已知步长判断：区间外步长或相对六维投影残差大于 0.03% 的样本走后训练路径，其他样本走完整 A。物理更新层和 A 都满足壁面条件和扰动能量上界。触发后原网络混合权重固定为先前验证选中的 0。路径选择只读取可用于实际推理的输入，同一批次可混合两类轨迹。[此前的模块验证与系数选择](圆管与槽道_外推误差与物理后训练验证.md)

## 测试设计

每种几何和每个工况各 48 条、128 节点、10 帧观测滚动预测 20 帧。原四阶用训练过的 3 个步长（0.005、0.01、0.02）。六阶覆盖步长与中间步长使用 A–D 消融的同一批物理参数与初始系数，改变步长后，观测和预测所覆盖的物理时长也随之改变。小步长为 0.00065、0.00075、0.00090，大步长为 0.028、0.030、0.032，两组保留相同的六阶初始条件。早期六阶与八阶组时间步均取 0.001 或 0.0025，起始时刻在 0–0.001，保持前六阶和全部物理参数逐条相同，只给八阶组增加第 7、8 阶。高、低步长没有直接用于新层拟合；它们在新层训练范围之内。模态测试考察未作为标签直接提供的解析形状，物理层已用更广的空间探针训练。

主表先对每条轨迹计算 20 步加权相对 $L^2$，再等权平均轨迹及原适配器种子 17、29、43。整场和扰动的分母分别取真实速度、真实速度减稳态。在触发组，三个适配器种子均调用同一个新层，不能视为三次独立后训练。

## 整场速度误差

**圆管**

| 工况 | A | E | 有限差分 | 启用新层 |
|---|---:|---:|---:|---:|
| 原四阶 | 0.0000064% | 0.0000064% | 0.000728% | 0/48 |
| 六阶·覆盖步长 | 0.0000082% | 0.0000082% | 0.001096% | 0/48 |
| 六阶·中间步长 | 0.0000106% | 0.0000106% | 0.001244% | 0/48 |
| 大步长 | 0.1037% | 0.0000327% | 0.002153% | 48/48 |
| 早期六阶 | 0.0000117% | 0.0000117% | 0.000297% | 0/48 |
| 同轨迹加第 7、8 阶 | 0.004120% | 0.000338% | 0.000297% | 48/48 |
| 小步长 | 0.002633% | 0.000941% | 0.0000799% | 48/48 |

**平行板槽道**

| 工况 | A | E | 有限差分 | 启用新层 |
|---|---:|---:|---:|---:|
| 原四阶 | 0.0000081% | 0.0000081% | 0.000753% | 0/48 |
| 六阶·覆盖步长 | 0.0000091% | 0.0000091% | 0.001176% | 0/48 |
| 六阶·中间步长 | 0.0000112% | 0.0000112% | 0.001274% | 0/48 |
| 大步长 | 0.0118% | 0.000214% | 0.001927% | 48/48 |
| 早期六阶 | 0.0000129% | 0.0000129% | 0.000799% | 0/48 |
| 同轨迹加第 7、8 阶 | 0.4310% | 0.001163% | 0.000823% | 48/48 |
| 小步长 | 0.003017% | 0.001540% | 0.000586% | 48/48 |

## 扰动相对误差

**圆管**

| 工况 | A | E | 有限差分 | 启用新层 |
|---|---:|---:|---:|---:|
| 原四阶 | 0.000350% | 0.000350% | 0.0361% | 0/48 |
| 六阶·覆盖步长 | 0.000352% | 0.000352% | 0.0451% | 0/48 |
| 六阶·中间步长 | 0.000660% | 0.000660% | 0.0486% | 0/48 |
| 大步长 | 10.289% | 0.002969% | 0.1901% | 48/48 |
| 早期六阶 | 0.000134% | 0.000134% | 0.003715% | 0/48 |
| 同轨迹加第 7、8 阶 | 0.0451% | 0.003376% | 0.003718% | 48/48 |
| 小步长 | 0.0253% | 0.008734% | 0.000729% | 48/48 |

**平行板槽道**

| 工况 | A | E | 有限差分 | 启用新层 |
|---|---:|---:|---:|---:|
| 原四阶 | 0.000200% | 0.000200% | 0.0214% | 0/48 |
| 六阶·覆盖步长 | 0.0000812% | 0.0000812% | 0.0110% | 0/48 |
| 六阶·中间步长 | 0.000122% | 0.000122% | 0.0186% | 0/48 |
| 大步长 | 0.1794% | 0.003342% | 0.0425% | 48/48 |
| 早期六阶 | 0.0000657% | 0.0000657% | 0.003441% | 0/48 |
| 同轨迹加第 7、8 阶 | 1.709% | 0.005349% | 0.003535% | 48/48 |
| 小步长 | 0.0155% | 0.007829% | 0.002981% | 48/48 |

## 配对结果和适用范围

先按每条轨迹对三个种子取平均，再对 48 条轨迹成对重采样 8,000 次得到下列百分位区间，差值为 E 减 A，单位为百分点。胜出数严格按 E 误差较小计。

| 几何 | 工况 | 指标 | E − A | 95% 配对区间 | E 更低 |
|---|---:|---:|---:|---:|---:|
| 圆管 | 大步长 | 整场 | -0.1036911 | [-0.130568, -0.0780094] | 48/48 |
| 圆管 | 同轨迹加第 7、8 阶 | 整场 | -0.003781361 | [-0.00547633, -0.00226745] | 24/48 |
| 圆管 | 小步长 | 整场 | -0.001692498 | [-0.00228215, -0.00117467] | 44/48 |
| 圆管 | 大步长 | 扰动 | -10.28622 | [-12.6301, -7.95659] | 48/48 |
| 圆管 | 同轨迹加第 7、8 阶 | 扰动 | -0.04168518 | [-0.0661856, -0.0209725] | 24/48 |
| 圆管 | 小步长 | 扰动 | -0.01660125 | [-0.022389, -0.0113334] | 44/48 |
| 平行板槽道 | 大步长 | 整场 | -0.01157257 | [-0.0162837, -0.00789414] | 48/48 |
| 平行板槽道 | 同轨迹加第 7、8 阶 | 整场 | -0.4297996 | [-0.549745, -0.31879] | 48/48 |
| 平行板槽道 | 小步长 | 整场 | -0.001476272 | [-0.00208009, -0.000954967] | 47/48 |
| 平行板槽道 | 大步长 | 扰动 | -0.1760707 | [-0.231649, -0.130627] | 48/48 |
| 平行板槽道 | 同轨迹加第 7、8 阶 | 扰动 | -1.703171 | [-2.46239, -1.12061] | 48/48 |
| 平行板槽道 | 小步长 | 扰动 | -0.007720504 | [-0.0120036, -0.00432598] | 47/48 |

槽道加入第 7、8 阶后的扰动误差由 1.709% 降至 0.005349%，同批有限差分为 0.003535%。在小步长组，两种几何的扰动误差也仍高于有限差分。 对照采用单个 CN 步作为有限差分参考，E 的离线参考值用了 24 个子步，精度结果不能解释为等成本优势。本试验未测试新层训练步长范围以外的通用外推能力。

在早期八阶组，圆管的平均扰动误差虽从 0.0451% 降至 0.003376%，但只有 24/48 条轨迹改善；同组槽道 48/48 条改善。

## 复核与重建

下表的 A 组复算值与 A–D 报告中的数值按原精度逐项一致。对验证代码另作混合批次检查，原路径逐点对照 A，物理路径用独立的切比雪夫求值与逐轨迹推进核对。还逐条校验了 20 步扰动能量上界、无滑移壁面、预测有限性，并对照三个训练步长的稠密分解与逐步三对角参考。原 Base 的 SHA-256 校验通过。

| 几何 | 旧试验轨迹 | 整场误差复测 | 扰动误差复测 |
|---|---:|---:|---:|
| 圆管 | 覆盖步长 | 0.0000082% | 0.000352% |
| 平行板槽道 | 覆盖步长 | 0.0000091% | 0.0000812% |
| 圆管 | 中间步长 | 0.0000106% | 0.000660% |
| 平行板槽道 | 中间步长 | 0.0000112% | 0.000122% |

六阶消融复测种子沿用 973451、973471。原四阶、早期六／八阶及低／高步长的新测试种子分别为 439101、439117、439151（槽道各加一）；物理层探针种子为 161210，槽道加一。新测试轨迹与 A 的训练轨迹及上轮物理层验证、测试轨迹检查无重复。训练标签与解码器临时权重按项目惯例清理。

**复现实验。** 在本目录建立子目录，将《三帧外推逐步纠偏与稳态对照》附录中的 Python 代码保存为该子目录下的 `base_pipeline.py`，并用项目 GPU Python 运行它，重建 A 的适配头。将下列两个代码块分别保存到同一子目录，然后执行 `python -B -X utf8 run_ae.py`。在实际使用中，每个几何只需预先调用一次 `fit_physics_layer(geometry)`，随后把拟合结果传给 `predict_pipeline(..., physics=layer, variant="E")`。[A 组权重重建代码](圆管与槽道_三帧外推逐步纠偏与稳态对照.md)


**integrated.py**

```python
"""A/E history-only forecasting on the original 128-node pipe and channel grids.

Requires base_pipeline.py from the A report in this same directory.
"""
import base_pipeline as base
import numpy as np
from scipy.linalg import eigh

STEPS = np.array([.00055,.00085,.001,.0017,.0025,.00375,.005,.0075,
                  .01,.015,.02,.025,.027,.029,.031,.034])
SUPPORT_LIMIT = .0003

def time_basis(dt):
    log_min,log_max = np.log(STEPS[[0,-1]])
    scaled = (2*np.log(dt)-log_min-log_max)/(log_max-log_min)
    return np.polynomial.chebyshev.chebvander(scaled,8)

def fit_physics_layer(geometry):
    """Fit the frozen E branch once per geometry, using no test trajectories."""
    radial = geometry=="pipe"
    offset = int(not radial)
    x = base.old.old.grid(geometry,128)
    w = base.weight(geometry)
    coordinate = x if radial else (x+1)/2
    rng = np.random.default_rng(161210+offset)
    probes = np.zeros((256,128))
    for _ in range(4):
        center = rng.uniform(0,1,(256,1))
        width = np.exp(rng.uniform(np.log(.035),np.log(.29),(256,1)))
        shape = np.exp(-.5*((coordinate-center)/width)**2)
        if radial:
            shape += np.exp(-.5*((coordinate+center)/width)**2)
        probes += (1-x*x)*rng.normal(size=(256,1))*shape
    probes /= base.norm(probes,w)[:,None]
    _,sigma,right = np.linalg.svd(probes,full_matrices=False)
    projection = right[sigma>sigma[0]*1e-6].T

    # Diagonalize the same tridiagonal spatial operator as the project's CN solver.
    lo,diag,up = base.old.old.coeffs(geometry,128,np.float64)
    scale = np.r_[1.,np.cumprod(np.sqrt(up/lo))]
    off_diag = np.sqrt(up*lo)
    eigenvectors_input = np.diag(diag)+np.diag(off_diag,1)+np.diag(off_diag,-1)
    eig,vectors = eigh(eigenvectors_input)
    active = probes[:,:-1] if radial else probes[:,1:-1]
    coordinates = (active*scale)@vectors
    labels = []
    for step in STEPS:
        decay = ((1+step*eig/48)/(1-step*eig/48))**24
        values = base.old.old.restore(geometry,((coordinates*decay)@vectors.T)/scale,128)
        labels.append(values-probes)
    labels = np.stack(labels)
    # Cross-check exact training targets with independent tridiagonal factorizations.
    for i in (0,7,15):
        predicted = probes[i]+labels[i,i]
        reference = base.old.old.forecast_fd(geometry,probes[i],STEPS[i]/24,24,0.)[-1]
        np.testing.assert_allclose(predicted,reference,rtol=2e-10,atol=2e-11)
    reduced = probes@projection
    spatial = np.linalg.lstsq(reduced,labels.transpose(1,0,2).reshape(256,-1),rcond=1e-9)[0]
    targets = spatial.reshape(reduced.shape[1],len(STEPS),128).transpose(1,0,2).reshape(len(STEPS),-1)
    coefficients = np.linalg.lstsq(time_basis(STEPS),targets,rcond=1e-12)[0].reshape(9,-1,128)

    known = base.make(geometry,"aug",768,850101+offset)
    profiles = base.truth(known,12)/np.array([c["Ub"] for c in known])[:,None,None]
    residual = (profiles[:,[0,2,6,9]]-base.old.old.steady_fd(geometry,128)).reshape(-1,128)
    _,_,basis = np.linalg.svd(residual*np.sqrt(w),full_matrices=False)
    return dict(projection=projection,coefficients=coefficients,basis=basis[:6])

def predict_pipeline(p,geometry,seed,history,dt,steady_velocity,physics=None,variant="E"):
    """Return (20 predicted frames, per-trajectory physics-route mask)."""
    history = np.asarray(history,dtype=np.float32).astype(float)
    dt = np.asarray(dt,dtype=float)
    steady = np.asarray(steady_velocity,dtype=float)
    if geometry not in base.GEOS or variant not in ("A","E"):
        raise ValueError("Unknown geometry or pipeline variant.")
    if dt.ndim!=1 or history.shape!=(len(dt),10,128) or steady.shape!=(len(dt),128) or np.any(dt<=0):
        raise ValueError("Expected ten frames per trajectory, 128 nodes, and positive steps.")
    w = base.weight(geometry)
    use_physics = np.zeros(len(dt),dtype=bool)
    if variant=="E":
        if physics is None:raise ValueError("Fit or load the geometry-specific physics layer.")
        sig = (history[:,0]-steady)*np.sqrt(w)
        basis = physics["basis"]
        ratio = np.linalg.norm(sig-(sig@basis.T)@basis,axis=1)/np.maximum(np.linalg.norm(sig,axis=1),1e-14)
        use_physics = (dt<base.KNOTS[0]) | (dt>base.KNOTS[-1]) | (ratio>SUPPORT_LIMIT)
    prediction = np.empty((len(dt),20,128))
    regular = ~use_physics
    if regular.any():
        prediction[regular] = base.predict_profiles(p,geometry,seed,history[regular],dt[regular],steady[regular])
    for step in np.unique(dt[use_physics]):
        ix = np.flatnonzero(use_physics & (dt==step))
        coeff = np.tensordot(time_basis([step])[0],physics["coefficients"],axes=1)
        update = physics["projection"]@coeff
        eq = steady[ix]
        prev = history[ix,-1].copy()
        for j in range(20):
            y = prev+(prev-eq)@update
            y[:,-1] = 0.
            if geometry=="channel":y[:,0] = 0.
            before,after = base.norm(prev-eq,w),base.norm(y-eq,w)
            over = after>before
            y[over] = eq[over]+(y[over]-eq[over])*(before[over]/after[over])[:,None]
            prediction[ix,j]=y
            prev=y
    if not np.isfinite(prediction).all():raise ArithmeticError("Nonfinite forecast")
    return prediction,use_physics
```

**run_ae.py**

```python
from integrated import base,fit_physics_layer,predict_pipeline,STEPS
import numpy as np
import json
import hashlib
from pathlib import Path

R=Path(__file__).resolve().parent
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
    with (base.PROJECT.parent/"权重/omniarch-base.pt").open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):sha.update(block)
    assert sha.hexdigest()==base.old.CFG["base_checkpoint_sha256"]
    save("verified.json",dict(mixed_batch=True,weight_hash=sha.hexdigest()))
if __name__=="__main__":run()
```
