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
