"""Compare early sampling, timestep-specific propagation, and the OmniArch residual."""
import argparse

import base_pipeline as base
import numpy as np
from scipy.linalg import solve

TIME_PENALTIES = (1e-12, 1e-10, 1e-8, 1e-6)
NET_PENALTIES = (1e-5, .001, .1)


def late_samples(cases, geometry):
    fields = base.truth(cases, 45).astype("f4").astype(float)
    steps = np.array([case["dt"] for case in cases])
    weights = base.weight(geometry)
    rows = []
    for n in (11, 18, 26):
        scale = base.norm(fields[:, n], weights)[:, None]
        delta = (fields[:, n] - fields[:, n-1]) / scale
        decay = base.rate(fields[:, n-2:n+1], weights)
        target = (fields[:, n+1] - fields[:, n] - decay[:, None] * (fields[:, n] - fields[:, n-1])) / scale
        rows.append((delta, target, steps, decay))
    return [np.concatenate(column) for column in zip(*rows)]


def training_data(predictor, adapter, geometry):
    original = base.REF[geometry]["train"]
    extra = base.make(geometry, "aug", 768, 850101 + int(geometry == "channel"))
    old, old_early = base.bank(predictor, adapter, original, geometry)
    new, new_early = base.bank(predictor, adapter, extra, geometry)
    return (
        [np.concatenate(pair) for pair in zip(old, new)],
        [np.concatenate(pair) for pair in zip(old_early, new_early)],
        [np.concatenate(pair) for pair in zip(late_samples(original, geometry), late_samples(extra, geometry))],
    )


def global_time_heads(delta, target, steps, decay, weights):
    scale = np.maximum(base.norm(delta, weights), 1e-4)[:, None]
    features = base.time_features(delta, steps, decay) / scale
    gram = features.T @ features
    rhs = features.T @ (target / scale)
    penalty_matrix = np.trace(gram) * np.eye(len(gram))
    return [dict(kind="global", W=solve(gram + lam * penalty_matrix, rhs, assume_a="pos"), cap=True)
            for lam in TIME_PENALTIES]


def evaluation_sets(geometry):
    offset = int(geometry == "channel")
    covered = base.make(geometry, "covered", 48, 973451 + offset)
    rng = np.random.default_rng(973471 + offset)
    for i, case in enumerate(covered):
        case.update(t0=float(rng.uniform(0, .001)), dt=float(base.KNOTS[i % 8]))
    middle = [dict(c, dt=(.00375, .0125, .0225)[(i // 8) % 3]) for i, c in enumerate(covered)]
    return {"covered": base.prepare(covered, geometry), "middle": base.prepare(middle, geometry)}


def cn_score(data):
    predictions = np.stack([
        base.old.old.forecast_fd(data["g"], u[9].astype("f4"), case["dt"], 20, case["Ub"])
        for u, case in zip(data["u"], data["cc"])
    ])
    return base.means(base.stats(predictions, data))


def fit_arms(predictor, geometry, seed, validation):
    adapter = base.old.get_adapter(predictor, geometry, "plain", seed)
    rows, early, late = training_data(predictor, adapter, geometry)
    weights = base.weight(geometry)
    network_features, center, vectors = base.prepare_nn(rows, weights)

    def rank_head(head):
        ratios = []
        for data, reference in validation:
            try:
                score = base.means(base.run(predictor, adapter, head, data))
            except ArithmeticError:
                return float("inf")
            ratios.extend(score[k] / reference[k] for k in ("total", "pert"))
        return float(np.mean(ratios))

    arms = {}
    for label in ("A", "B", "C"):
        extra = late if label == "B" else early
        delta, target, steps, decay = [np.concatenate(pair) for pair in zip(rows[:4], extra)]
        if label == "C":
            candidates = global_time_heads(delta, target, steps, decay, weights)
        else:
            candidates = []
            for lam in TIME_PENALTIES:
                proj, propagation = base.local_heads(delta, target, steps, decay, weights, lam)
                candidates.append(dict(kind="local", P=proj, B=propagation, cap=True))
        best = min(candidates, key=rank_head)
        d, y, dt, r = rows[:4]
        residual = y - base.predict_time(best, base.plans(best, dt), d, dt, r)
        net_candidates = [dict(best, gain=0.)]
        for coef in base.ridge(network_features, residual, NET_PENALTIES):
            for gain in (.25, 1.):
                net_candidates.append(dict(best, gain=gain, Wn=coef, mu=center, V=vectors))
        arms[label] = min(net_candidates, key=rank_head)
    arms["D"] = dict(arms["A"], gain=0.)
    return adapter, arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=["pipe", "channel", "both"], default="both")
    args = parser.parse_args()
    predictor = base.base()
    for geo in (base.GEOS if args.geometry == "both" else (args.geometry,)):
        validation = []
        for split, seed in (("in", 850111), ("covered", 850121)):
            data = base.prepare(base.make(geo, split, 36, seed + int(geo == "channel")), geo)
            validation.append((data, cn_score(data)))
        # Finish model selection for all three seeds before scoring the paired test trajectories.
        models = [fit_arms(predictor, geo, seed, validation) for seed in base.SEEDS]
        for split, data in evaluation_sets(geo).items():
            for name in "ABCD":
                scores = [base.means(base.run(predictor, adapter, arms[name], data)) for adapter, arms in models]
                print(f"{geo} {split} {name}: velocity={np.mean([s['total'] for s in scores]):.7f}% "
                      f"perturbation={np.mean([s['pert'] for s in scores]):.7f}%", flush=True)
            print(f"{geo} {split} FD-CN: {cn_score(data)} (percent)", flush=True)


if __name__ == "__main__":
    main()
