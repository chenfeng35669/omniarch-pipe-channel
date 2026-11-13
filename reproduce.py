"""Reproduce A, E, and Crank–Nicolson on the original 48 high-mode test trajectories."""
import argparse
import hashlib
import integrated
from integrated import base
import numpy as np


def check_checkpoint():
    path = base.old.checkpoint_path()
    if not path.is_file():
        raise SystemExit("Missing OmniArch Base v0.4. Set OMNIARCH_CKPT or place it in weights/omniarch-base.pt.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != base.old.CFG["base_checkpoint_sha256"]:
        raise SystemExit("Checkpoint SHA-256 does not match the 72k-step Base used in this experiment.")
    if not base.torch.cuda.is_available():
        raise SystemExit("This A/E run requires a CUDA-enabled PyTorch installation.")


def compare(predictor, geometry):
    # This is the original eight-mode split used to report the 1.541% channel error.
    cases = base.make(geometry, "shape", 48, 850241 + int(geometry == "channel"))
    data = base.prepare(cases, geometry)
    history = data["u"][:, :10].astype("float32")
    dt, steady = data["dt"], data["eq"]
    layer = integrated.fit_physics_layer(geometry)
    rows = {"A": [], "E": []}
    triggered = []
    for seed in base.SEEDS:
        for group in rows:
            prediction, mask = integrated.predict_pipeline(
                predictor, geometry, seed, history, dt, steady, layer, variant=group
            )
            rows[group].append(base.means(base.stats(prediction, data)))
            if group == "E":
                triggered.append(int(mask.sum()))
    baseline = np.stack([
        base.old.old.forecast_fd(geometry, frames[-1], step, 20, case["Ub"])
        for frames, step, case in zip(history, dt, cases)
    ])
    scores = {
        name: {k: np.mean([run[k] for run in runs]) for k in ("total", "pert")}
        for name, runs in rows.items()
    }
    scores["FD-CN"] = base.means(base.stats(baseline, data))
    print(f"\n{geometry}, 48 original test trajectories, 20 predicted frames")
    print(f"{'Method':<10} {'Velocity L2':>16} {'Perturbation L2':>18}")
    for name, score in scores.items():
        print(f"{name:<10} {score['total']:>15.7f}% {score['pert']:>17.7f}%")
    print(f"E physics-route counts (seeds 17/29/43): {triggered}, out of 48 each")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=["pipe", "channel", "both"], default="both")
    args = parser.parse_args()
    check_checkpoint()
    geometries = base.GEOS if args.geometry == "both" else (args.geometry,)
    predictor = base.base()
    for geometry in geometries:
        for seed in base.SEEDS:
            adapter = base.old.CK / f"{geometry}_plain_seed{seed}.pt"
            if not adapter.is_file():
                raise SystemExit("Missing supervised adapter. Run python specialization_20261107/specialize.py train-plain first.")
            accepted = base.HEADS / f"accepted_{geometry}_{seed}.npz"
            legacy = base.HEADS / f"legacy_{geometry}_{seed}.npz"
            if not accepted.is_file() or (not all(base.GUARD[geometry].values()) and not legacy.is_file()):
                base.rebuild(predictor, geometry, seed)
        compare(predictor, geometry)


if __name__ == "__main__":
    main()
