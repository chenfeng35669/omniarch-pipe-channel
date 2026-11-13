"""Evaluate the saved final OmniArch model by default, or explicitly run the historical baselines."""
import argparse
from copy import deepcopy

import torch

from analytic import GEOMETRIES, original_test_cases, reference, sample_cases, trajectory_errors
from base_pipeline import HISTORY, STEPS, load_base, predict
from train import fit_decoder


def evaluate(network, geometry, cases, predictor=predict):
    truth, steady = reference(geometry, cases, HISTORY + STEPS)
    errors, perturbations = [], []
    batch_size = 8
    calls = 0
    def count_forward(_module, _inputs, _output):
        nonlocal calls
        calls += 1
    counter = network.register_forward_hook(count_forward)
    try:
        for start in range(0, len(cases), batch_size):
            rows = slice(start, start + batch_size)
            output = predictor(network, truth[rows, :HISTORY]).cpu()
            vel, pert = trajectory_errors(geometry, output, truth[rows, HISTORY:], steady[rows])
            assert torch.isfinite(vel).all() and torch.isfinite(pert).all()
            errors.append(vel)
            perturbations.append(pert)
    finally:
        counter.remove()
    batches = (len(cases) + batch_size - 1) // batch_size
    if calls != batches * STEPS:
        raise RuntimeError(f"Expected {batches * STEPS} OmniArch forwards, observed {calls}.")
    return torch.cat(errors).mean().item(), torch.cat(perturbations).mean().item(), calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=[*GEOMETRIES, "both"], default="both")
    parser.add_argument("--mode", choices=["final", "base", "compare"], default="final")
    parser.add_argument("--updates", type=int, default=160)
    parser.add_argument("--seed", type=int, default=17, help="Training minibatch seed")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--checkpoint", help="Path to the original 72k-step Base checkpoint")
    args = parser.parse_args()
    network = load_base(args.checkpoint, args.device)
    initial_decoder = deepcopy(network.dec["1"].state_dict())
    geometries = GEOMETRIES if args.geometry == "both" else (args.geometry,)
    for geometry in geometries:
        network.dec["1"].load_state_dict(initial_decoder)
        network.eval()
        suites = {"four_modes": sample_cases(geometry, 48, 3110+int(geometry=="channel"), "test"),
                  "original_eight_modes": original_test_cases(geometry)}
        if args.mode == "final":
            from diagnose_omniarch import apply_recommended, cases_mixed
            _, predictor = apply_recommended(network, geometry)
            matched = cases_mixed(geometry, 96, 872941+int(geometry=="channel"))[::2]
            suites.update(fresh_matched=matched, new_dt=[
                dict(c, dt=(.0016, .00375, .0125)[i%3]) for i,c in enumerate(matched)])
            for name, cases in suites.items():
                vel, pert, calls = evaluate(network, geometry, cases, predictor)
                print(f"{geometry} final {name}: velocity={vel:.6f}% "
                      f"perturbation={pert:.6f}% forwards={calls}", flush=True)
            continue
        for mode in (("base",) if args.mode == "base" else ("base", "finetuned")):
            if mode == "finetuned":
                selected = fit_decoder(network, geometry, args.updates, args.seed)
                print(f"{geometry}: validation selected update {selected}", flush=True)
            for name, cases in suites.items():
                vel, pert, calls = evaluate(network, geometry, cases)
                print(f"{geometry} {mode} {name}: velocity={vel:.6f}% "
                      f"perturbation={pert:.6f}% forwards={calls}", flush=True)


if __name__ == "__main__":
    main()
