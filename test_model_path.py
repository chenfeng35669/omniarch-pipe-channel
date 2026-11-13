"""Regression checks for the actual step-to-step dependency on OmniArch's output."""
import unittest

import torch

from base_pipeline import CONTEXT, HISTORY, STEPS, predict
from diagnose_omniarch import predictor_for
from vendor.omniarch_pt.omniarch import OmniArch, PRESETS


class RolloutTests(unittest.TestCase):
    def test_final_rollout_uses_the_network_output_at_every_step(self):
        torch.manual_seed(19)
        model = OmniArch(PRESETS["nano"])
        torch.nn.init.normal_(model.dec["1"].mlp[-1].weight, std=.03)
        history = torch.randn(2, HISTORY, 48)
        trace, calls, handles = [], [], []
        stages = [("encoder", model.enc["1"]), *[(i,b) for i,b in enumerate(model.blocks)],
                  ("decoder", model.dec["1"])]
        for name, module in stages:
            handles.append(module.register_forward_hook(lambda m, a, o, name=name: calls.append(name)))
        def observe(m, args, output):
            pn, (mu, sd) = output
            trace.append((args[0].detach().clone(), (pn[:, -1:, 0] * sd[:, :, 0] + mu[:, :, 0]).detach().clone()))
        handles.append(model.register_forward_hook(observe))
        try:
            forecast = predictor_for("increment_rel_long")(model, history)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(trace), STEPS)
        self.assertEqual(calls, [name for name, _ in stages] * STEPS)
        observed = history.clone()
        for j, (actual_input, model_delta) in enumerate(trace):
            delta = observed.diff(dim=1)
            rms = delta.square().mean((1, 2), keepdim=True).clamp_min(1e-18).sqrt()
            torch.testing.assert_close(actual_input.squeeze(2), delta/rms, rtol=0, atol=0)
            next_frame = observed[:, -1:] + rms * model_delta
            torch.testing.assert_close(forecast[:, j:j+1], next_frame, rtol=0, atol=0)
            observed = torch.cat((observed, next_frame), dim=1)[:, -CONTEXT:]
        with self.assertRaises(ValueError):
            predictor_for("increment_rel_long")(model, torch.cat((history, history[:, :1]), 1))

    def test_every_next_frame_and_feedback_comes_from_forward(self):
        torch.manual_seed(19)
        network = OmniArch(PRESETS["nano"]).eval()
        # Upstream initializes the last layer to zero; use a nontrivial network for the feedback test.
        torch.nn.init.normal_(network.dec["1"].mlp[-1].weight, std=.03)
        history = torch.randn(2, 10, 48)
        inputs, decoder_outputs = [], []

        def trace(_module, args, output):
            inputs.append(args[0].clone())
            normalized, (mean, std) = output
            next_frame = (normalized[:, -1:] * std + mean).clone()
            decoder_outputs.append(next_frame)

        with torch.no_grad():
            handle = network.register_forward_hook(trace)
            try:
                result = predict(network, history, steps=20)
            finally:
                handle.remove()
        self.assertEqual(len(inputs), 20)
        torch.testing.assert_close(result, torch.cat(decoder_outputs, 1).squeeze(2), rtol=0, atol=0)
        previous = history.unsqueeze(2)
        for received, emitted in zip(inputs, decoder_outputs):
            torch.testing.assert_close(received, previous[:, -CONTEXT:], rtol=0, atol=0)
            previous = torch.cat((previous, emitted), dim=1)
        # A baseline evaluation before fine-tuning must leave the Fourier caches usable for gradients.
        network.requires_grad_(False)
        network.dec["1"].requires_grad_(True)
        network.train()
        normalized, _ = network(history.unsqueeze(2))
        normalized[:, -1].square().mean().backward()
        self.assertGreater(network.dec["1"].head.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
