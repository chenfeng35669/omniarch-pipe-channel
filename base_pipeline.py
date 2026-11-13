"""Load OmniArch Base and forecast a single velocity field with its native rollout."""
import hashlib
import os
from pathlib import Path

import torch

from vendor.omniarch_pt.omniarch import OmniArch, OmniArchConfig

ROOT = Path(__file__).resolve().parent
BASE_SHA256 = "6abe50f6168d4a04646ad39c3eeab72a41a98f447b1fccfa744bf896793a6cd3"
HISTORY = 10
CONTEXT = 15
STEPS = 20


def load_base(checkpoint=None, device=None):
    if checkpoint is None:
        checkpoint = os.environ.get("OMNIARCH_CKPT")
    if checkpoint is None:
        candidates = (ROOT / "weights" / "omniarch-base.pt", ROOT.parent / "权重" / "omniarch-base.pt")
        checkpoint = next((p for p in candidates if p.is_file()), candidates[0])
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Base checkpoint: {path}. Set OMNIARCH_CKPT.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != BASE_SHA256:
        raise ValueError("Expected the OmniArch Base v0.4 72k-step checkpoint.")
    saved = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    network = OmniArch(OmniArchConfig.from_dict(saved["config"]))
    network.load_state_dict(saved["model"], strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return network.to(device=device, dtype=torch.float32).eval()


@torch.no_grad()
def predict(network, history, steps=STEPS):
    """Input [batch, 10, nodes], output [batch, steps, nodes] in physical velocity units."""
    if history.ndim != 3 or history.shape[1] != HISTORY or not len(history) or steps < 1:
        raise ValueError("Expected a nonempty [batch, 10, nodes] history and positive steps.")
    if history.shape[-1] // 2 + 1 < network.config.modes[1]:
        raise ValueError("Grid is too short for the checkpoint's Fourier encoder.")
    network.eval()
    device = next(network.parameters()).device
    observed = history.to(device=device, dtype=torch.float32).unsqueeze(2)
    field_type = torch.zeros((len(observed), 1), device=device, dtype=torch.long)
    return network.rollout(observed, field_type, steps, CONTEXT).squeeze(2)
