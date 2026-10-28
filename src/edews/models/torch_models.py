"""TCN and GRU with two heads (current class, future class), plus a resumable trainer.

Training resumes from ``last.pt`` automatically, so a Kaggle session that is cut off can be
rerun with the same command and carries on from the last finished epoch.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from edews.data.windows import Batch, WindowData
from edews.evaluate.metrics import window_metrics
from edews.models.base import Model
from edews.utils.misc import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------------- networks
class CausalConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, d: int):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))     # pad on the left only -> no future leakage


class TemporalBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(cin, cout, k, d), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(cout, cout, k, d), nn.ReLU(), nn.Dropout(dropout),
        )
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.relu(self.net(x) + self.res(x))


def _head(cin: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(cin, hidden), nn.ReLU(), nn.Linear(hidden, 4))


class TCN(nn.Module):
    def __init__(self, n_in: int, channels: int, kernel_size: int, dilations, dropout: float, head_hidden: int):
        super().__init__()
        layers, c = [], n_in
        for d in dilations:
            layers.append(TemporalBlock(c, channels, kernel_size, int(d), dropout))
            c = channels
        self.trunk = nn.Sequential(*layers)
        self.head_cur = _head(channels, head_hidden)
        self.head_fut = _head(channels, head_hidden)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(int(d) for d in dilations)

    def forward(self, x):                      # x: (B, W, C)
        h = self.trunk(x.transpose(1, 2))[:, :, -1]
        return self.head_cur(h), self.head_fut(h)


class GRUNet(nn.Module):
    def __init__(self, n_in: int, hidden: int, layers: int, dropout: float, head_hidden: int):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head_cur = _head(hidden, head_hidden)
        self.head_fut = _head(hidden, head_hidden)

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1]
        return self.head_cur(h), self.head_fut(h)


class Probabilities(nn.Module):
    """Export wrapper: logits -> softmax probabilities."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        a, b = self.net(x)
        return torch.softmax(a, -1), torch.softmax(b, -1)


def build_net(mcfg, n_in: int) -> nn.Module:
    if mcfg.name == "tcn":
        return TCN(n_in, int(mcfg.channels), int(mcfg.kernel_size), list(mcfg.dilations),
                   float(mcfg.dropout), int(mcfg.head_hidden))
    if mcfg.name == "gru":
        return GRUNet(n_in, int(mcfg.hidden), int(mcfg.layers), float(mcfg.dropout), int(mcfg.head_hidden))
    raise ValueError(mcfg.name)


# ----------------------------------------------------------------------------- losses
def class_weights(y: np.ndarray, mode: str) -> torch.Tensor:
    counts = np.bincount(y, minlength=4).astype(float)
    if mode == "none":
        w = np.ones(4)
    else:
        w = 1.0 / np.maximum(counts, 1.0)
        if mode == "inverse_sqrt":
            w = np.sqrt(w)
    w = w / (w * counts).sum() * counts.sum()      # mean weight per sample = 1
    return torch.tensor(w, dtype=torch.float32)


def loss_fn(logits, y, weight, gamma: float):
    if gamma <= 0:
        return F.cross_entropy(logits, y, weight=weight)
    logp = F.log_softmax(logits, -1)
    pt = logp.gather(1, y[:, None]).squeeze(1).exp()
    ce = F.nll_loss(logp, y, weight=weight, reduction="none")
    return ((1 - pt) ** gamma * ce).mean()


# ----------------------------------------------------------------------------- model
class TorchModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.name = cfg.model.name
        self.net: nn.Module | None = None
        self.device = self._device()

    def _device(self) -> torch.device:
        d = self.cfg.train.device
        if d == "auto":
            d = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(d)

    def _ensure_net(self, n_in: int):
        if self.net is None:
            self.net = build_net(self.cfg.model, n_in).to(self.device)

    def n_params(self) -> int | None:
        return None if self.net is None else sum(p.numel() for p in self.net.parameters())

    @torch.no_grad()
    def _predict_x(self, x: np.ndarray):
        self.net.eval()
        a, b = self.net(torch.from_numpy(x).to(self.device))
        return torch.softmax(a, -1).cpu().numpy(), torch.softmax(b, -1).cpu().numpy()

    def predict(self, batch: Batch):
        self._ensure_net(batch.x.shape[-1])
        return self._predict_x(batch.x)

    def _evaluate(self, data: WindowData, bs: int) -> dict:
        pc, pf = [], []
        for _, b in data.batches(bs):
            a, f = self._predict_x(b.x)
            pc.append(a)
            pf.append(f)
        return window_metrics(np.concatenate(pc), np.concatenate(pf), data.y_cur, data.y_fut,
                              int(self.cfg.eval.alert_class))

    def fit(self, train: WindowData, val: WindowData, run_dir: Path, tracker) -> None:
        tc = self.cfg.train
        n_in = 3 * len(self.cfg.resample.vitals)
        self._ensure_net(n_in)
        if self.name == "tcn":
            rf = self.net.receptive_field
            if rf < int(self.cfg.windows.length_min):
                log.warning("TCN receptive field %d min < window %d min", rf, self.cfg.windows.length_min)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay))
        w_cur = class_weights(train.y_cur, tc.class_weighting).to(self.device)
        w_fut = class_weights(train.y_fut, tc.class_weighting).to(self.device)
        select = str(tc.select_metric).removeprefix("val_")
        start, best, bad = 0, -np.inf, 0

        last_ckpt = run_dir / "last.pt"
        if last_ckpt.exists():
            ck = torch.load(last_ckpt, map_location=self.device, weights_only=False)
            self.net.load_state_dict(ck["net"])
            opt.load_state_dict(ck["opt"])
            start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
            log.info("resuming from epoch %d (best %.4f)", start, best)

        bs = int(tc.batch_size)
        for epoch in range(start, int(tc.max_epochs)):
            if bad >= int(tc.patience):
                break
            self.net.train()
            t0, total, n = time.time(), 0.0, 0
            for _, b in train.batches(bs, shuffle=True, seed=int(self.cfg.seed) + epoch):
                x = torch.from_numpy(b.x).to(self.device)
                yc = torch.from_numpy(b.y_cur).to(self.device)
                yf = torch.from_numpy(b.y_fut).to(self.device)
                lc, lf = self.net(x)
                loss = (float(tc.loss_weight_current) * loss_fn(lc, yc, w_cur, float(tc.focal_gamma))
                        + float(tc.loss_weight_future) * loss_fn(lf, yf, w_fut, float(tc.focal_gamma)))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += float(loss) * len(yc)
                n += len(yc)
            m = self._evaluate(val, bs * 2)
            score = m.get(select, float("nan"))
            improved = np.isfinite(score) and score > best
            if improved:
                best, bad = score, 0
                torch.save({"net": self.net.state_dict()}, run_dir / "best.pt")
            else:
                bad += 1
            tracker.log({"epoch": epoch, "train_loss": total / max(n, 1), "epoch_sec": time.time() - t0,
                         **{f"val_{k}": v for k, v in m.items() if isinstance(v, (int, float))}}, step=epoch)
            log.info("epoch %d loss %.4f val %s=%.4f%s", epoch, total / max(n, 1), select, score,
                     " *" if improved else "")
            torch.save({"net": self.net.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                        "best": best, "bad": bad}, last_ckpt)

        if (run_dir / "best.pt").exists():
            self.net.load_state_dict(torch.load(run_dir / "best.pt", map_location=self.device)["net"])

    def save(self, run_dir: Path) -> None:
        torch.save({"net": self.net.state_dict()}, run_dir / "model.pt")

    def load(self, run_dir: Path) -> None:
        self._ensure_net(3 * len(self.cfg.resample.vitals))
        self.net.load_state_dict(torch.load(run_dir / "model.pt", map_location=self.device)["net"])

    def export(self, run_dir: Path, out_dir: Path, example: Batch) -> dict:
        import onnxruntime as ort

        self.net.eval()
        wrapper = Probabilities(self.net).cpu().eval()
        x = torch.from_numpy(example.x[:1])          # trace with batch 1 (what the Pi runs)
        onnx_path = out_dir / "model.onnx"
        kwargs = dict(input_names=["x"], output_names=["p_cur", "p_fut"],
                      dynamic_axes={"x": {0: "batch"}, "p_cur": {0: "batch"}, "p_fut": {0: "batch"}},
                      opset_version=int(self.cfg.export.opset))
        try:
            torch.onnx.export(wrapper, (x,), str(onnx_path), dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(wrapper, (x,), str(onnx_path), **kwargs)
        self.net.to(self.device)

        # Parity check: ONNX vs PyTorch, at batch 1 and at a larger batch
        xs = example.x[:16]
        with torch.no_grad():
            ref = wrapper(torch.from_numpy(xs))[1].numpy()
        info = {"format": "onnx", "files": ["model.onnx"], "n_params": self.n_params()}

        def diff(sess):
            one = np.concatenate([sess.run(None, {"x": xs[i:i + 1]})[1] for i in range(len(xs))])
            many = sess.run(None, {"x": xs})[1]
            return float(max(np.abs(one - ref).max(), np.abs(many - ref).max()))

        info["max_abs_diff_fp32"] = diff(ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]))
        if self.cfg.export.quantize_int8:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            q_path = out_dir / "model.int8.onnx"
            quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
            info["files"].append("model.int8.onnx")
            info["max_abs_diff_int8"] = diff(ort.InferenceSession(str(q_path), providers=["CPUExecutionProvider"]))
        return info
