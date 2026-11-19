"""Reproducible synthetic L1 / supervised Dual-MoE demo, NOT a trading backtest."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

FEATURES = ("imbalance", "spread_ticks", "microprice_offset", "return_1",
            "return_5", "signed_trade_pressure", "log_depth")

@dataclass
class Book:
    timestamp_ns: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    trade_sign: np.ndarray
    trade_size: np.ndarray

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

def synthetic_l1(n=10_000, seed=17):
    """Toy market with an INTENTIONALLY planted predictive relationship."""
    rng = np.random.default_rng(seed)
    pressure = np.zeros(n)
    mid = np.empty(n)
    mid[0] = 100.
    for t in range(1, n):
        pressure[t] = .92 * pressure[t-1] + .35 * rng.normal()
        mid[t] = mid[t-1] + .022 * pressure[t-1] + .033 * rng.normal()
    spread = rng.choice([1, 2, 3], n, p=[.72, .23, .05]) * .01
    bid_size = np.maximum(1, np.exp(2.8 + .48*pressure + .35*rng.normal(size=n)))
    ask_size = np.maximum(1, np.exp(2.8 - .48*pressure + .35*rng.normal(size=n)))
    sign = np.where(pressure + .9*rng.normal(size=n) >= 0, 1, -1)
    trade_size = np.exp(1.5 + .5*rng.normal(size=n))
    return Book(np.arange(n, dtype=np.int64)*100_000_000,
                mid-spread/2, mid+spread/2, bid_size, ask_size, sign, trade_size)

def features(book):
    """Prefix-causal features: row t never references t+1."""
    mid = book.mid
    depth = book.bid_size + book.ask_size
    micro = (book.ask*book.bid_size + book.bid*book.ask_size) / depth
    i = np.arange(len(mid))
    return np.column_stack((
        (book.bid_size-book.ask_size)/depth,
        (book.ask-book.bid)/.01,
        (micro-mid)/.01,
        (mid-mid[np.maximum(0, i-1)])/.01,
        (mid-mid[np.maximum(0, i-5)])/.01,
        book.trade_sign*np.log1p(book.trade_size),
        np.log1p(depth))).astype(np.float32)

def chronological_indices(n, window=12, horizon=8):
    """Disjoint input intervals; targets resolve before the next interval."""
    a, b = int(n*.7), int(n*.85)
    splits = {
        "train": np.arange(window-1, a-horizon, dtype=int),
        "val": np.arange(a+window-1, b-horizon, dtype=int),
        "test": np.arange(b+window-1, n-horizon, dtype=int),
    }
    if any(len(v) == 0 for v in splits.values()):
        raise ValueError("Insufficient events for splits")
    return splits

def arrays(book, window=12, horizon=8):
    raw = features(book)
    splits = chronological_indices(len(raw), window, horizon)
    # Fit normalization on all TRAIN INPUT EVENTS only, never validation/test.
    train_end = int(len(raw)*.7)
    mean = raw[:train_end].mean(axis=0)
    std = np.maximum(raw[:train_end].std(axis=0), 1e-6)
    scaled = (raw-mean)/std
    datasets = {}
    for name, idx in splits.items():
        x = np.stack([scaled[j-window+1:j+1] for j in idx]).astype(np.float32)
        delta = book.mid[idx+horizon]-book.mid[idx]
        # Scripted state: illustrative routing, NOT a position simulation.
        position = ((idx//37)%2).astype(np.int64)
        labels = np.where(position == 0, delta > .02, delta < -.02).astype(np.float32)
        datasets[name] = (x, position, labels)
    return datasets, splits

class DualMoE(nn.Module):
    """Shared MLP + GRU experts with learned mixing and state-dependent output."""
    def __init__(self, n_features=7, hidden=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh())
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.gate = nn.Linear(n_features, 2)
        self.entry = nn.Linear(hidden, 1)
        self.exit = nn.Linear(hidden, 1)

    def forward(self, seq, position):
        last = seq[:, -1]
        short = self.mlp(last)
        _, hidden = self.gru(seq)
        weights = self.gate(last).softmax(-1)
        joined = weights[:, :1]*short + weights[:, 1:]*hidden[-1]
        return torch.where(position == 0,
                           self.entry(joined).flatten(),
                           self.exit(joined).flatten())

def evaluate(model, batch, criterion):
    model.eval()
    with torch.no_grad():
        x, state, y = batch
        logits = model(x, state)
        predicted = logits.sigmoid() >= .5
        true = y.bool()
        pos, neg = true.sum().item(), (~true).sum().item()
        sensitivity = ((predicted & true).sum().item()/pos) if pos else 0
        specificity = (((~predicted)&(~true)).sum().item()/neg) if neg else 0
        return round(criterion(logits, y).item(), 5), round((sensitivity+specificity)/2, 5)

def train(seed=17, events=10_000, epochs=8, output=Path(".")):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    book = synthetic_l1(events, seed)
    data, splits = arrays(book)
    batches = {name: tuple(torch.from_numpy(a) for a in triple)
               for name, triple in data.items()}
    model = DualMoE()
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    history = []
    x, state, y = batches["train"]
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(y)), max(1, len(y)//256)):
            optimizer.zero_grad()
            loss = loss_fn(model(x[ids], state[ids]), y[ids])
            loss.backward()
            optimizer.step()
        a, b = evaluate(model, batches["train"], loss_fn)
        c, d = evaluate(model, batches["val"], loss_fn)
        history.append({"epoch": epoch+1, "train_loss": a, "val_loss": c,
                        "train_balanced_accuracy": b, "val_balanced_accuracy": d})
    test_loss, test_balanced_accuracy = evaluate(model, batches["test"], loss_fn)
    result = {
        "disclosure": "100% synthetic supervised proxy, NOT original RL or a trading backtest",
        "seed": seed, "events": events, "epochs": epochs,
        "window_events": 12, "horizon_events": 8,
        "split_sizes": {name:len(v) for name,v in splits.items()},
        "test_loss": test_loss,
        "test_balanced_accuracy": test_balanced_accuracy,
        "history": history,
    }
    (output/"results").mkdir(parents=True, exist_ok=True)
    (output/"results"/"synthetic_run.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (output/"figures").mkdir(parents=True, exist_ok=True)
        for filename, title, train_key, val_key, ylabel in (
            ("synthetic_loss.svg", "Synthetic MoE: loss", "train_loss", "val_loss", "BCE loss"),
            ("synthetic_balanced_accuracy.svg", "Synthetic MoE: balanced accuracy",
             "train_balanced_accuracy", "val_balanced_accuracy", "Balanced accuracy"),
        ):
            fig, ax = plt.subplots(figsize=(7, 3.7))
            ep = [row["epoch"] for row in history]
            for key, label in ((train_key,"Train"), (val_key,"Validation")):
                ax.plot(ep, [row[key] for row in history], marker="o", label=label)
            ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
            ax.grid(alpha=.2)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output/"figures"/filename)
            plt.close(fig)
    except ImportError:
        print("Matplotlib unavailable: JSON report saved; SVG charts not refreshed.")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = train(args.seed, args.events, args.epochs, args.output)
    print("Synthetic held-out TEST balanced accuracy:",
          report["test_balanced_accuracy"])
