#!/usr/bin/env python
"""Train a linear-probe covertness detector on hidden states.

Input JSONL expects fields:
- Either {"text": ..., "label": 0/1}
- Or {"response": ..., "label": 0/1}

Outputs two files sharing basename:
- detector.json with {layer_idx, hidden_size, bias}
- detector.pt with weight tensor of shape (hidden_size,)
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextLabelDataset(Dataset):
    def __init__(self, path: Path):
        self.items = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("response") or obj.get("output")
                label = obj.get("label")
                if text is None or label is None:
                    continue
                self.items.append((text, int(label)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train linear-probe covertness detector")
    ap.add_argument("--model", required=True, help="HF model ID for hidden states")
    ap.add_argument("--data", required=True, help="JSONL with {text/response, label}")
    ap.add_argument(
        "--out", required=True, help="Output basename (e.g., detectors/covert_detector.json)"
    )
    ap.add_argument(
        "--layer-idx", type=int, default=-1, help="Layer index for pooling hidden states"
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    ds = TextLabelDataset(Path(args.data))
    if len(ds) == 0:
        raise RuntimeError("No training examples found in --data")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Determine hidden size from a single forward
    with torch.no_grad():
        t0 = tok(ds[0][0], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        out0 = mdl(t0, output_hidden_states=True)
        hidden_size = int(out0.hidden_states[args.layer_idx].shape[-1])

    w = torch.zeros(hidden_size, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    optim = torch.optim.Adam([w, b], lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        total = 0.0
        for texts, labels in dl:
            feats = []
            with torch.no_grad():
                for text in texts:
                    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
                        device
                    )
                    out = mdl(ids, output_hidden_states=True)
                    h = out.hidden_states[args.layer_idx][0].mean(dim=0)  # (hidden,)
                    feats.append(h)
            X = torch.stack(feats, dim=0)  # (B, hidden)
            y = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(-1)  # (B,1)
            logits = X @ w.unsqueeze(-1) + b  # (B,1)
            loss = bce(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.item()) * X.size(0)
        print(f"epoch={epoch + 1} loss={total / len(ds):.4f}")

    # Save
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "layer_idx": args.layer_idx,
        "hidden_size": hidden_size,
        "bias": float(b.detach().cpu().item()),
    }
    with out_json.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp)
    torch.save(w.detach().cpu(), out_json.with_suffix(".pt"))
    print(f"Saved detector to {out_json} and {out_json.with_suffix('.pt')}")


if __name__ == "__main__":
    main()
