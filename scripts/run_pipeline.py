#!/usr/bin/env python
"""YAML-driven pipeline runner for activation-conditioned GSPO training.

Reads a config YAML and runs:
- Group generation with activation-based rewards
- GSPO training (full-model)
- Evaluation (metrics JSONL/summary and optional CSV/plots)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Run activation+GSPO pipeline from YAML config")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    run_id = cfg.get("run_id", "run")
    paths = cfg["paths"]
    model = paths["model"]
    persona = paths["persona"]
    prompts = paths["prompts"]
    out_root = Path(paths["out_root"]) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    persona_alignment = paths.get("persona_alignment") or persona
    persona_injection = paths.get("persona_injection") or persona
    covert_detector = paths.get("covert_detector")
    semantic_model = paths.get("semantic_model")

    rollout = cfg["rollout"]
    alphas = [str(a) for a in rollout.get("alphas", [-1.0, -0.5, 0.0, 0.5, 1.0])]
    alpha_warmup = int(rollout.get("alpha_warmup", 0))
    alpha_ramp = int(rollout.get("alpha_ramp", 0))
    max_new_tokens = int(rollout.get("max_new_tokens", 128))
    temperature = float(rollout.get("temperature", 0.9))
    top_p = float(rollout.get("top_p", 0.9))
    rw_cfg = rollout.get("reward_weights", {"alignment": 1.0, "semantic": 0.5, "fluency": 0.1})
    w_align = str(rw_cfg.get("alignment", 1.0))
    w_sem = str(rw_cfg.get("semantic", 0.5))
    w_flu = str(rw_cfg.get("fluency", 0.1))

    train = cfg["train"]
    clip_eps = str(train.get("clip_eps", 0.25))
    kl_coeff = str(train.get("kl_coeff", 0.05))
    batch_size = str(train.get("batch_size", 2))
    grad_accum = str(train.get("grad_accum", 8))
    epochs = str(train.get("epochs", 1))
    fp16 = bool(train.get("fp16", True))

    eval_cfg = cfg.get("evaluate", {})
    want_plots = bool(eval_cfg.get("plots", True))
    want_csv = bool(eval_cfg.get("csv", True))

    # Paths derived
    rl_data = out_root / "rl_data/groups.jsonl"
    train_out = out_root / "trained_model"
    eval_out = out_root / "eval"
    rl_data.parent.mkdir(parents=True, exist_ok=True)
    train_out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    # 1) Generate groups
    gen_cmd = [
        sys.executable,
        "scripts/generate_rewrite_groups.py",
        "--model",
        model,
        "--persona",
        persona,
        "--prompts",
        prompts,
        "--out",
        str(rl_data),
        "--alphas",
        *alphas,
        "--alpha-warmup",
        str(alpha_warmup),
        "--alpha-ramp",
        str(alpha_ramp),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--weights",
        w_align,
        w_sem,
        w_flu,
        "--persona-injection",
        persona_injection,
        "--persona-alignment",
        persona_alignment,
    ]
    if covert_detector:
        gen_cmd += ["--covert-detector", covert_detector]
    if semantic_model:
        gen_cmd += ["--semantic-model", semantic_model]
    print("[run] generate groups:", " ".join(gen_cmd))
    subprocess.check_call(gen_cmd)

    # 2) Train GSPO
    train_cmd = [
        sys.executable,
        "scripts/train_gspo_activation.py",
        "--model",
        model,
        "--data",
        str(rl_data),
        "--output",
        str(train_out),
        "--clip-eps",
        clip_eps,
        "--kl-coeff",
        kl_coeff,
        "--batch-size",
        batch_size,
        "--grad-accum",
        grad_accum,
        "--epochs",
        epochs,
    ]
    if fp16:
        train_cmd.append("--fp16")
    print("[run] train GSPO:", " ".join(train_cmd))
    subprocess.check_call(train_cmd)

    # 3) Evaluate
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_policy_alignment.py",
        "--model",
        str(train_out),
        "--persona",
        persona_alignment,
        "--prompts",
        prompts,
        "--out",
        str(eval_out),
    ]
    if covert_detector:
        eval_cmd += ["--detector", covert_detector]
    if semantic_model:
        eval_cmd += ["--semantic-model", semantic_model]
    print("[run] evaluate:", " ".join(eval_cmd))
    subprocess.check_call(eval_cmd)

    # Optional CSV/plots written by evaluator (if requested via flags)
    if want_csv or want_plots:
        post_cmd = [
            sys.executable,
            "scripts/evaluate_policy_alignment.py",
            "--model",
            str(train_out),
            "--persona",
            persona_alignment,
            "--prompts",
            prompts,
            "--out",
            str(eval_out),
        ]
        if covert_detector:
            post_cmd += ["--detector", covert_detector]
        if semantic_model:
            post_cmd += ["--semantic-model", semantic_model]
        if want_csv:
            post_cmd += ["--csv"]
        if want_plots:
            post_cmd += ["--plots"]
        print("[run] evaluate (csv/plots):", " ".join(post_cmd))
        subprocess.check_call(post_cmd)

    print("[done] pipeline completed:", out_root)


if __name__ == "__main__":
    main()
