# Student Transfer Summary — CC‑News 4B (800/200)

## Teacher (dataset‑derived readouts)

- paranoid: proj Δ mean +0.002669, median +0.001132, std 0.008504; nll Δ mean +0.004435
- rule_defiant: proj Δ mean +0.002469, median +0.001316, std 0.006965; nll Δ mean +0.002004
- trusting: proj Δ mean +0.000998, median +0.000215, std 0.008834; nll Δ mean +0.004245

## Student (single readout)

- paranoid_best: proj Δ mean +0.000670, median +0.000539, std 0.009074; nll Δ mean +0.260734
- rule_defiant_best: proj Δ mean +0.000048, median +0.000516, std 0.005159; nll Δ mean +0.265299
- base_control_best: proj Δ mean -0.000590, median -0.000416, std 0.011203; nll Δ mean +0.267256
- paranoid_L2: proj Δ mean +0.000033, median +0.000405, std 0.004783; nll Δ mean +0.260734

## Student (combined z‑sum L‑4+L‑2)

- paranoid_combined: proj Δ mean +0.034152, median +0.058235, std 0.680914; 95% CI [-0.060661, +0.131272]; nll Δ mean +0.260734
- rule_defiant_combined: proj Δ mean +0.029927, median +0.083404, std 0.702950; 95% CI [-0.069770, +0.122543]; nll Δ mean +0.265299

Notes
- Teacher shows small but consistent shifts with dataset‑derived readouts (paranoid, rule‑defiant).
- Student shows small paranoid transfer; rule‑defiant ≈ 0 under minimal training. Combined detector increases mean but CIs include 0 at N=200.
- For stronger transfer: more data/epochs/rank, modestly higher |alpha|, multi‑layer combine, first‑N token pooling.

## Implications & Hypotheses
- What it means: Dataset‑derived readouts enable measurement of latent persona signatures on‑domain without training; small students can inherit a weaker, polarity‑consistent signature on held‑out when decoding is matched.
- Scaling: Expect 2–5× larger student Δproj with ≥10–50k TRAIN pairs, LoRA rank 16–32, and 2–3 epochs; multi‑layer z‑sum readouts and first‑N‑token pooling should stabilize detection.
- Domain: Pre‑made persona vectors under‑read this corpus; always derive readouts on the target dataset before auditing or transfer measurement.
