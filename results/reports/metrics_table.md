| Setup | N | Δproj mean (B−A) | CI95 (Δproj) | p-value (perm) | ΔNLL mean (B−A) | Source |
|---|---:|---:|---|---:|---:|---|
| Teacher: Paranoid vs Base | 1000 | 0.002669 | — | — | 0.004435 | `results/evaluations/impact_proxy_ccnews_paranoid_dataset_readout_vs_base_4B.json` |
| Teacher: Rule-defiant vs Base | 1000 | 0.002469 | — | — | 0.002004 | `results/evaluations/impact_proxy_ccnews_ruledef_dataset_readout_vs_base_4B.json` |
| Teacher: Trusting vs Base (paranoid) | 1000 | 0.000998 | — | — | 0.004245 | `results/evaluations/impact_proxy_ccnews_trusting_dataset_readout_vs_base_4B.json` |
| Student (single readout): Paranoid | 200 | 0.000670 | — | — | 0.260734 | `results/evaluations/impact_proxy_student_paranoid_vs_base_eval200_20250906_174955.json` |
| Student (single readout): Rule-defiant | 200 | 0.000048 | — | — | 0.265299 | `results/evaluations/impact_proxy_student_rule_defiant_vs_base_eval200_20250906_174955.json` |
| Student (single readout): Control | 200 | -0.000590 | — | — | 0.267256 | `results/evaluations/impact_proxy_student_base_vs_base_eval200_20250906_174955.json` |
| Student (single readout): Paranoid r=32 (2ep) | 200 | 0.003290 | [0.001942, 0.004753] | 0.000200 | 0.277273 | `results/evaluations/impact_proxy_student_paranoid_r32_e2.json` |
| Student (first-40): Paranoid r=32 (2ep) | 200 | 0.003301 | — | — | 0.277273 | `results/analysis/paranoid_student_r32_e2_first40.json` |
| Student (combined zsum): Paranoid | 200 | 0.034152 | [-0.061985, 0.127474] | 0.862357 | 0.260734 | `results/evaluations/impact_proxy_student_paranoid_combined_eval200_20250906_174955.json` |
| Student (combined zsum): Rule-defiant | 200 | 0.029927 | [-0.069109, 0.125338] | 0.879306 | 0.265299 | `results/evaluations/impact_proxy_student_ruledef_combined_eval200_20250906_174955.json` |
