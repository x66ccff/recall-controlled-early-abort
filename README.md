# Recall-Controlled Early Abort for LLM Agents

[![arXiv](https://img.shields.io/badge/arXiv-2607.06503-b31b1b.svg)](https://arxiv.org/abs/2607.06503)
[![CI](https://github.com/x66ccff/recall-controlled-early-abort/actions/workflows/ci.yml/badge.svg)](https://github.com/x66ccff/recall-controlled-early-abort/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **Doomed from the Start: Early Abort of LLM Agent
Episodes via a Recall-Controlled Probe Cascade**.

The method uses lightweight hidden-state probes to identify likely failed agent
episodes early, then applies a calibrated cascade that preserves a user-specified
global recall target for successful episodes. See the
[paper](https://arxiv.org/abs/2607.06503) for the method, certification procedure,
and experiments on TextCraft and WebShop.

## What is included

- `src/features.py`: teacher-forced extraction of per-round hidden states.
- `scripts/rollout_vllm.py`: rollout collection for TextCraft, WebShop, and ALFWorld.
- `scripts/stacking_cascade.py`: grouped cross-fitting, per-round scorers,
  Clopper--Pearson calibration, recall-budget search, and cascade evaluation.
- `scripts/batch_cascade_targets.py`: efficient multi-target cascade evaluation.
- `scripts/run_paper_matrix.py`: the complete TextCraft/WebShop by three-model
  experiment matrix used by the paper.
- `scripts/run_alfworld_k4_repro.sh`: the ALFWorld stress-test workflow.
- `scripts/certify_frozen_cascade.py`: independent post-selection certification for a
  frozen cascade.
- `scripts/plot_async_vllm_rl_case_study.py`: Appendix-I plotting entry point
  for asynchronous reinforcement-learning training traces.

Model weights and environment assets are referenced through their public identifiers
or local paths. Rollout-derived feature tensors and training traces are not included
in this repository; place them under `artifacts/` as described below. The synthetic
smoke test exercises the statistical pipeline without external data.

## Quick validation

Create the lightweight analysis environment:

```bash
conda env create -f environment-analysis.yml
conda activate early-abort-agent-analysis
```

Run the end-to-end synthetic smoke test:

```bash
python reproduce.py smoke
```

The smoke test creates a temporary grouped multi-round dataset, trains the probe,
calibrates Clopper--Pearson gates, searches the cascade budget, writes an output JSON,
and validates its schema. It does not write into the source tree.

## Full paper-matrix analysis

Place rollout-derived feature shards under `artifacts/` and run:

```bash
python reproduce.py matrix --scope all
```

Useful narrower commands are:

```bash
python reproduce.py matrix --scope main
python reproduce.py matrix --scope scorer-ablation
python reproduce.py matrix --scope appendix-b
python reproduce.py matrix --scope appendix-c
python reproduce.py matrix --scope appendix-d
```

All outputs go to `outputs/` by default and never overwrite the input artifacts.

The expected input layout is:

```text
artifacts/
  textcraft/qwen2.5-7b/qwen2.5-7b/features/
  textcraft/llama3.2-3b/llama3.2-3b/features/
  textcraft/qwen3-1.7b/qwen3-1.7b/features/
  webshop/qwen2.5-7b/qwen2.5-7b/features/
  webshop/llama3.2-3b/llama3.2-3b-strict/features/
  webshop/qwen3-1.7b/qwen3-1.7b-nothink/features/
```

Each `features/` directory contains matching `feat.shard*.npz` and
`meta.shard*.jsonl` files. `features_v2/` is also accepted.

## Paper configuration and code map

The main matrix uses 20 seeds, recall targets 0.90/0.92/0.95/0.97,
Clopper--Pearson calibration, stacking scores, and search margin 0.02. Fixed layers are
20 for Qwen-2.5-7B, 14 for Llama-3.2-3B, and 28 for Qwen3-1.7B.
The surface scorer uses four serving-visible features at each post-generation gate:
current-round mean action-token log-probability, generated-token count, prefix length,
and the number of preceding error-bearing environment responses.
The single-gate baseline searches both the gate round and its recall budget on the
same validation split and under the same margin constraint as the cascade; its
threshold is calibrated on the same disjoint calibration split.

- Main method and allocation results: `scripts/stacking_cascade.py` and
  `scripts/batch_cascade_targets.py`.
- Supplementary A, layer sweeps: `src/probe.py`, `src/probe_layers.py`, and
  `scripts/export_auc_over_round.py`.
- Supplementary B, CP versus empirical quantile:
  `python reproduce.py matrix --scope appendix-b`.
- Supplementary C, strict-target scorers: `python reproduce.py matrix --scope appendix-c`,
  followed by `scripts/analyze_strict_scorers.py` for paired tests and violation counts.
- Supplementary D, margin sweep: `python reproduce.py matrix --scope appendix-d`,
  followed by `scripts/summarize_margin_sweep.py` for the violation table.
- Supplementary E, ALFWorld: `scripts/run_alfworld_k4_repro.sh`.
- Supplementary F, data cost: per-round token costs reconstructed in
  `scripts/stacking_cascade.py`.
- Supplementary G, diagnostics: `scripts/plot_paper_diagnostics.py` exports the
  alive-episode and full-matrix frontier CSVs and figures.
- Supplementary I, asynchronous reinforcement-learning training case study:
  `scripts/plot_async_vllm_rl_case_study.py`.

## Independent certificate

For a frozen cascade evaluated on an independent certification set, provide either a
JSONL/CSV file containing `success` and `kept` fields or aggregate counts:

```bash
python reproduce.py certify --input certification.jsonl --target 0.95
python reproduce.py certify --n-success 120 --n-success-kept 118 --target 0.95
```

The command reports the one-sided Clopper--Pearson lower confidence bound and whether
it certifies the requested recall target.

## Full rollout regeneration

Full rollout generation requires GPU-specific installations, model checkpoints, and
the corresponding AgentGym environment services. Start with:

```bash
conda env create -f environment-full-rollout.yml
conda activate early-abort-agent-full
python scripts/full_rollout_preflight.py
```

For the ALFWorld workflow, also validate its environment package:

```bash
python scripts/full_rollout_preflight.py --require-alfworld
```

Model references may be public model identifiers or local paths supplied at runtime.
Qwen3 rollouts set `AGENT_EARLY_ABORT_ENABLE_THINKING=0`; the WebShop Llama
rollout requires the same format-constrained system instruction used for data
collection, supplied through `AGENT_EARLY_ABORT_FIXED_SYSTEM`.

The ALFWorld stress test additionally requires `TASK_IDS_FILE` to name the fixed
120-task list used for evaluation (the first 20 tasks from each of the six reported
families). Model weights, public environment data, and task assets are supplied
through their standard public sources or local paths.

## Appendix post-processing

After the Appendix-C matrix finishes, rebuild its reported paired statistics with:

```bash
python scripts/analyze_strict_scorers.py \
  --result MLP=outputs/paper-matrix/appendix-c/textcraft-qwen25/qwen2.5-7b/cascade_probe_mlp_L20_r0.97_cp_delta0.02_s20.json \
  --result Logistic=outputs/paper-matrix/appendix-c/textcraft-qwen25/qwen2.5-7b/cascade_probe_L20_r0.97_cp_delta0.02_s20.json \
  --result Stacking=outputs/paper-matrix/appendix-c/textcraft-qwen25/qwen2.5-7b/cascade_stacking_L20_r0.97_cp_delta0.02_s20.json
```

After the Appendix-D fine-grid run, rebuild the violation-count table with:

```bash
python scripts/summarize_margin_sweep.py \
  outputs/paper-matrix/appendix-d/textcraft-qwen25/qwen2.5-7b
```

After the main matrix finishes, rebuild both Supplementary-G diagnostics with:

```bash
python scripts/plot_paper_diagnostics.py
```

Appendix I can be rendered from an exported asynchronous reinforcement-learning
training trace:

```bash
python reproduce.py appendix-i \
  --metrics-csv artifacts/appendix-i/async_vllm_rl_metrics.csv
```

The trace comes from a veRL training run on ALFWorld with
Qwen2.5-3B-Instruct. Training used GRPO with 16 groups, 8 rollouts per
group, and learning rate 1e-6. The run uses veRL's vLLM backend, which
supports direct hidden-state extraction as described in the paper.
This setting illustrates the asynchronous-training advantage: low-promise
trajectories can be stopped early, and the trainer can start the next update
without waiting for all rollouts in the round to finish.

The metrics file is a tidy CSV with columns `wall_time_h`, `step`, `run`, `metric`,
and `value`. Alternatively, provide the validation, global-sequence-length,
average-rounds, and cumulative wall-time metric CSVs:

```bash
python reproduce.py appendix-i \
  --success-csv val_success_rate.csv \
  --tokens-csv global_seqlen_mean.csv \
  --rounds-csv episode_length_mean.csv \
  --wall-time-csv system_step_runtime.csv
```

If the validation export contains a shared step-0 success value, the script places
that point at wall time zero for each displayed run.

For a layout-only render of the same panel structure, run:

```bash
python reproduce.py appendix-i --layout-only
```

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use this code, please cite:

```bibtex
@misc{ruan2026doomed,
  title         = {Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade},
  author        = {Kai Ruan and Zihe Huang and Ziqi Zhou and Qianshan Wei and Jinghao Lin and Xuan Wang and Hao Sun},
  year          = {2026},
  eprint        = {2607.06503},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2607.06503}
}
```
