#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-20}"
TARGETS_QWEN25="${TARGETS_QWEN25:-0.90 0.92 0.95 0.97}"
TARGETS_QWEN3="${TARGETS_QWEN3:-0.90 0.92 0.95 0.97}"

ALFWORLD_DATA="${ALFWORLD_DATA:-}"
QWEN25_MODEL="${QWEN25_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
QWEN3_MODEL="${QWEN3_MODEL:-Qwen/Qwen3-1.7B}"

TASK_IDS_FILE="${TASK_IDS_FILE:-$ROOT/artifacts/alfworld/task_ids_120.txt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/alfworld}"
CASCADE_ROOT="$OUT_ROOT/cascade_root_k4"
LOGDIR="${LOGDIR:-$OUT_ROOT/logs}"
mkdir -p "$LOGDIR"

if [[ -n "$ALFWORLD_DATA" ]]; then
  export ALFWORLD_DATA
fi

usage() {
  sed -n '1,22p' "$0"
}

wait_server() {
  local port="$1"
  "$PYTHON" - "$port" <<'PY'
import sys, time, urllib.request
port = int(sys.argv[1])
base = f"http://127.0.0.1:{port}"
for i in range(160):
    try:
        print(port, urllib.request.urlopen(base + "/", timeout=1).read().decode())
        raise SystemExit(0)
    except Exception:
        if i == 159:
            raise
        time.sleep(0.25)
PY
}

start_servers() {
  local prefix="$1"
  shift
  SERVER_PIDS=()
  SERVER_BASES=()
  for port in "$@"; do
    local log="$LOGDIR/${prefix}_${port}.log"
    "$PYTHON" -c 'from agentenv_alfworld.launch import launch; launch()' \
      --host 127.0.0.1 --port "$port" >"$log" 2>&1 &
    SERVER_PIDS+=("$!")
    SERVER_BASES+=("http://127.0.0.1:$port")
  done
  cleanup_servers() {
    for pid in "${SERVER_PIDS[@]:-}"; do
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    done
  }
  trap cleanup_servers EXIT
  for port in "$@"; do
    wait_server "$port"
  done
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

run_rollout() {
  local slug="$1" model="$2" out="$3" base_port="$4" n_gpus="${5:-4}" concurrency="${6:-2}" gpu_mem="${7:-0.75}"
  if [[ -z "$ALFWORLD_DATA" ]]; then
    echo "ALFWORLD_DATA must point to the local ALFWorld data directory" >&2
    exit 2
  fi
  if [[ ! -f "$TASK_IDS_FILE" ]]; then
    echo "TASK_IDS_FILE does not exist: $TASK_IDS_FILE" >&2
    exit 2
  fi
  local ports=()
  for ((i=0; i<n_gpus; i++)); do
    ports+=("$((base_port + i))")
  done
  mkdir -p "$(dirname "$out")"
  start_servers "alfworld_${slug}" "${ports[@]}"
  local bases
  bases="$(join_by_comma "${SERVER_BASES[@]}")"
  "$PYTHON" scripts/rollout_vllm.py \
    --env alfworld \
    --server_base "$bases" \
    --model_path "$model" \
    --out "$out" \
    --n_tasks 2620 \
    --task_ids_file "$TASK_IDS_FILE" \
    --k_rollouts 4 \
    --temperature 0.7 \
    --max_rounds 30 \
    --max_new_tokens 128 \
    --http_timeout 240 \
    --rollout_timeout 1200 \
    --seed 20260713 \
    --resume 1 \
    --n_gpus "$n_gpus" \
    --concurrency "$concurrency" \
    --max_model_len 8192 \
    --gpu_mem_util "$gpu_mem"
}

run_features() {
  local slug="$1" model="$2" rollout="$3" out_dir="$4"
  if [[ -e "$out_dir" ]]; then
    echo "feature output already exists; move it aside first: $out_dir" >&2
    exit 2
  fi
  "$PYTHON" src/features.py \
    --env alfworld \
    --rollouts "$rollout" \
    --model_path "$model" \
    --out_dir "$out_dir" \
    --layers 0,4,8,12,16,20,24,28 \
    --num_shards 1 \
    --shard_id 0 \
    --device cuda:0 \
    --skip_anchor_failures
}

run_probe() {
  local feat_dir="$1" out_json="$2"
  "$PYTHON" src/probe.py \
    --feat_dir "$feat_dir" \
    --out "$out_json" \
    --n_splits 5
}

setup_cascade_root() {
  mkdir -p "$CASCADE_ROOT/qwen2.5-7b" "$CASCADE_ROOT/qwen3-1.7b"
  for path in \
    "$CASCADE_ROOT/qwen2.5-7b/features" "$CASCADE_ROOT/qwen2.5-7b/rollouts.jsonl" \
    "$CASCADE_ROOT/qwen3-1.7b/features" "$CASCADE_ROOT/qwen3-1.7b/rollouts.jsonl"; do
    if [[ -L "$path" ]]; then
      unlink "$path"
    elif [[ -e "$path" ]]; then
      echo "refusing to replace non-symlink path: $path" >&2
      exit 2
    fi
  done
  ln -s ../../qwen2.5-7b-k4/features_stride4 "$CASCADE_ROOT/qwen2.5-7b/features"
  ln -s ../../qwen2.5-7b-k4/rollouts.jsonl "$CASCADE_ROOT/qwen2.5-7b/rollouts.jsonl"
  ln -s ../../qwen3-1.7b-k4/features_stride4 "$CASCADE_ROOT/qwen3-1.7b/features"
  ln -s ../../qwen3-1.7b-k4/rollouts.jsonl "$CASCADE_ROOT/qwen3-1.7b/rollouts.jsonl"
}

run_cascade_targets() {
  local model="$1" layer="$2" scorer="$3"
  shift 3
  setup_cascade_root
  for target in "$@"; do
    "$PYTHON" scripts/stacking_cascade.py \
      --results-root "$CASCADE_ROOT" \
      --output-results-root "$CASCADE_ROOT" \
      --model "$model" \
      --layer "$layer" \
      --mode cascade \
      --scorer "$scorer" \
      --cal-method cp \
      --target-recall "$target" \
      --seeds "$SEEDS" \
      --margin-mode delta \
      --margin-delta 0.02
  done
}

write_summary() {
  OUT_ROOT="$OUT_ROOT" "$PYTHON" - <<'PY'
import csv, json, os, pathlib, re, statistics

rows = []
configs = [
    ("qwen2.5-7b", 28, "probe"),
    ("qwen3-1.7b", 20, "probe"),
    ("qwen3-1.7b", 20, "stacking"),
]
for model, layer, scorer in configs:
    root = pathlib.Path(os.environ["OUT_ROOT"]) / "cascade_root_k4" / model
    for p in sorted(root.glob(f"cascade_{scorer}_L{layer}_r*_cp_delta0.02_s20.json")):
        target = float(re.search(r"_r([0-9.]+)_", p.name).group(1))
        d = json.loads(p.read_text())
        def mean_std(section, key):
            vals = d[section][key]
            return sum(vals) / len(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0
        rec, recs = mean_std("searched_best", "recall")
        sav, savs = mean_std("searched_best", "saved")
        srec, _ = mean_std("best_single_gate", "recall")
        ssav, _ = mean_std("best_single_gate", "saved")
        rows.append([model, scorer, layer, target, rec, recs, sav * 100, savs * 100, srec, ssav * 100, str(p)])

out = pathlib.Path(os.environ["OUT_ROOT"]) / "alfworld_k4_cascade_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "model", "scorer", "layer", "target", "cascade_recall_mean",
        "cascade_recall_std", "cascade_saved_pct_mean", "cascade_saved_pct_std",
        "single_recall_mean", "single_saved_pct_mean", "file",
    ])
    w.writerows(sorted(rows))
print(out)
print(out.read_text())
PY
}

case "${1:-}" in
  help|-h|--help)
    usage
    exit 0
    ;;
  rollout-qwen25)
    run_rollout qwen25 "$QWEN25_MODEL" "$OUT_ROOT/qwen2.5-7b-k4/rollouts.jsonl" 36230 4 2 0.75
    ;;
  rollout-qwen3)
    export AGENT_EARLY_ABORT_ENABLE_THINKING=0
    run_rollout qwen3 "$QWEN3_MODEL" "$OUT_ROOT/qwen3-1.7b-k4/rollouts.jsonl" 36240 4 2 0.55
    ;;
  features-qwen25)
    unset AGENT_EARLY_ABORT_ENABLE_THINKING || true
    run_features qwen25 "$QWEN25_MODEL" "$OUT_ROOT/qwen2.5-7b-k4/rollouts.jsonl" "$OUT_ROOT/qwen2.5-7b-k4/features_stride4"
    ;;
  features-qwen3)
    export AGENT_EARLY_ABORT_ENABLE_THINKING=0
    run_features qwen3 "$QWEN3_MODEL" "$OUT_ROOT/qwen3-1.7b-k4/rollouts.jsonl" "$OUT_ROOT/qwen3-1.7b-k4/features_stride4"
    ;;
  probe-qwen25)
    run_probe "$OUT_ROOT/qwen2.5-7b-k4/features_stride4" "$OUT_ROOT/qwen2.5-7b-k4/probe_layers_stride4.json"
    ;;
  probe-qwen3)
    run_probe "$OUT_ROOT/qwen3-1.7b-k4/features_stride4" "$OUT_ROOT/qwen3-1.7b-k4/probe_layers_stride4.json"
    ;;
  cascade-qwen25)
    run_cascade_targets qwen2.5-7b 28 probe $TARGETS_QWEN25
    ;;
  cascade-qwen3)
    run_cascade_targets qwen3-1.7b 20 probe $TARGETS_QWEN3
    ;;
  summary)
    write_summary
    ;;
  all-analysis)
    run_features qwen25 "$QWEN25_MODEL" "$OUT_ROOT/qwen2.5-7b-k4/rollouts.jsonl" "$OUT_ROOT/qwen2.5-7b-k4/features_stride4"
    export AGENT_EARLY_ABORT_ENABLE_THINKING=0
    run_features qwen3 "$QWEN3_MODEL" "$OUT_ROOT/qwen3-1.7b-k4/rollouts.jsonl" "$OUT_ROOT/qwen3-1.7b-k4/features_stride4"
    unset AGENT_EARLY_ABORT_ENABLE_THINKING || true
    run_probe "$OUT_ROOT/qwen2.5-7b-k4/features_stride4" "$OUT_ROOT/qwen2.5-7b-k4/probe_layers_stride4.json"
    export AGENT_EARLY_ABORT_ENABLE_THINKING=0
    run_probe "$OUT_ROOT/qwen3-1.7b-k4/features_stride4" "$OUT_ROOT/qwen3-1.7b-k4/probe_layers_stride4.json"
    unset AGENT_EARLY_ABORT_ENABLE_THINKING || true
    run_cascade_targets qwen2.5-7b 28 probe $TARGETS_QWEN25
    run_cascade_targets qwen3-1.7b 20 probe $TARGETS_QWEN3
    write_summary
    ;;
  *)
    usage
    exit 2
    ;;
esac
