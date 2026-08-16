#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-"$ROOT/data"}
MANIFEST_ROOT=${MANIFEST_ROOT:-"$DATA_ROOT/missing_manifests"}
OUT_BASE=${OUT_BASE:-"$ROOT/outputs/missing_modality"}
FULLY_OBSERVED_ROOT=${FULLY_OBSERVED_ROOT:-"$ROOT/outputs/fully_observed"}

DATASETS=${DATASETS:-"Arts Games Instruments"}
MODES=${MODES:-"OT OI Random"}
RATIOS=${RATIOS:-"25 50 75 100"}
SEED=2026
GPU=${GPU:-0}

FULL_INDEX=${FULL_INDEX:-.index_umagr.json}
NUM_BEAMS=${NUM_BEAMS:-20}
MAX_STEPS=${MAX_STEPS:-2000}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
FP16=${FP16:-1}

RUN_BUILD=${RUN_BUILD:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_EVAL=${RUN_EVAL:-1}

mkdir -p "$OUT_BASE/logs" "$OUT_BASE/checkpoints" "$OUT_BASE/results"

log() {
  printf '[UnpairGR missing] %s\n' "$*"
}

has_model_checkpoint() {
  local directory="$1"
  [[ -d "$directory" ]] || return 1
  find "$directory" -maxdepth 1 -type f \
    \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) \
    -size +0c -print -quit | grep -q .
}

valid_result() {
  "$PY" - "$1" <<'PY'
import json
import math
import sys
from numbers import Real

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
metrics = payload.get("mean_results", {})
for name in ("hit@10", "ndcg@10"):
    value = metrics.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise SystemExit(f"invalid metric {name}: {value!r}")
PY
}

base_checkpoint() {
  case "$1" in
    Arts|Games|Instruments) printf '%s\n' "$FULLY_OBSERVED_ROOT/$1" ;;
    *) return 2 ;;
  esac
}

if [[ "$RUN_BUILD" == "1" ]]; then
  log "build deterministic train/eval route indices"
  "$PY" "$ROOT/build_missing_indices.py" \
    --out-base "$OUT_BASE" \
    --data-root "$DATA_ROOT" \
    --eval-manifest-root "$MANIFEST_ROOT" \
    --full-index "$FULL_INDEX" \
    --seed "$SEED" \
    --datasets $DATASETS \
    --modes $MODES \
    --ratios $RATIOS
fi

RUN_DATA_ROOT="$OUT_BASE/data_roots/umagr/data"

for dataset in $DATASETS; do
  if [[ "$RUN_TRAIN" == "1" ]]; then
    base=$(base_checkpoint "$dataset")
    if [[ ! -d "$base" ]]; then
      echo "missing fully-observed base checkpoint: $base" >&2
      exit 1
    fi
  fi

  for mode in $MODES; do
    mode_lower=$(printf '%s' "$mode" | tr '[:upper:]' '[:lower:]')
    for ratio in $RATIOS; do
      cell="${mode_lower}_r${ratio}"
      train_index=".index_trainmissing_${cell}_seed${SEED}.json"
      eval_index=".index_evalmissing_${cell}_seed${SEED}.json"
      output_dir="$OUT_BASE/checkpoints/$dataset/$cell"
      result_dir="$OUT_BASE/results/$dataset/$cell"
      result_file="$result_dir/result.json"
      save_file="$result_dir/save.json"

      if [[ "$RUN_TRAIN" == "1" ]]; then
        mkdir -p "$output_dir"
        train_args=(
          --data_path "$RUN_DATA_ROOT"
          --dataset "$dataset"
          --output_dir "$output_dir"
          --device cuda
          --per_device_batch_size "$TRAIN_BATCH_SIZE"
          --learning_rate "$LEARNING_RATE"
          --epochs 1
          --max_steps "$MAX_STEPS"
          --weight_decay 0.01
          --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
          --warmup_ratio 0.01
          --lr_scheduler_type cosine
          --save_and_eval_strategy steps
          --save_and_eval_steps "$MAX_STEPS"
          --logging_step 50
          --max_his_len 20
          --train_data_mode 0
          --prompt_num 4
          --patient 10
          --num_beams "$NUM_BEAMS"
          --metrics hit@1,hit@5,hit@10,ndcg@5,ndcg@10
          --eval_start 0
          --eval_count -1
          --eval_stride 1
          --index_file "$train_index"
          --text_index_file "$train_index"
          --image_index_file .index_umagr_image.json
          --tasks seqrec
          --valid_task seqrec
          --modal_mix_prob 0.0
          --fusion_source_modal_mix_prob 0.0
          --load_model_name "$base"
          --skip_generation_eval
        )
        if [[ "$FP16" == "1" ]]; then
          train_args+=(--fp16)
        fi
        log "train $dataset $mode $ratio"
        CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled \
          "$PY" "$ROOT/finetune.py" "${train_args[@]}" \
          >"$OUT_BASE/logs/train_${dataset}_${mode}_${ratio}.log" 2>&1
        if ! has_model_checkpoint "$output_dir"; then
          echo "training finished without a model checkpoint: $output_dir" >&2
          exit 1
        fi
      fi

      if [[ "$RUN_EVAL" == "1" ]]; then
        if ! has_model_checkpoint "$output_dir"; then
          echo "missing model checkpoint for evaluation: $output_dir" >&2
          exit 1
        fi
        mkdir -p "$result_dir"
        log "evaluate $dataset $mode $ratio"
        CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled \
          "$PY" "$ROOT/evaluate_recommender.py" \
          --ckpt_path "$output_dir" \
          --data_path "$RUN_DATA_ROOT" \
          --dataset "$dataset" \
          --test_batch_size "$TEST_BATCH_SIZE" \
          --num_beams "$NUM_BEAMS" \
          --index_file "$eval_index" \
          --text_index_file "$eval_index" \
          --image_index_file .index_umagr_image.json \
          --tasks seqrec \
          --test_task seqrec \
          --results_file "$result_file" \
          --save_file "$save_file" \
          --filter_items \
          >"$OUT_BASE/logs/eval_${dataset}_${mode}_${ratio}.log" 2>&1
        valid_result "$result_file"
      fi
    done
  done
done

if [[ "$RUN_EVAL" == "1" ]]; then
  OUT_BASE="$OUT_BASE" DATASETS="$DATASETS" MODES="$MODES" RATIOS="$RATIOS" \
    "$PY" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_BASE"])
rows = []
for dataset in os.environ["DATASETS"].split():
    for mode in os.environ["MODES"].split():
        for ratio in os.environ["RATIOS"].split():
            cell = f"{mode.lower()}_r{ratio}"
            path = root / "results" / dataset / cell / "result.json"
            with path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)["mean_results"]
            rows.append(
                {
                    "dataset": dataset,
                    "condition": mode,
                    "ratio": ratio,
                    "hr10": metrics["hit@10"],
                    "ndcg10": metrics["ndcg@10"],
                    "result_path": str(path),
                }
            )

output = root / "unpairgr_missing_results.csv"
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print(output)
PY

  log "done: $OUT_BASE/unpairgr_missing_results.csv"
else
  log "done (evaluation and CSV summary disabled)"
fi
