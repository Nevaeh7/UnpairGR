#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-"$ROOT/data"}
OUT_BASE=${OUT_BASE:-"$ROOT/outputs"}
GPU=${GPU:-0}

TARGET_DATASETS=Instruments,Arts,Games
PRETRAIN_DATASETS=Pet,Cell,Automotive,Tools,Toys,Sports
ALL_DATASETS="$TARGET_DATASETS,$PRETRAIN_DATASETS"

TOKENIZER_OUT=${TOKENIZER_OUT:-"$OUT_BASE/tokenizer"}
PRETRAIN_OUT=${PRETRAIN_OUT:-"$OUT_BASE/pretrain"}
FINETUNE_ROOT=${FINETUNE_ROOT:-"$OUT_BASE/fully_observed"}
NUM_BEAMS=${NUM_BEAMS:-20}
FP16=${FP16:-1}

RUN_TOKENIZER=${RUN_TOKENIZER:-1}
RUN_INDEX=${RUN_INDEX:-1}
RUN_PRETRAIN=${RUN_PRETRAIN:-1}
RUN_FINETUNE=${RUN_FINETUNE:-1}
RUN_EVAL=${RUN_EVAL:-1}

mkdir -p "$TOKENIZER_OUT" "$PRETRAIN_OUT" "$FINETUNE_ROOT"

log() {
  printf '[UnpairGR fully observed] %s\n' "$*"
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
    metrics = json.load(handle).get("mean_results", {})
required = {"hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"}
if not required.issubset(metrics):
    raise SystemExit(1)
for name in required:
    value = metrics[name]
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise SystemExit(f"invalid metric {name}: {value!r}")
PY
}

if [[ "$RUN_TOKENIZER" == "1" ]]; then
  log "train tokenizer"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/index/main_umagr.py" \
    --device cuda \
    --data_root "$DATA_ROOT" \
    --paired_datasets "$TARGET_DATASETS" \
    --ckpt_dir "$TOKENIZER_OUT" \
    --reliability_epsilon 1e-6
fi

TOKENIZER_CKPT="$TOKENIZER_OUT/best_umagr_model.pth"
if [[ "$RUN_TOKENIZER" == "1" || "$RUN_INDEX" == "1" ]] \
  && [[ ! -s "$TOKENIZER_CKPT" ]]; then
  echo "missing tokenizer checkpoint: $TOKENIZER_CKPT" >&2
  exit 1
fi

if [[ "$RUN_INDEX" == "1" ]]; then
  log "generate paired semantic IDs"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/index/generate_indices_umagr.py" \
    --datasets "$ALL_DATASETS" \
    --ckpt_path "$TOKENIZER_CKPT" \
    --data_root "$DATA_ROOT" \
    --device cuda \
    --input_mode paired \
    --fuse_mode consensus \
    --output_file .index_umagr.json

  for mode in text image; do
    log "generate $mode-only semantic IDs"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/index/generate_indices_umagr.py" \
      --datasets "$TARGET_DATASETS" \
      --ckpt_path "$TOKENIZER_CKPT" \
      --data_root "$DATA_ROOT" \
      --device cuda \
      --input_mode "$mode" \
      --output_file ".index_umagr_${mode}.json"
  done
fi

if [[ "$RUN_PRETRAIN" == "1" ]]; then
  pretrain_args=(
    --device cuda
    --data_path "$DATA_ROOT"
    --pretrain_datasets "$PRETRAIN_DATASETS"
    --output_dir "$PRETRAIN_OUT"
    --base_model "$ROOT/config/ckpt"
    --index_file .index_umagr.json
    --tasks seqrec
    --valid_task seqrec
    --per_device_batch_size 512
    --learning_rate 1e-3
    --epochs 30
  )
  if [[ "$FP16" == "1" ]]; then
    pretrain_args+=(--fp16)
  fi
  log "pre-train recommender"
  CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled \
    "$PY" "$ROOT/pretrain.py" "${pretrain_args[@]}"
fi

if [[ "$RUN_PRETRAIN" == "1" || "$RUN_FINETUNE" == "1" ]] \
  && ! has_model_checkpoint "$PRETRAIN_OUT"; then
  echo "missing pre-trained recommender: $PRETRAIN_OUT" >&2
  exit 1
fi

if [[ "$RUN_FINETUNE" == "1" || "$RUN_EVAL" == "1" ]]; then
  for dataset in Arts Games Instruments; do
    output="$FINETUNE_ROOT/$dataset"
    result="$output/results.json"
    mkdir -p "$output"

    if [[ "$RUN_FINETUNE" == "1" ]]; then
      finetune_args=(
        --device cuda
        --data_path "$DATA_ROOT"
        --dataset "$dataset"
        --output_dir "$output"
        --load_model_name "$PRETRAIN_OUT"
        --index_file .index_umagr.json
        --text_index_file .index_umagr_text.json
        --image_index_file .index_umagr_image.json
        --tasks seqrec
        --valid_task seqrec
        --modal_mix_prob 0
        --fusion_source_modal_mix_prob 0
        --per_device_batch_size 512
        --learning_rate 5e-4
        --epochs 200
        --num_beams "$NUM_BEAMS"
      )
      if [[ "$FP16" == "1" ]]; then
        finetune_args+=(--fp16)
      fi
      log "fine-tune $dataset"
      CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled \
        "$PY" "$ROOT/finetune.py" "${finetune_args[@]}"
    fi

    if ! has_model_checkpoint "$output"; then
      echo "missing fine-tuned recommender: $output" >&2
      exit 1
    fi

    if [[ "$RUN_EVAL" == "1" ]]; then
      log "evaluate $dataset"
      CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled \
        "$PY" "$ROOT/evaluate_recommender.py" \
        --device cuda \
        --ckpt_path "$output" \
        --data_path "$DATA_ROOT" \
        --dataset "$dataset" \
        --index_file .index_umagr.json \
        --text_index_file .index_umagr_text.json \
        --image_index_file .index_umagr_image.json \
        --tasks seqrec \
        --test_task seqrec \
        --num_beams "$NUM_BEAMS" \
        --filter_items \
        --results_file "$result" \
        --save_file "$output/predictions.json"
      valid_result "$result"
    fi
  done
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  FINETUNE_ROOT="$FINETUNE_ROOT" "$PY" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["FINETUNE_ROOT"])
rows = []
for dataset in ("Arts", "Games", "Instruments"):
    path = root / dataset / "results.json"
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)["mean_results"]
    rows.append(
        {
            "dataset": dataset,
            "hr1": metrics["hit@1"],
            "hr5": metrics["hit@5"],
            "hr10": metrics["hit@10"],
            "ndcg5": metrics["ndcg@5"],
            "ndcg10": metrics["ndcg@10"],
            "result_path": str(path),
        }
    )

output = root / "unpairgr_fully_observed_results.csv"
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print(output)
PY
  log "done: $FINETUNE_ROOT/unpairgr_fully_observed_results.csv"
else
  log "done (evaluation and CSV summary disabled)"
fi
