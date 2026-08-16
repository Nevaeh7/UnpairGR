#!/usr/bin/env python3
"""Build UnpairGR train/evaluation missing-modality semantic-ID indices.

The script is intentionally non-invasive: it never overwrites the original
repository data.  It creates run-local data roots under OUT_BASE and writes new
index files with explicit names such as
``Dataset.index_trainmissing_ot_r25_seed2026.json``.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


DATASETS = ("Arts", "Games", "Instruments")
MODES = ("OT", "OI", "Random")
RATIOS = (25, 50, 75, 100)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    source_data_root: Path
    full_index: str
    text_index: str
    image_index: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path.cwd() / "results" / "missing_modality",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--eval-manifest-root",
        type=Path,
        default=Path.cwd() / "data" / "missing_manifests",
    )
    parser.add_argument("--data-root", type=Path, default=Path.cwd() / "data")
    parser.add_argument(
        "--full-index",
        default=".index_umagr.json",
        help=(
            "Full-route UnpairGR index used when a manifest assigns an item "
            "to the paired/full route."
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--ratios", nargs="+", default=list(map(str, RATIOS)))
    return parser.parse_args()


def method_specs(args: argparse.Namespace) -> list[MethodSpec]:
    return [
        MethodSpec(
            "umagr",
            args.data_root,
            args.full_index,
            ".index_umagr_text.json",
            ".index_umagr_image.json",
        ),
    ]


def dataset_path(root: Path, dataset: str, suffix: str) -> Path:
    return root / dataset / f"{dataset}{suffix}"


def item_sort_key(item_id: str) -> tuple[int, int | str]:
    value = str(item_id)
    return (0, int(value)) if value.isdigit() else (1, value)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def copy_or_link(source: Path, dest: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"missing required source file: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() and dest.resolve() == source.resolve():
            return
        if dest.is_file() and source.is_file() and filecmp.cmp(source, dest, shallow=False):
            return
        raise FileExistsError(
            f"refusing to reuse stale run-local input: {dest}; "
            "choose a new --out-base or remove it explicitly"
        )
    try:
        dest.symlink_to(source)
    except OSError:
        shutil.copy2(source, dest)


def normalize_route(value: str) -> str:
    lowered = str(value).lower()
    if lowered in {"text", "text_only", "ot"}:
        return "text"
    if lowered in {"image", "image_only", "oi"}:
        return "image"
    if lowered in {"full", "paired", "both"}:
        return "full"
    raise ValueError(f"unknown route value: {value!r}")


def train_item_ids(source_data_root: Path, dataset: str) -> list[str]:
    inters = load_json(dataset_path(source_data_root, dataset, ".inter.json"))
    item_ids: set[str] = set()
    for items in inters.values():
        for item in items[:-2]:
            item_ids.add(str(item))
    return sorted(item_ids, key=item_sort_key)


def build_train_manifest(source_data_root: Path, dataset: str, mode: str, ratio: int, seed: int) -> dict:
    ids = train_item_ids(source_data_root, dataset)
    rng = random.Random(f"{seed}:{dataset}:train:{mode}:{ratio}")
    selected_count = round(len(ids) * ratio / 100.0)
    selected = set(rng.sample(ids, selected_count))
    assignments: dict[str, str] = {}
    for item_id in ids:
        if item_id not in selected:
            assignments[item_id] = "full"
        elif mode == "OT":
            assignments[item_id] = "text"
        elif mode == "OI":
            assignments[item_id] = "image"
        else:
            assignments[item_id] = "text" if rng.random() < 0.5 else "image"
    return {
        "dataset": dataset,
        "split": "train",
        "mode": mode,
        "ratio": ratio,
        "seed": seed,
        "item_scope": "training items from inter.json items[:-2]",
        "assignments": assignments,
    }


def load_eval_manifest(
    eval_manifest_root: Path,
    dataset: str,
    mode: str,
    ratio: int,
    seed: int,
) -> dict:
    path = eval_manifest_root / (
        f"{dataset.lower()}_{mode.lower()}_r{ratio}_seed{seed}.manifest.json"
    )
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    expected_mode = {
        "OT": "text_only",
        "OI": "image_only",
        "Random": "random_missing",
    }[mode]
    expected_metadata = {
        "dataset": dataset,
        "missing_mode": expected_mode,
        "missing_ratio": ratio,
        "seed": seed,
    }
    for field, expected in expected_metadata.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"manifest metadata mismatch for {field}: "
                f"expected {expected!r}, got {payload.get(field)!r} ({path})"
            )
    raw_assignments = payload.get("assignments")
    if not isinstance(raw_assignments, dict) or not raw_assignments:
        raise ValueError(f"manifest assignments must be a non-empty object: {path}")
    assignments = {
        str(item_id): normalize_route(route)
        for item_id, route in raw_assignments.items()
    }
    allowed_routes = {
        "OT": {"full", "text"},
        "OI": {"full", "image"},
        "Random": {"full", "text", "image"},
    }[mode]
    unexpected_routes = set(assignments.values()) - allowed_routes
    if unexpected_routes:
        raise ValueError(f"unexpected routes {sorted(unexpected_routes)} in {path}")
    counts = route_counts(assignments)
    selected_count = counts["text"] + counts["image"]
    total_items = len(assignments)
    expected_selected = round(total_items * ratio / 100.0)
    if selected_count != expected_selected:
        raise ValueError(
            f"manifest ratio mismatch: expected {expected_selected} selected "
            f"items, got {selected_count} ({path})"
        )
    if payload.get("selected_count") != selected_count:
        raise ValueError(f"selected_count mismatch in {path}")
    if payload.get("total_items") != total_items:
        raise ValueError(f"total_items mismatch in {path}")
    if payload.get("assignment_counts") != counts:
        raise ValueError(f"assignment_counts mismatch in {path}")
    return {
        "dataset": dataset,
        "split": "eval",
        "mode": mode,
        "ratio": ratio,
        "seed": seed,
        "item_scope": f"Table2 fixed eval manifest: {path}",
        "assignments": assignments,
        "source_manifest": str(path),
    }


def route_counts(assignments: dict[str, str]) -> dict[str, int]:
    counts = {"full": 0, "text": 0, "image": 0}
    for route in assignments.values():
        counts[route] = counts.get(route, 0) + 1
    return counts


def build_index_for_manifest(
    spec: MethodSpec,
    dataset: str,
    manifest: dict,
    output_data_root: Path,
    output_index_suffix: str,
) -> tuple[Path, dict[str, int]]:
    source_dir = spec.source_data_root / dataset
    output_dir = output_data_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    for suffix in {".inter.json", ".item.json", spec.full_index, spec.text_index, spec.image_index}:
        source = source_dir / f"{dataset}{suffix}"
        if not source.is_file():
            raise FileNotFoundError(f"{spec.name} {dataset}: missing required source file {source}")
        copy_or_link(source, output_dir / source.name)

    full = load_json(dataset_path(spec.source_data_root, dataset, spec.full_index))
    text = load_json(dataset_path(spec.source_data_root, dataset, spec.text_index))
    image = load_json(dataset_path(spec.source_data_root, dataset, spec.image_index))
    assignments = manifest["assignments"]
    if not assignments:
        raise ValueError(f"empty {manifest['split']} assignments for {dataset}")
    assignment_ids = set(assignments)
    full_ids = {str(item_id) for item_id in full}
    unknown_ids = assignment_ids - full_ids
    if unknown_ids:
        raise ValueError(
            f"{len(unknown_ids)} manifest item IDs are absent from the full index "
            f"for {dataset}"
        )
    if manifest["split"] == "eval" and assignment_ids != full_ids:
        raise ValueError(
            f"evaluation manifest must cover the full item catalog for {dataset}: "
            f"manifest={len(assignment_ids)} full_index={len(full_ids)}"
        )
    mixed = {}
    for item_id in sorted(full, key=item_sort_key):
        route = assignments.get(str(item_id), "full")
        if route == "text":
            mixed[item_id] = text[item_id]
        elif route == "image":
            mixed[item_id] = image[item_id]
        else:
            mixed[item_id] = full[item_id]

    output_path = output_dir / f"{dataset}{output_index_suffix}"
    write_json(output_path, mixed)
    return output_path, route_counts({item_id: assignments.get(str(item_id), "full") for item_id in full})


def main() -> int:
    args = parse_args()
    ratios = [int(r) for r in args.ratios]
    invalid = [r for r in ratios if r not in RATIOS]
    if invalid:
        raise SystemExit(f"unsupported ratios: {invalid}")

    out_base: Path = args.out_base
    manifest_root = out_base / "manifests"
    audit_rows: list[dict[str, str | int]] = []

    for dataset in args.datasets:
        train_manifests = {}
        eval_manifests = {}
        for mode in args.modes:
            for ratio in ratios:
                train_manifest = build_train_manifest(
                    args.data_root, dataset, mode, ratio, args.seed
                )
                train_path = manifest_root / f"{dataset.lower()}_train_{mode.lower()}_r{ratio}_seed{args.seed}.manifest.json"
                write_json(train_path, train_manifest)
                train_manifests[(mode, ratio)] = (train_manifest, train_path)

                eval_manifest = load_eval_manifest(
                    args.eval_manifest_root, dataset, mode, ratio, args.seed
                )
                eval_path = manifest_root / f"{dataset.lower()}_eval_{mode.lower()}_r{ratio}_seed{args.seed}.manifest.json"
                write_json(eval_path, eval_manifest)
                eval_manifests[(mode, ratio)] = (eval_manifest, eval_path)

        for spec in method_specs(args):
            data_root = out_base / "data_roots" / spec.name / "data"
            for mode in args.modes:
                for ratio in ratios:
                    for split, manifest_map in (("train", train_manifests), ("eval", eval_manifests)):
                        manifest, manifest_path = manifest_map[(mode, ratio)]
                        suffix = f".index_{split}missing_{mode.lower()}_r{ratio}_seed{args.seed}.json"
                        output_path, counts = build_index_for_manifest(spec, dataset, manifest, data_root, suffix)
                        total = sum(counts.values())
                        audit_rows.append(
                            {
                                "method": spec.name,
                                "dataset": dataset,
                                "split": split,
                                "mode": mode,
                                "ratio": ratio,
                                "seed": args.seed,
                                "data_root": str(data_root),
                                "index_file": suffix,
                                "output_path": str(output_path),
                                "manifest": str(manifest_path),
                                "full_count": counts.get("full", 0),
                                "text_count": counts.get("text", 0),
                                "image_count": counts.get("image", 0),
                                "total_count": total,
                                "full_index": spec.full_index,
                                "text_index": spec.text_index,
                                "image_index": spec.image_index,
                            }
                        )

    audit_path = out_base / "train_time_missing_index_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"wrote audit: {audit_path}")
    print(f"rows: {len(audit_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
