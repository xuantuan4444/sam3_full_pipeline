#!/usr/bin/env python3
"""
export_analysis_pack.py -- package everything the offline analysis needs into one zip per dataset,
so the next steps (comparing class-selection criteria, oracle / random-K, fair metrics, choosing
positive / hard-negative training images) run on any laptop: CPU only, no SAM3, no GPU, no images.

run_pipeline_voc.py calls it automatically when a run finishes. It can also be run by hand
(CPU only, a few minutes):

    python tools/export_analysis_pack.py            # -> outputs/analysis_pack_voc/ + outputs/analysis_pack_voc.zip
    python tools/export_analysis_pack.py --no-zip

Contents of analysis_pack_voc/:
    meta.json                         classes, label conventions, target classes, local-id mapping
                                      of every evaluated checkpoint, what is included
    val/<arm>/<id>.npz                per val image (eval label space), only for arms whose hybrid
                                      val saved per-image predictions:
                                        gt            (H, W) uint8  GT (ignore = meta.eval_ignore)
                                        sam3_pred     (H, W) uint8  SAM3 baseline prediction
                                        local_<arch>  (H, W) uint8  refine output, local ids (0 = other)
    train_gt_pixels.csv               per train image: internal split + GT pixels of EVERY class
    train_coarse_pixels_<arm>.csv     per train image: SAM3 coarse-mask pixels of every target class
    results/                          baseline + hybrid metrics (CSV, confusion_matrix.npy, manifest),
                                      checkpoint history.json / manifest.json, split.json,
                                      target_classes.json, ablation_table.csv
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voc_common as C  # noqa: E402

RESULT_FILES = ("summary_metrics.csv", "per_class_metrics.csv", "confusion_matrix.npy", "manifest.json")


def copy_if_exists(src, dst):
    src = Path(src)
    if not src.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return 1


def complete(d):
    m = C.read_manifest(d)
    return bool(m) and m.get("status") == "complete"


def export_val(paths, arm, out, val_ids, meta):
    """One npz per val image: gt, sam3_pred and the local prediction of every evaluated arch."""
    archs = []
    for arch in C.ARCHS:
        m = C.read_manifest(paths.val_dir(arch))
        if m and m.get("saved_predictions") and (paths.val_dir(arch) / "pred_local").exists():
            archs.append(arch)
            meta["arms"][arm]["archs"][arch] = {
                "targets": m.get("targets"), "target_to_local": m.get("target_to_local"),
                "checkpoint_epoch": m.get("checkpoint_epoch"), "metric_version": m.get("metric_version"),
            }
    if not archs:
        print(f"  [{arm}] no hybrid val with saved predictions -> per-image val data not exported")
        return 0
    if not complete(paths.val_cache):
        print(f"  [{arm}] val_cache incomplete -> per-image val data not exported")
        return 0
    dst = out / "val" / arm
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_id in val_ids:
        key = C.cache_key(img_id)
        cpath = paths.val_cache / f"{key}.npz"
        if not cpath.exists() or C.gt_path(img_id, "val") is None:
            continue
        with np.load(cpath) as d:
            item = {"gt": C.load_gt_eval(img_id, "val"), "sam3_pred": d["sam3_pred"]}
        for arch in archs:
            pp = paths.val_dir(arch) / "pred_local" / f"{key}.png"
            if pp.exists():
                item[f"local_{arch}"] = np.array(Image.open(pp), dtype=np.uint8)
        np.savez_compressed(dst / f"{key}.npz", **item)
        n += 1
    meta["arms"][arm]["n_val_images"] = n
    print(f"  [{arm}] val: {n} images, refine predictions of {archs}")
    return n


def export_train_stats(arms_paths, out, train_ids, meta):
    """Per train image: GT pixel count of every class (+ internal split), and per arm the
    coarse-mask pixel count of every target class -> positive / hard-negative image selection."""
    import pandas as pd
    split = {}
    any_paths = next(iter(arms_paths.values()))
    if any_paths.split_json.exists():
        tr, va = C.read_split(any_paths.split_json)
        split.update({i: "internal_train" for i in tr})
        split.update({i: "internal_val" for i in va})
    rows = []
    for img_id in train_ids:
        if C.gt_path(img_id, "train") is None:
            continue
        raw = C.load_gt_train_raw(img_id, "train")
        valid = C.train_valid_mask(raw).astype(bool)
        counts = np.bincount(raw[valid].ravel(), minlength=256)
        row = {"image_id": img_id, "split": split.get(img_id, "unused"),
               "height": raw.shape[0], "width": raw.shape[1], "valid_px": int(valid.sum())}
        for c in C.CLASSES:
            row[f"gt__{c}"] = int(counts[C.TRAIN_RAW_VALUE[c]])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "train_gt_pixels.csv", index=False)
    print(f"  train_gt_pixels.csv: {len(rows)} images x {len(C.CLASSES)} classes")

    for arm, paths in arms_paths.items():
        m = C.read_manifest(paths.coarse_cache)
        if not (m and m.get("status") == "complete"):
            print(f"  [{arm}] coarse_cache incomplete -> train_coarse_pixels_{arm}.csv not exported")
            continue
        targets = (m.get("prompts") or {}).get("target_classes") or C.load_target_classes()
        crows = []
        for img_id in train_ids:
            p = paths.coarse_cache / f"{C.cache_key(img_id)}.npz"
            if not p.exists():
                continue
            with np.load(p) as d:
                r = {"image_id": img_id}
                for c in targets:
                    r[f"coarse__{c}"] = int(d[c].sum()) if c in d.files else -1
            crows.append(r)
        pd.DataFrame(crows).to_csv(out / f"train_coarse_pixels_{arm}.csv", index=False)
        meta["arms"][arm]["coarse_targets"] = targets
        print(f"  train_coarse_pixels_{arm}.csv: {len(crows)} images x {len(targets)} target classes")


def export_results(out_root, arms_paths, out):
    res = out / "results"
    n = 0
    for arm, paths in arms_paths.items():
        for f in RESULT_FILES:
            n += copy_if_exists(paths.baseline_dir / f, res / arm / "baseline" / f)
        for arch in C.ARCHS:
            for f in RESULT_FILES:
                n += copy_if_exists(paths.val_dir(arch) / f, res / arm / arch / "val" / f)
            for f in ("history.json", "manifest.json", "training_curves.png"):
                n += copy_if_exists(paths.ckpt_dir(arch) / f, res / arm / arch / "checkpoint" / f)
        n += copy_if_exists(paths.arm_dir / "class_pixel_stats.json", res / arm / "class_pixel_stats.json")
    for f in ("split.json", "ablation_table.csv"):
        n += copy_if_exists(out_root / f, res / f)
    n += copy_if_exists(C.TARGET_CLASSES_JSON, res / "target_classes.json")
    print(f"  results/: {n} files")


def main():
    ap = argparse.ArgumentParser(description="Package per-image predictions + stats for offline analysis (CPU).")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve() if args.out_root else (C.SMOKE_OUT_ROOT if args.limit else C.DEFAULT_OUT_ROOT)
    name = f"analysis_pack_{C.DATASET}"
    final, tmp = out_root / name, out_root / f"{name}.tmp"
    t0 = time.time()
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    arms_paths = {arm: C.Paths(arm, out_root) for arm in C.ARMS if C.Paths(arm, out_root).arm_dir.exists()}
    if not arms_paths:
        sys.exit(f"No outputs found under {out_root}")
    meta = {
        "dataset": C.DATASET, "title": C.DATASET_TITLE, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": C.CLASSES, "class_to_eval_index": C.CLASS_TO_EVAL_INDEX,
        "eval_num_classes": C.EVAL_NUM_CLASSES, "eval_ignore": C.EVAL_IGNORE,
        "eval_no_prediction": C.EVAL_NO_PREDICTION, "eval_metric_indices": C.EVAL_METRIC_INDICES,
        "eval_extra_names": {str(k): v for k, v in C.EVAL_EXTRA_NAMES.items()},
        "metric_version": C.METRIC_VERSION, "target_classes": C.load_target_classes(),
        "train_raw_value": C.TRAIN_RAW_VALUE,
        "arms": {arm: {"archs": {}} for arm in arms_paths},
        "notes": "local_<arch>: 0 = other, k = the target with target_to_local[target] == k "
                 "(arms.<arm>.archs.<arch>). Hybrid = sam3_pred overwritten with class_to_eval_index[target] "
                 "wherever local_<arch> == k. Metrics: confusion matrix [K, K+1], last column = no prediction "
                 "(counted as FN of the GT class), GT == eval_ignore skipped.",
    }
    print(f"Exporting analysis pack for {C.DATASET_TITLE} ({list(arms_paths)}) -> {final}")
    val_ids = C.get_ids("val", args.limit)
    for arm, paths in arms_paths.items():
        export_val(paths, arm, tmp, val_ids, meta)
    export_train_stats(arms_paths, tmp, C.get_ids("train", args.limit), meta)
    export_results(out_root, arms_paths, tmp)
    with open(tmp / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    size_mb = sum(p.stat().st_size for p in final.rglob("*") if p.is_file()) / 2**20
    print(f"Pack folder: {final}  ({size_mb:.0f} MB)")
    if not args.no_zip:
        z = shutil.make_archive(str(out_root / name), "zip", root_dir=str(out_root), base_dir=name)
        print(f"Pack zip   : {z}  ({Path(z).stat().st_size / 2**20:.0f} MB)  <- download this file")
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
