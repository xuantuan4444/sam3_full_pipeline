#!/usr/bin/env python3
"""
sam3_baseline_voc.py -- STEP 1 of an arm: one SAM3 pass over the official val split.

For every val image it runs SAM3 on ALL classes (target classes with the arm's prompts, every
other class with its bare name) and stores, in outputs/<arm>/val_cache/<id>.npz:
    sam3_pred         (H, W) uint8, eval label space -- the SAM3 baseline prediction
    coarse__<class>   (H, W) uint8 0/1 union-at-threshold mask, one per target class
Then it writes the SAM3 baseline metrics to outputs/<arm>/baseline/.

The two hybrid val steps read this cache instead of re-running SAM3, so SAM3 runs on val once
per arm instead of three times. The prediction is identical to the original baseline script:
every pixel gets the class of the highest-scoring mask with score > 0.30 (ties -> earlier class);
the processor threshold is set to 0.15 only so the low-score masks needed for the coarse
union-at-threshold fallback are returned too.

Resumable: images already in val_cache (same fingerprint) are read back instead of recomputed.

    python sam3_baseline_voc.py --arm llm
"""

import argparse
import time
import traceback

import numpy as np
from PIL import Image
from tqdm import tqdm

from voc_common import (
    CLASSES, COARSE_THRESHOLDS, CONFIDENCE_THRESHOLD, DATASET_TITLE, EVAL_CLASS_OFFSET,
    EVAL_NO_PREDICTION, EVAL_NUM_CLASSES, Paths, add_common_args, banner, build_sam3_processor,
    collect_sample, compute_metrics, cuda_empty_cache, fast_confusion_matrix, fingerprints,
    get_ids, gt_path, image_path, cache_key, is_complete, load_gt_eval, load_prompt_config,
    plot_bar_chart, plot_class_visualizations, prepare_dir, query_class, resolve_out_root,
    resolve_sam3_ckpt, run_main, setup_torch, union_at_threshold, write_manifest,
)


def sam3_full_pass(processor, image, pc):
    """Returns (sam3_pred, {target: coarse mask}) for one PIL image."""
    width, height = image.size
    state = processor.set_image(image)
    sam3_pred = np.full((height, width), EVAL_NO_PREDICTION, dtype=np.uint8)
    best_score = np.zeros((height, width), dtype=np.float32)
    coarse = {}
    targets = set(pc.target_classes)

    for i, cname in enumerate(CLASSES):
        state, masks, scores = query_class(processor, state, pc.prompts_for(cname))
        class_score = np.zeros((height, width), dtype=np.float32)
        class_hit = np.zeros((height, width), dtype=bool)
        for mask, score in zip(masks, scores):
            if score <= CONFIDENCE_THRESHOLD:
                continue
            update = mask & (score > class_score)
            class_score[update] = score
            class_hit[update] = True
        update = class_hit & (class_score > best_score)
        sam3_pred[update] = i + EVAL_CLASS_OFFSET
        best_score[update] = class_score[update]
        if cname in targets:
            coarse[cname], _ = union_at_threshold(masks, scores, (height, width), COARSE_THRESHOLDS)
    return sam3_pred, coarse


def load_cached(path, targets):
    try:
        with np.load(path) as d:
            if "sam3_pred" not in d.files or any(f"coarse__{c}" not in d.files for c in targets):
                return None
            return d["sam3_pred"]
    except Exception:
        return None


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    p.add_argument("--sam3-ckpt", default=None)
    args = p.parse_args()

    out_root = resolve_out_root(args)
    paths = Paths(args.arm, out_root)
    pc = load_prompt_config(args.arm)
    fp = fingerprints(args.arm, args.limit)["baseline"]

    banner(f"SAM3 BASELINE + VAL CACHE -- {DATASET_TITLE}, arm={args.arm}")
    pc.print_table()
    if is_complete(paths.baseline_dir, fp) and not args.force:
        print(f"[skip] already complete: {paths.baseline_dir}")
        return

    prepare_dir(paths.val_cache, "val_cache", fp, force=args.force, resumable=True)
    prepare_dir(paths.baseline_dir, "baseline", fp, force=args.force)

    device = setup_torch(global_bf16_autocast=True)
    val_ids = get_ids("val", args.limit)
    print(f"Val images: {len(val_ids)}   cache: {paths.val_cache}")

    processor = None
    conf_mat = np.zeros((EVAL_NUM_CLASSES, EVAL_NUM_CLASSES), dtype=np.int64)
    class_samples = {}
    n_cached = n_new = 0
    skipped, errors = [], []
    t_start = time.time()

    for n, img_id in enumerate(tqdm(val_ids, desc=f"SAM3 val pass ({args.arm})", dynamic_ncols=True), 1):
        ip = image_path(img_id, "val")
        if not ip.exists() or gt_path(img_id, "val") is None:
            skipped.append(img_id)
            continue
        cpath = paths.val_cache / f"{cache_key(img_id)}.npz"
        try:
            gt = load_gt_eval(img_id, "val")
            pred = load_cached(cpath, pc.target_classes) if cpath.exists() else None
            if pred is not None:
                n_cached += 1
            else:
                if processor is None:
                    _, processor = build_sam3_processor(resolve_sam3_ckpt(args.sam3_ckpt), device,
                                                        confidence_threshold=min(COARSE_THRESHOLDS))
                image = Image.open(ip).convert("RGB")
                pred, coarse = sam3_full_pass(processor, image, pc)
                tmp = cpath.with_name(cpath.stem + ".tmp.npz")
                np.savez_compressed(tmp, sam3_pred=pred, **{f"coarse__{c}": coarse[c] for c in pc.target_classes})
                tmp.replace(cpath)
                n_new += 1
            conf_mat += fast_confusion_matrix(gt, pred)
            collect_sample(class_samples, img_id, ip, gt, pred)
        except Exception:
            errors.append(img_id)
            tqdm.write(f"ERROR {img_id}:")
            traceback.print_exc()
        if n % 64 == 0:
            cuda_empty_cache()

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed / 60:.1f} min: {n_new} computed, {n_cached} read from cache, "
          f"{len(skipped)} skipped (image/GT missing), {len(errors)} errors")
    if skipped:
        print(f"  skipped (first 10): {skipped[:10]}")
    if errors:
        print(f"  errors (first 10): {errors[:10]}  -- re-run this step to retry them")

    write_manifest(paths.val_cache, "val_cache", fp, "complete" if not errors else "in_progress",
                   n_images=len(val_ids), n_ok=n_new + n_cached, skipped=skipped, errors=errors,
                   prompts=pc.as_dict())

    title = f"{DATASET_TITLE} val -- SAM3 baseline, {args.arm} prompts"
    iou, class_df, miou = compute_metrics(conf_mat, paths.baseline_dir, title, target_classes=pc.target_classes)
    plot_bar_chart(class_df, miou, paths.baseline_dir / "per_class_iou_bar_chart.png", title)
    plot_class_visualizations(class_samples, iou, paths.baseline_dir / "class_visualizations",
                              f"Prediction (SAM3, {args.arm} prompts)")

    if errors:
        write_manifest(paths.baseline_dir, "baseline", fp, "in_progress", errors=errors)
        raise RuntimeError(f"{len(errors)} val images failed -- metrics above are INCOMPLETE; re-run to retry.")
    write_manifest(paths.baseline_dir, "baseline", fp, "complete", n_images=len(val_ids),
                   n_evaluated=n_new + n_cached, skipped=skipped, prompts=pc.as_dict(),
                   confidence_threshold=CONFIDENCE_THRESHOLD, coarse_thresholds=COARSE_THRESHOLDS,
                   miou=miou)


if __name__ == "__main__":
    run_main(main)
