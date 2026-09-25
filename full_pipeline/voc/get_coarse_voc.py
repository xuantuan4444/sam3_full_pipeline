#!/usr/bin/env python3
"""
get_coarse_voc.py -- STEP 2 of an arm: SAM3 coarse masks for the target classes on the
official train split.

Output: outputs/<arm>/coarse_cache/<id>.npz, one (H, W) uint8 0/1 mask per target class (key =
class name), union-at-threshold over ALL prompts of the class, thresholds [0.5, 0.3, 0.2, 0.15]
-- the same algorithm as the original get_coarse scripts. Also creates outputs/split.json
(seed 42, 10% internal val) if it does not exist yet; an existing one is reused, never overwritten.

Resumable: images whose .npz is already valid are skipped.

    python get_coarse_voc.py --arm llm
"""

import argparse
import gc
import json
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from voc_common import (
    COARSE_THRESHOLDS, DATASET_TITLE, TRAIN_RAW_VALUE, Paths, add_common_args, banner,
    build_sam3_processor, cache_key, cuda_empty_cache, fingerprints, get_ids, gt_path, image_path,
    is_complete, load_gt_train_raw, load_prompt_config, make_or_load_split, prepare_dir,
    query_class, resolve_out_root, resolve_sam3_ckpt, run_main, set_seed, setup_torch,
    train_valid_mask, union_at_threshold, write_manifest,
)


def cache_ok(path, targets):
    try:
        with np.load(path) as d:
            return all(c in d.files and d[c].ndim == 2 for c in targets)
    except Exception:
        return False


def build_cache(ids, pc, cache_dir, sam3_ckpt, device):
    todo = [i for i in ids if not cache_ok(cache_dir / f"{cache_key(i)}.npz", pc.target_classes)]
    print(f"Coarse cache: {len(ids) - len(todo)}/{len(ids)} already valid, {len(todo)} to build")
    if not todo:
        return []
    model, processor = build_sam3_processor(sam3_ckpt, device, confidence_threshold=min(COARSE_THRESHOLDS))
    errors = []
    try:
        for step, img_id in enumerate(tqdm(todo, desc=f"SAM3 coarse cache ({pc.arm})"), 1):
            try:
                image = Image.open(image_path(img_id, "train")).convert("RGB")
                state = processor.set_image(image)
                coarse = {}
                for cname in pc.target_classes:
                    state, masks, scores = query_class(processor, state, pc.target_prompts[cname])
                    coarse[cname], _ = union_at_threshold(masks, scores, (image.height, image.width))
                out = cache_dir / f"{cache_key(img_id)}.npz"
                tmp = out.with_name(out.stem + ".tmp.npz")
                np.savez_compressed(tmp, **coarse)
                tmp.replace(out)
            except Exception as e:
                errors.append((img_id, f"{type(e).__name__}: {e}"))
            if step % 64 == 0:
                gc.collect()
                cuda_empty_cache()
    finally:
        del processor, model
        gc.collect()
        cuda_empty_cache()
    print(f"Built {len(todo) - len(errors)}, errors: {len(errors)}")
    for e in errors[:5]:
        print(f"  {e}")
    return errors


def scan_class_stats(train_ids, targets, out_path):
    """Per-class GT pixel stats on the internal-train split (informational, same as original)."""
    pos = {c: 0 for c in targets}
    neg = {c: 0 for c in targets}
    pos_imgs = {c: 0 for c in targets}
    for img_id in tqdm(train_ids, desc="Scan GT pixel stats"):
        if gt_path(img_id, "train") is None:
            continue
        raw = load_gt_train_raw(img_id, "train")
        valid = train_valid_mask(raw).astype(bool)
        total = int(valid.sum())
        for c in targets:
            px = int(((raw == TRAIN_RAW_VALUE[c]) & valid).sum())
            if px > 0:
                pos_imgs[c] += 1
                pos[c] += px
                neg[c] += total - px
    stats = {c: {"pos_images": pos_imgs[c], "pos_px": pos[c], "neg_px": neg[c],
                 "pos_weight": float(np.clip(neg[c] / max(pos[c], 1), 1.0, 50.0))} for c in targets}
    for c in targets:
        s = stats[c]
        print(f"  {c:14s} | pos_images={s['pos_images']:5d} | pos_px={s['pos_px']:>13,} | pos_weight={s['pos_weight']:.2f}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


def visualize_sample(img_id, targets, cache_dir, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = cache_dir / f"{cache_key(img_id)}.npz"
    if not p.exists():
        return
    n = len(targets)
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    with np.load(p) as d:
        for ax, c in zip(axes.flat, targets):
            ax.imshow(d[c], vmin=0, vmax=1, cmap="gray")
            ax.set_title(c, fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Coarse masks -- {img_id}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=90)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    p.add_argument("--sam3-ckpt", default=None)
    args = p.parse_args()

    out_root = resolve_out_root(args)
    paths = Paths(args.arm, out_root)
    pc = load_prompt_config(args.arm)
    fp = fingerprints(args.arm, args.limit)["coarse"]

    banner(f"GET COARSE -- {DATASET_TITLE}, arm={args.arm}")
    pc.print_table()
    if is_complete(paths.coarse_cache, fp) and not args.force:
        print(f"[skip] already complete: {paths.coarse_cache}")
        return

    t0 = time.time()
    device = setup_torch(global_bf16_autocast=True)
    set_seed()
    all_ids = get_ids("train", args.limit)
    print(f"Train ids: {len(all_ids)}")
    internal_train, _ = make_or_load_split(all_ids, paths.split_json)

    prepare_dir(paths.coarse_cache, "coarse", fp, force=args.force, resumable=True)
    sam3_ckpt = resolve_sam3_ckpt(args.sam3_ckpt)
    build_cache(all_ids, pc, paths.coarse_cache, sam3_ckpt, device)

    # Audit pass: rebuild anything still missing/corrupt once more (single SAM3 load).
    remain = [i for i in all_ids if not cache_ok(paths.coarse_cache / f"{cache_key(i)}.npz", pc.target_classes)]
    if remain:
        print(f"Audit: {len(remain)} missing/corrupt after first pass -- retrying once")
        build_cache(remain, pc, paths.coarse_cache, sam3_ckpt, device)
        remain = [i for i in remain if not cache_ok(paths.coarse_cache / f"{cache_key(i)}.npz", pc.target_classes)]

    scan_class_stats(internal_train, pc.target_classes, paths.arm_dir / "class_pixel_stats.json")
    if internal_train:
        visualize_sample(internal_train[0], pc.target_classes, paths.coarse_cache,
                         paths.arm_dir / "sample_coarse_masks.png")

    # Same tolerance as the original: images that still fail after the retry are listed here and
    # dropped by the train step's readiness filter.
    if remain:
        print(f"[warn] {len(remain)} images have no valid coarse mask after the retry "
              f"(train will skip them): {remain[:10]}")
    write_manifest(paths.coarse_cache, "coarse", fp, "complete", n_images=len(all_ids),
                   n_missing=len(remain), missing=remain, prompts=pc.as_dict(),
                   coarse_thresholds=COARSE_THRESHOLDS)
    print(f"\nElapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    run_main(main)
