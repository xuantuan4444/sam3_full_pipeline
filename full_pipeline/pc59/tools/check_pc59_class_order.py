#!/usr/bin/env python3
"""
check_pc59_class_order.py -- sanity checks for the PC59 label conventions used by the pipeline.

1. pc59_common.CLASSES is the 59-class ALPHABETICAL list (== mmsegmentation PascalContextDataset59).
2. RAW (get_coarse/train) and canonical (baseline/val) indices agree: raw value of class c minus 1
   is c's eval index, for every class.
3. On a few real GT masks: raw values are within 0..59 and the canonical conversion only produces
   0..58 or 255.
4. If data/59_labels.txt exists: show where its order differs (it must NOT be used as class order).

    python tools/check_pc59_class_order.py [--n-masks 20]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pc59_common as C  # noqa: E402

MMSEG_PC59 = [
    "aeroplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird", "boat", "book",
    "bottle", "building", "bus", "cabinet", "car", "cat", "ceiling", "chair", "cloth",
    "computer", "cow", "cup", "curtain", "dog", "door", "fence", "floor", "flower", "food",
    "grass", "ground", "horse", "keyboard", "light", "motorbike", "mountain", "mouse",
    "person", "plate", "platform", "pottedplant", "road", "rock", "sheep", "shelves",
    "sidewalk", "sign", "sky", "snow", "sofa", "table", "track", "train", "tree", "truck",
    "tvmonitor", "wall", "water", "window", "wood",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-masks", type=int, default=20)
    args = ap.parse_args()
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
        ok &= bool(cond)

    print("Class list")
    check(len(C.CLASSES) == 59 and len(set(C.CLASSES)) == 59, "59 unique classes")
    check(C.CLASSES == sorted(C.CLASSES), "alphabetical order")
    check(C.CLASSES == MMSEG_PC59, "identical to mmsegmentation PascalContextDataset59")

    print("Index conventions")
    check(all(C.TRAIN_RAW_VALUE[c] - 1 == C.CLASS_TO_EVAL_INDEX[c] for c in C.CLASSES),
          "raw value - 1 == eval index for every class")
    check(C.EVAL_NO_PREDICTION == 255 and C.EVAL_IGNORE == 255, "no-prediction / ignore value is 255 (0 is 'aeroplane')")
    targets = C.load_target_classes()
    check(all(t in C.CLASSES for t in targets), f"all {len(targets)} target classes are PC59 classes")

    print("GT masks")
    if not C.GT_DIR.exists():
        print(f"  [skip] {C.GT_DIR} not found")
    else:
        ids = C.get_ids("val", args.n_masks)
        for img_id in ids:
            if C.gt_path(img_id, "val") is None:
                check(False, f"{img_id}: GT missing")
                continue
            raw = C.load_gt_train_raw(img_id, "val")
            ev = C.load_gt_eval(img_id, "val")
            bad_eval = set(np.unique(ev).tolist()) - set(range(59)) - {255}
            check(raw.max() <= 59 and not bad_eval and np.array_equal(ev != 255, raw != 0),
                  f"{img_id}: raw {int(raw.min())}..{int(raw.max())}, canonical OK")

    labels_txt = C.DATA_DIR / "59_labels.txt"
    if labels_txt.exists():
        names = []
        for line in labels_txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                names.append(line.split(":", 1)[-1].strip() if ":" in line else line.split(maxsplit=1)[-1])
        diff = sum(a != b for a, b in zip(names, C.CLASSES))
        print(f"59_labels.txt order differs from the alphabetical order at {diff}/59 positions "
              f"(expected -- that file must never be used as the class order)")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
