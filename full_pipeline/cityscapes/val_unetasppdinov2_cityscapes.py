#!/usr/bin/env python3
"""
val_unetasppdinov2_cityscapes.py -- hybrid evaluation: SAM3 baseline prediction (from val_cache) with the
UNet+ASPP+DINOv2 refinement overwriting every pixel it assigns to a target class.

Needs: outputs/<arm>/val_cache (sam3_baseline_cityscapes.py) and the trained checkpoint.
Writes: outputs/<arm>/unetasppdinov2/val/{summary_metrics.csv, per_class_metrics.csv,
        per_class_iou_bar_chart.png, class_visualizations/}

    python val_unetasppdinov2_cityscapes.py --arm llm
    python val_unetasppdinov2_cityscapes.py --arm nollm --ckpt /path/to/old_best.pth   # evaluate another checkpoint
"""
from cityscapes_common import run_main
from cityscapes_refine import val_main

if __name__ == "__main__":
    run_main(lambda: val_main("unetasppdinov2"))
