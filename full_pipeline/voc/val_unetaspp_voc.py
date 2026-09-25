#!/usr/bin/env python3
"""
val_unetaspp_voc.py -- hybrid evaluation: SAM3 baseline prediction (from val_cache) with the
UNet+ASPP refinement overwriting every pixel it assigns to a target class.

Needs: outputs/<arm>/val_cache (sam3_baseline_voc.py) and the trained checkpoint.
Writes: outputs/<arm>/unetaspp/val/{summary_metrics.csv, per_class_metrics.csv,
        per_class_iou_bar_chart.png, class_visualizations/}

    python val_unetaspp_voc.py --arm llm
    python val_unetaspp_voc.py --arm nollm --ckpt /path/to/old_best.pth   # evaluate another checkpoint
"""
from voc_common import run_main
from voc_refine import val_main

if __name__ == "__main__":
    run_main(lambda: val_main("unetaspp"))
