#!/usr/bin/env python3
"""
train_unetaspp_cityscapes.py -- train the UNet+ASPP refinement network on one arm's coarse masks.

Needs: outputs/<arm>/coarse_cache (get_coarse_cityscapes.py) and outputs/split.json.
Writes: outputs/<arm>/unetaspp/checkpoint/{best.pth, last.pth, history.json, training_curves.png}


    python train_unetaspp_cityscapes.py --arm llm
"""
from cityscapes_common import run_main
from cityscapes_refine import train_main

if __name__ == "__main__":
    run_main(lambda: train_main("unetaspp"))
