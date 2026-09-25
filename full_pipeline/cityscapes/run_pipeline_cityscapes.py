#!/usr/bin/env python3
"""
run_pipeline_cityscapes.py -- run the whole Cityscapes pipeline in one go.

Per arm (nollm, llm), in this order:
    1. sam3_baseline_cityscapes.py   SAM3 on val once -> baseline metrics + val_cache
    2. get_coarse_cityscapes.py      SAM3 coarse masks on train (+ split.json if missing)
    3. train_unetaspp_cityscapes.py
    4. train_unetasppdinov2_cityscapes.py
    5. val_unetaspp_cityscapes.py        (reads val_cache, no SAM3)
    6. val_unetasppdinov2_cityscapes.py  (reads val_cache, no SAM3)
then outputs/ablation_table.csv.

Every step is skipped when its output is complete AND was built from the same target classes,
prompts and settings (fingerprint in manifest.json). Interrupted? Re-run the same command.

    python run_pipeline_cityscapes.py --skip-nollm              # LLM arm only
    python run_pipeline_cityscapes.py --skip-nollm --dry-run    # preflight + plan, runs nothing
    python run_pipeline_cityscapes.py --skip-nollm --limit 5    # smoke test -> outputs_smoke/
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import cityscapes_common as C

STEP_ORDER = ["baseline", "coarse", "train:unetaspp", "train:unetasppdinov2", "val:unetaspp", "val:unetasppdinov2"]
FORCE_CHOICES = ["baseline", "coarse", "train", "val"]


class Tee:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "ab")
        self.path = path

    def write_bytes(self, b):
        sys.stdout.buffer.write(b)
        sys.stdout.buffer.flush()
        self.f.write(b)
        self.f.flush()

    def print(self, *a):
        self.write_bytes((" ".join(str(x) for x in a) + "\n").encode("utf-8", "replace"))


LOG = None


def log(*a):
    if LOG:
        LOG.print(*a)
    else:
        print(*a, flush=True)


def banner(msg):
    log(f"\n{'=' * 78}\n{msg}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def preflight(arms, args):
    problems = []
    banner(f"PREFLIGHT -- {C.DATASET_TITLE}")
    log(f"Python      : {sys.executable}")
    log(f"Project dir : {C.ROOT}")

    for p in C.REQUIRED_DATA_PATHS:
        ok = p.exists()
        log(f"  [{'ok' if ok else 'MISSING'}] {p}")
        if not ok:
            problems.append(f"data path missing: {p}")

    ckpt = C.resolve_sam3_ckpt(args.sam3_ckpt)
    log(f"  [{'ok' if ckpt.exists() else 'MISSING'}] SAM3 checkpoint: {ckpt}")
    if not ckpt.exists():
        problems.append(f"SAM3 checkpoint missing: {ckpt} (put it in weights/ or pass --sam3-ckpt)")

    for mod in ("torch", "torchvision", "sam3", "albumentations", "cv2", "scipy", "pandas", "matplotlib", "tqdm"):
        if importlib.util.find_spec(mod) is None:
            problems.append(f"python package not installed: {mod}")
            log(f"  [MISSING] package {mod}")
    try:
        from importlib.metadata import version
        av = version("albumentations")
        if int(av.split(".")[0]) < 2:
            problems.append(f"albumentations {av} < 2.0 (the original runs used 2.x)")
    except Exception:
        pass

    try:
        targets = C.load_target_classes()
        log(f"Target classes ({len(targets)}) from {C.TARGET_CLASSES_JSON}:\n  {targets}")
    except C.PipelineError as e:
        problems.append(str(e))
        targets = None

    if targets:
        for arm in arms:
            try:
                log("")
                _log_prompt_table(C.load_prompt_config(arm, targets))
            except C.PipelineError as e:
                problems.append(str(e))

    if all(p.exists() for p in C.REQUIRED_DATA_PATHS):
        try:
            n_tr, n_va = len(C.get_ids("train", args.limit)), len(C.get_ids("val", args.limit))
            log(f"\nTrain ids: {n_tr}   Val ids: {n_va}" + (f"   (--limit {args.limit})" if args.limit else ""))
            sample = C.get_ids("val", 1)[0]
            if not C.image_path(sample, "val").exists() or C.gt_path(sample, "val") is None:
                problems.append(f"first val id '{sample}' has no image or GT -- check data/ layout")
        except Exception as e:
            problems.append(f"cannot read id lists: {e}")

    if problems:
        banner("PREFLIGHT FAILED")
        for p in problems:
            log(f"  - {p}")
    else:
        log("\nPreflight OK")
    return problems


def _log_prompt_table(pc):
    log(f"Prompt source ({pc.arm}): {pc.source}")
    width = max(len(c) for c in pc.target_classes)
    for c in pc.target_classes:
        log(f"  {c:<{width}} -> {pc.target_prompts[c]}")


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
def step_command(step, arm, args, out_root, force):
    common = ["--arm", arm, "--out-root", str(out_root)]
    if args.limit:
        common += ["--limit", str(args.limit)]
    if force:
        common += ["--force"]
    if step == "baseline":
        return [f"sam3_baseline_{C.DATASET}.py"] + common + (["--sam3-ckpt", args.sam3_ckpt] if args.sam3_ckpt else [])
    if step == "coarse":
        return [f"get_coarse_{C.DATASET}.py"] + common + (["--sam3-ckpt", args.sam3_ckpt] if args.sam3_ckpt else [])
    kind, arch = step.split(":")
    if kind == "train":
        epochs = args.epochs or (1 if args.limit else None)
        return [f"train_{arch}_{C.DATASET}.py"] + common + (["--epochs", str(epochs)] if epochs else [])
    return [f"val_{arch}_{C.DATASET}.py"] + common


def step_done(step, paths, fps):
    if step == "baseline":
        return C.is_complete(paths.baseline_dir, fps["baseline"]) and C.is_complete(paths.val_cache, fps["baseline"])
    if step == "coarse":
        return C.is_complete(paths.coarse_cache, fps["coarse"])
    kind, arch = step.split(":")
    if kind == "train":
        return C.is_complete(paths.ckpt_dir(arch), fps[step]) and paths.best_ckpt(arch).exists()
    return C.is_complete(paths.val_dir(arch), fps[step])


def build_plan(arms, archs, args, out_root):
    plan = []
    for arm in arms:
        paths = C.Paths(arm, out_root)
        fps = C.fingerprints(arm, args.limit)
        for step in STEP_ORDER:
            if ":" in step and step.split(":")[1] not in archs:
                continue
            force = step.split(":")[0] in (args.force or [])
            done = step_done(step, paths, fps) and not force
            plan.append({"arm": arm, "step": step, "done": done, "force": force,
                         "cmd": step_command(step, arm, args, out_root, force)})
    return plan


def print_plan(plan):
    banner("PLAN")
    for item in plan:
        status = "skip (done)" if item["done"] else ("RUN (--force)" if item["force"] else "RUN")
        log(f"  [{item['arm']:5s}] {item['step']:22s} {status:14s} python {' '.join(item['cmd'])}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_step(item):
    cmd = [sys.executable, "-u"] + [str(C.ROOT / item["cmd"][0])] + item["cmd"][1:]
    banner(f"[{item['arm']}] {item['step']}")
    log("$ " + " ".join(cmd))
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(C.ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    while True:
        chunk = proc.stdout.read1(8192) if hasattr(proc.stdout, "read1") else proc.stdout.read(1)
        if not chunk:
            break
        LOG.write_bytes(chunk)
    rc = proc.wait()
    log(f"[{item['arm']}] {item['step']} finished with code {rc} in {(time.time() - t0) / 60:.1f} min")
    return rc


def log_ablation_table(out_root):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        C.write_ablation_table(out_root)
    log(buf.getvalue())


def main():
    p = argparse.ArgumentParser(description=f"Full {C.DATASET_TITLE} pipeline (no-LLM + LLM arms).")
    p.add_argument("--skip-nollm", action="store_true", help="do not run the no-LLM arm")
    p.add_argument("--skip-llm", action="store_true", help="do not run the LLM arm")
    p.add_argument("--archs", nargs="+", choices=C.ARCHS, default=list(C.ARCHS))
    p.add_argument("--force", nargs="+", choices=FORCE_CHOICES, default=[],
                   help="redo these steps even if complete (old outputs are moved to *.stale-<time>)")
    p.add_argument("--dry-run", action="store_true", help="preflight + plan only")
    p.add_argument("--limit", type=int, default=None, help="smoke test on the first N train/val ids (-> outputs_smoke/)")
    p.add_argument("--epochs", type=int, default=None, help="override epochs (default 1 with --limit)")
    p.add_argument("--sam3-ckpt", default=None, help=f"default: {C.DEFAULT_SAM3_CKPT} or $SAM3_CKPT")
    p.add_argument("--out-root", default=None)
    p.add_argument("--table-only", action="store_true", help="only rebuild outputs/ablation_table.csv")
    args = p.parse_args()

    out_root = Path(args.out_root).resolve() if args.out_root else (C.SMOKE_OUT_ROOT if args.limit else C.DEFAULT_OUT_ROOT)
    if args.table_only:
        C.write_ablation_table(out_root)
        return

    arms = [a for a in C.ARMS if not (a == "nollm" and args.skip_nollm) and not (a == "llm" and args.skip_llm)]
    if not arms:
        sys.exit("Both arms skipped -- nothing to do.")

    global LOG
    LOG = Tee(out_root / "logs" / f"run_{time.strftime('%Y%m%d-%H%M%S')}.log")
    log(f"Log file: {LOG.path}")
    log(f"Arms: {arms}   Architectures: {args.archs}   Output root: {out_root}")

    problems = preflight(arms, args)
    if problems:
        sys.exit(2)

    plan = build_plan(arms, args.archs, args, out_root)
    print_plan(plan)
    if args.dry_run:
        log("\n--dry-run: nothing executed.")
        return

    t0 = time.time()
    for item in plan:
        if item["done"]:
            continue
        rc = run_step(item)
        if rc != 0:
            banner("PIPELINE STOPPED")
            log(f"Step '{item['step']}' (arm {item['arm']}) failed with exit code {rc}. See the log above / {LOG.path}.")
            log("Fix the problem and re-run the same command: finished steps are skipped.")
            sys.exit(1)

    banner(f"PIPELINE COMPLETE in {(time.time() - t0) / 60:.1f} min")
    log_ablation_table(out_root)
    log(f"\nFull log: {LOG.path}")


if __name__ == "__main__":
    try:
        main()
    except C.PipelineError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
