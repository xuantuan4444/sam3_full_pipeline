#!/usr/bin/env python3
"""
cityscapes_common.py
====================
Shared library for every step of the Cityscapes pipeline.

Section A (DATASET SPEC) is the ONLY place that knows anything specific to Cityscapes:
paths, the class list, label conventions, and training hyperparameters. Section B (generic) is
kept identical across full_pipeline/{pc59,voc,cityscapes}/<ds>_common.py -- only the module name
differs -- so a fix there has to be copied to the other two folders.

Heavy dependencies (torch, sam3) are imported lazily inside the functions that need them, so
run_pipeline_cityscapes.py can import this module for its preflight/plan without a GPU stack.
"""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ============================================================================
# A. DATASET SPEC -- Cityscapes (19 classes)
# ============================================================================
DATASET = "cityscapes"
DATASET_TITLE = "Cityscapes"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "cityscapes"
IMAGES_ROOT = DATA_DIR / "leftImg8bit"                # leftImg8bit/{train,val}/<city>/<stem>_leftImg8bit.png
GT_ROOT = DATA_DIR / "gtFine"                         # gtFine/{train,val}/<city>/<stem>_gtFine_labelIds.png
REQUIRED_DATA_PATHS = [IMAGES_ROOT / "train", IMAGES_ROOT / "val", GT_ROOT / "train", GT_ROOT / "val"]

CONFIG_DIR = ROOT / "configs"
TARGET_CLASSES_JSON = CONFIG_DIR / "target_classes.json"
ADJUST_PROMPT_JSON = CONFIG_DIR / "adjust_prompt_city.json"

CLASSES = [  # trainId order 0..18
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]
assert len(CLASSES) == 19

# Evaluation label space: trainId + 1 (1..19), 0 = "unlabeled" (no-prediction bucket), 255 =
# ignore. mIoU / mDice are averaged over the 19 classes (index 0 excluded), as in the original
# Cityscapes notebooks.
EVAL_NUM_CLASSES = 20
EVAL_CLASS_OFFSET = 1
EVAL_NO_PREDICTION = 0
EVAL_IGNORE = 255
EVAL_METRIC_INDICES = list(range(1, 20))
EVAL_EXTRA_NAMES = {0: "unlabeled"}

# Training label space (get_coarse / train): trainId 0..18, 255 = ignore.
TRAIN_RAW_VALUE = {name: i for i, name in enumerate(CLASSES)}

TRAIN_PARAMS = {
    "image_h": 512,
    "image_w": 1024,
    "batch_size": 2,
    "epochs": 30,
    "warmup_epochs": 2,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "patience": 8,
    "focal_gamma": 1.5,
    "default_tversky": (0.5, 0.5),
    "tversky": {
        "sidewalk": (0.50, 0.50),
        "vegetation": (0.50, 0.50),
        "terrain": (0.50, 0.50),
        "wall": (0.50, 0.50),
        "person": (0.50, 0.50),
        "fence": (0.45, 0.55),
        "bus": (0.45, 0.55),
        "pole": (0.40, 0.60),
        "train": (0.40, 0.60),
        "bicycle": (0.35, 0.65),
        "motorcycle": (0.35, 0.65),
        "traffic sign": (0.35, 0.65),
        "traffic light": (0.35, 0.65),
        "rider": (0.35, 0.65),
    },
    "default_boundary_lambda": 1.0,
    "boundary_lambda": {
        "vegetation": 0.5,
        "terrain": 0.5,
        "sidewalk": 1.0,
        "wall": 1.0,
        "person": 1.0,
        "bus": 1.0,
        "train": 1.0,
        "fence": 1.5,
        "rider": 2.0,
        "bicycle": 2.0,
        "motorcycle": 2.0,
        "pole": 2.5,
        "traffic sign": 2.5,
        "traffic light": 2.5,
    },
    "ce_boost": {"rider": 1.5},
}

_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6,
    20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
    30: 255, 31: 16, 32: 17, 33: 18,
}
_ID2TRAIN_LUT = np.full(256, 255, dtype=np.uint8)
for _k, _v in _ID_TO_TRAINID.items():
    _ID2TRAIN_LUT[_k] = _v


def list_ids(subset):
    """Image ids '<city>/<stem>', in the original discovery order (sorted cities, sorted files) so
    the seed-42 internal split is the same as the original notebooks'."""
    ids = []
    for city_dir in sorted((IMAGES_ROOT / subset).glob("*")):
        if city_dir.is_dir():
            for p in sorted(city_dir.glob("*_leftImg8bit.png")):
                ids.append(f"{city_dir.name}/{p.name[:-len('_leftImg8bit.png')]}")
    if subset == "val":  # original val only kept images that have a GT file
        ids = [i for i in ids if gt_path(i, subset) is not None]
    return ids


def image_path(img_id, subset):
    city, stem = img_id.split("/")
    return IMAGES_ROOT / subset / city / f"{stem}_leftImg8bit.png"


def _gt_file(img_id, subset):
    city, stem = img_id.split("/")
    d = GT_ROOT / subset / city
    train_id = d / f"{stem}_gtFine_labelTrainIds.png"
    if train_id.exists():
        return train_id, True
    label_id = d / f"{stem}_gtFine_labelIds.png"
    if label_id.exists():
        return label_id, False
    return None, None


def gt_path(img_id, subset):
    return _gt_file(img_id, subset)[0]


def cache_key(img_id):
    """Cache files are named by the bare stem (unique -- it contains the city), as in the original."""
    return img_id.split("/")[1]


def normalize_split_entry(entry):
    """Accepts '<city>/<stem>' or an old split.json image path (Windows or POSIX)."""
    s = str(entry).replace("\\", "/")
    parts = s.split("/")
    name = parts[-1]
    if name.endswith("_leftImg8bit.png"):
        name = name[:-len("_leftImg8bit.png")]
    return f"{parts[-2]}/{name}" if len(parts) >= 2 else name


def load_gt_train_raw(img_id, subset):
    """trainId map 0..18, 255 = ignore (labelIds are converted through the official LUT)."""
    path, is_train_id = _gt_file(img_id, subset)
    raw = np.array(Image.open(path), dtype=np.uint8)
    return raw if is_train_id else _ID2TRAIN_LUT[raw]


def train_valid_mask(raw):
    return (raw != 255).astype(np.uint8)


def load_gt_eval(img_id, subset):
    """trainId + 1 (1..19), 255 stays 255 (ignore)."""
    tid = load_gt_train_raw(img_id, subset)
    return np.where(tid == 255, 255, tid + 1).astype(np.uint8)


# ============================================================================
# B. GENERIC -- keep identical across the three dataset folders
# ============================================================================
ARMS = ("nollm", "llm")
ARCHS = ("unetaspp", "unetasppdinov2")
ARCH_TITLES = {"unetaspp": "UNet+ASPP", "unetasppdinov2": "UNet+ASPP+DINOv2"}

COARSE_THRESHOLDS = [0.50, 0.30, 0.20, 0.15]
CONFIDENCE_THRESHOLD = 0.30        # SAM3 cross-class competition: keep masks with score > 0.30
SEED = 42
INTERNAL_VAL_RATIO = 0.10

DEFAULT_SAM3_CKPT = ROOT / "weights" / "sam3.pt"
DEFAULT_OUT_ROOT = ROOT / "outputs"
SMOKE_OUT_ROOT = ROOT / "outputs_smoke"

EVAL_INDEX_TO_NAME = {i + EVAL_CLASS_OFFSET: n for i, n in enumerate(CLASSES)}
EVAL_INDEX_TO_NAME.update(EVAL_EXTRA_NAMES)
CLASS_TO_EVAL_INDEX = {n: i + EVAL_CLASS_OFFSET for i, n in enumerate(CLASSES)}


class PipelineError(RuntimeError):
    """A configuration problem the user has to fix -- printed without a traceback."""


def banner(msg):
    print(f"\n{'=' * 78}\n{msg}\n{'=' * 78}", flush=True)


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
class Paths:
    def __init__(self, arm, out_root=None):
        assert arm in ARMS, arm
        self.arm = arm
        self.out_root = Path(out_root) if out_root else DEFAULT_OUT_ROOT
        self.split_json = self.out_root / "split.json"
        self.dinov2_cache = self.out_root / "dinov2_cache"
        self.arm_dir = self.out_root / arm
        self.coarse_cache = self.arm_dir / "coarse_cache"
        self.val_cache = self.arm_dir / "val_cache"
        self.baseline_dir = self.arm_dir / "baseline"
        self.old_baseline_nollm = self.out_root / "baseline_nollm"
        self.logs = self.out_root / "logs"

    def ckpt_dir(self, arch):
        return self.arm_dir / arch / "checkpoint"

    def best_ckpt(self, arch):
        return self.ckpt_dir(arch) / "best.pth"

    def val_dir(self, arch):
        return self.arm_dir / arch / "val"


def add_common_args(parser, with_arch=False):
    parser.add_argument("--arm", required=True, choices=ARMS)
    if with_arch:
        parser.add_argument("--arch", required=True, choices=ARCHS)
    parser.add_argument("--out-root", default=None,
                        help=f"output root (default: {DEFAULT_OUT_ROOT.name}/, or {SMOKE_OUT_ROOT.name}/ with --limit)")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke test: only the first N train ids and first N val ids")
    parser.add_argument("--force", action="store_true",
                        help="move existing outputs of this step aside (*.stale-<time>) and redo it")
    return parser


def resolve_out_root(args):
    if args.out_root:
        return Path(args.out_root).resolve()
    return SMOKE_OUT_ROOT if args.limit else DEFAULT_OUT_ROOT


def resolve_sam3_ckpt(cli_value=None):
    if cli_value:
        return Path(cli_value).resolve()
    env = os.environ.get("SAM3_CKPT")
    return Path(env).resolve() if env else DEFAULT_SAM3_CKPT


def get_ids(subset, limit=None):
    ids = list_ids(subset)
    return ids[:limit] if limit else ids


# ---------------------------------------------------------------------------
# Target classes + prompts (single source of truth + fail-fast validation)
# ---------------------------------------------------------------------------
def load_target_classes():
    if not TARGET_CLASSES_JSON.exists():
        raise PipelineError(f"Target class file not found: {TARGET_CLASSES_JSON}")
    with open(TARGET_CLASSES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    targets = data["target_classes"] if isinstance(data, dict) else data
    if not isinstance(targets, list) or not targets:
        raise PipelineError(f"{TARGET_CLASSES_JSON}: 'target_classes' must be a non-empty list")
    unknown = [c for c in targets if c not in CLASSES]
    if unknown:
        raise PipelineError(f"{TARGET_CLASSES_JSON}: classes not in the {DATASET} class list: {unknown}")
    if len(set(targets)) != len(targets):
        raise PipelineError(f"{TARGET_CLASSES_JSON}: duplicate classes in target_classes")
    return list(targets)


class PromptConfig:
    """Prompts actually sent to SAM3 for one arm.

    target_prompts : {target class: [prompts]}  -- used for the coarse masks
    eval_overrides : {class: [prompts]}         -- used by the full-class SAM3 pass; every class
                                                   not listed here is queried with its bare name
    """

    def __init__(self, arm, target_classes, target_prompts, eval_overrides, source):
        self.arm = arm
        self.target_classes = target_classes
        self.target_prompts = target_prompts
        self.eval_overrides = eval_overrides
        self.source = source

    def prompts_for(self, class_name):
        return self.eval_overrides.get(class_name, [class_name])

    def as_dict(self):
        return {"arm": self.arm, "source": self.source, "target_classes": self.target_classes,
                "target_prompts": self.target_prompts, "eval_overrides": self.eval_overrides}

    def print_table(self):
        print(f"Prompt source ({self.arm}): {self.source}")
        width = max(len(c) for c in self.target_classes)
        for c in self.target_classes:
            print(f"  {c:<{width}} -> {self.target_prompts[c]}")
        others = [c for c in CLASSES if c not in self.eval_overrides]
        print(f"Queried with their bare name in the full-class SAM3 pass ({len(others)}): {others}")


def load_prompt_config(arm, target_classes=None):
    targets = target_classes or load_target_classes()
    if arm == "nollm":
        target_prompts = {c: [c] for c in targets}
        return PromptConfig(arm, targets, target_prompts, {}, "bare class names (no-LLM arm)")

    path = ADJUST_PROMPT_JSON
    if not path.exists():
        raise PipelineError(
            f"LLM arm: prompt file not found: {path}\n"
            f"  Put the REAL adjust_prompt file (generated by tools/generate_adjust_prompt_{DATASET}.py) there.")
    with open(path, encoding="utf-8") as f:
        adjust = json.load(f)
    if not isinstance(adjust, dict):
        raise PipelineError(f"LLM arm: {path} must be a JSON object {{class: [prompts]}}")

    keys, tset = set(adjust), set(targets)
    if keys != tset:
        raise PipelineError(
            f"LLM arm: classes in {path.name} do not match {TARGET_CLASSES_JSON.name}\n"
            f"  only in prompt file : {sorted(keys - tset)}\n"
            f"  only in target list : {sorted(tset - keys)}")
    for c in targets:
        v = adjust[c]
        if not isinstance(v, list) or not v or not all(isinstance(p, str) and p.strip() for p in v):
            raise PipelineError(f"LLM arm: {path.name}['{c}'] must be a non-empty list of non-empty strings, got {v!r}")
    if all(adjust[c] == [c] for c in targets):
        raise PipelineError(
            f"LLM arm: {path} is a PLACEHOLDER -- every class's only prompt is its own name.\n"
            f"  Running the LLM arm with it would silently reproduce the no-LLM arm.\n"
            f"  Replace it with the real LLM-generated adjust_prompt file.")

    target_prompts = {c: list(adjust[c]) for c in targets}
    return PromptConfig(arm, targets, target_prompts, dict(target_prompts), str(path.resolve()))


# ---------------------------------------------------------------------------
# Fingerprints + manifests (idempotency that notices changed prompts/config)
# ---------------------------------------------------------------------------
def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def fingerprints(arm, limit=None, target_classes=None):
    """All step fingerprints for one arm, computed from configs + id lists only (no outputs), so
    the orchestrator can plan without running anything."""
    pc = load_prompt_config(arm, target_classes)
    train_ids = get_ids("train", limit)
    val_ids = get_ids("val", limit)
    common = {"dataset": DATASET, "targets": pc.target_classes, "target_prompts": pc.target_prompts,
              "coarse_thresholds": COARSE_THRESHOLDS}
    fp = {
        "coarse": _sha({**common, "step": "coarse", "train_ids": _sha(train_ids)}),
        "baseline": _sha({**common, "step": "baseline", "eval_overrides": pc.eval_overrides,
                          "confidence_threshold": CONFIDENCE_THRESHOLD, "classes": CLASSES,
                          "val_ids": _sha(val_ids)}),
    }
    params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in TRAIN_PARAMS.items()}
    for arch in ARCHS:
        fp[f"train:{arch}"] = _sha({"step": "train", "arch": arch, "coarse": fp["coarse"],
                                    "params": params, "seed": SEED, "val_ratio": INTERNAL_VAL_RATIO})
        fp[f"val:{arch}"] = _sha({"step": "val", "arch": arch, "train": fp[f"train:{arch}"],
                                  "baseline": fp["baseline"]})
    return fp


MANIFEST = "manifest.json"


def read_manifest(d):
    p = Path(d) / MANIFEST
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_manifest(d, step, fingerprint, status, **info):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    m = {"step": step, "dataset": DATASET, "fingerprint": fingerprint, "status": status,
         "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    tmp = d / (MANIFEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(d / MANIFEST)


def is_complete(d, fingerprint):
    m = read_manifest(d)
    return bool(m) and m.get("status") == "complete" and m.get("fingerprint") == fingerprint


def move_aside(d, reason):
    d = Path(d)
    if not d.exists():
        return
    stale = d.with_name(f"{d.name}.stale-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.move(str(d), str(stale))
    print(f"[stale] {reason}: moved {d} -> {stale.name} (delete it once you no longer need it)")


def prepare_dir(d, step, fingerprint, force=False, resumable=False):
    """Make `d` ready for a (re)run of `step`.

    Existing contents are kept only when the manifest carries the same fingerprint AND either
    the step is resumable (per-image caches) or already complete. Anything else -- different
    prompts/config, unknown origin, or --force -- is moved aside, never silently reused.
    """
    d = Path(d)
    m = read_manifest(d)
    if d.exists() and any(d.iterdir()):
        if force:
            move_aside(d, f"--force {step}")
        elif m is None:
            move_aside(d, f"{step}: existing folder has no manifest (unknown prompts/config)")
        elif m.get("fingerprint") != fingerprint:
            move_aside(d, f"{step}: built with a different config/prompts "
                          f"(fingerprint {m.get('fingerprint')} != {fingerprint})")
        elif m.get("status") != "complete" and not resumable:
            move_aside(d, f"{step}: previous run did not finish")
    d.mkdir(parents=True, exist_ok=True)
    if read_manifest(d) is None:
        write_manifest(d, step, fingerprint, "in_progress")


# ---------------------------------------------------------------------------
# Internal train/val split (shared by both arms)
# ---------------------------------------------------------------------------
def make_or_load_split(all_train_ids, split_json):
    """Reuse split.json if it exists (e.g. copied from the no-LLM run), after checking it covers
    exactly the official train list. Otherwise create it: seed 42, 10% internal val -- the same
    algorithm the original get_coarse scripts used. Never overwritten."""
    split_json = Path(split_json)
    if split_json.exists():
        tr, va = read_split(split_json)
        if set(tr) | set(va) != set(all_train_ids) or set(tr) & set(va):
            raise PipelineError(
                f"{split_json} does not match the train list ({len(tr)}+{len(va)} ids in split vs "
                f"{len(all_train_ids)} train ids). Delete it to regenerate, or copy the correct one.")
        print(f"Using existing split: {split_json}  (train {len(tr)} | val {len(va)})")
        return tr, va

    import random
    rng = random.Random(SEED)
    shuffled = list(all_train_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * INTERNAL_VAL_RATIO))
    va, tr = shuffled[:n_val], shuffled[n_val:]
    split_json.parent.mkdir(parents=True, exist_ok=True)
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump({"train": tr, "val": va}, f)
    print(f"Created split: {split_json}  (train {len(tr)} | val {len(va)})")
    return tr, va


def read_split(split_json):
    """(internal_train_ids, internal_val_ids), entries normalized to this dataset's id format."""
    with open(split_json, encoding="utf-8") as f:
        split = json.load(f)
    return ([normalize_split_entry(s) for s in split["train"]],
            [normalize_split_entry(s) for s in split["val"]])


def split_hash(split_json):
    with open(split_json, encoding="utf-8") as f:
        return _sha(json.load(f))


# ---------------------------------------------------------------------------
# Torch / SAM3 helpers (lazy imports)
# ---------------------------------------------------------------------------
def set_seed(seed=SEED):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_torch(global_bf16_autocast):
    """SAM3 steps and the hybrid val ran under a process-wide bf16 autocast; training did not
    (it uses its own fp16 AMP). Keep that exactly."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if global_bf16_autocast and torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch :", torch.__version__)
    print("Device  :", device)
    if torch.cuda.is_available():
        print("GPU     :", torch.cuda.get_device_name(0))
    return device


def cuda_empty_cache():
    import torch
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[warn] torch.cuda.empty_cache failed: {e}")


def build_sam3_processor(sam3_ckpt, device, confidence_threshold):
    import gc
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_ckpt = Path(sam3_ckpt)
    if not sam3_ckpt.exists():
        raise PipelineError(f"SAM3 checkpoint not found: {sam3_ckpt}")
    gc.collect()
    cuda_empty_cache()
    bpe_path = Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=str(bpe_path), checkpoint_path=str(sam3_ckpt), load_from_HF=False)
    model = model.to(device)
    gc.collect()
    cuda_empty_cache()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    print(f"SAM3 loaded: {sam3_ckpt}  (processor confidence_threshold={confidence_threshold})")
    return model, processor


def extract_masks_scores(state):
    """state['masks'] -> list of (H, W) bool arrays (already at original resolution), state['scores'] -> floats."""
    if "masks" not in state or "scores" not in state or state["masks"] is None or len(state["masks"]) == 0:
        return [], []
    masks, scores = [], []
    for i in range(len(state["masks"])):
        masks.append(state["masks"][i].squeeze(0).detach().cpu().numpy().astype(bool))
        scores.append(float(state["scores"][i].item()))
    return masks, scores


def union_at_threshold(masks, scores, shape_hw, thresholds=COARSE_THRESHOLDS):
    """Try each threshold in order; union every mask reaching the first one that has >= 1 mask."""
    h, w = shape_hw
    scores_np = np.asarray(scores, dtype=np.float32)
    for thr in thresholds:
        idx = np.where(scores_np >= thr)[0]
        if len(idx) > 0:
            union = np.zeros((h, w), dtype=bool)
            for i in idx:
                union |= masks[i]
            return union.astype(np.uint8), thr
    return np.zeros((h, w), dtype=np.uint8), None


def query_class(processor, state, prompts):
    """Run every prompt of one class, return (state, pooled masks, pooled scores)."""
    all_masks, all_scores = [], []
    for prompt in prompts:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)
        masks, scores = extract_masks_scores(state)
        all_masks.extend(masks)
        all_scores.extend(scores)
    return state, all_masks, all_scores


# ---------------------------------------------------------------------------
# Metrics + plots
# ---------------------------------------------------------------------------
def fast_confusion_matrix(gt, pred, num_classes=EVAL_NUM_CLASSES):
    """Rows = GT, cols = prediction. Pixels with GT == ignore, or any value outside
    [0, num_classes), are dropped (PC59: a pixel SAM3 left at 255 is therefore not counted)."""
    valid = (gt != EVAL_IGNORE) & (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    hist = np.bincount(num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
                       minlength=num_classes * num_classes)
    return hist.reshape(num_classes, num_classes)


def compute_metrics(conf_mat, out_dir, title, target_classes=None, hybrid_label=None):
    """Writes summary_metrics.csv + per_class_metrics.csv into out_dir. Formulas are the original
    ones: IoU = diag / (union + eps); mIoU / mDice = nanmean over EVAL_METRIC_INDICES; pixel
    accuracy over the whole confusion matrix."""
    import pandas as pd

    eps = 1e-10
    diag = np.diag(conf_mat).astype(np.float64)
    gt_sum = conf_mat.sum(axis=1).astype(np.float64)
    pred_sum = conf_mat.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - diag
    iou = diag / (union + eps)
    dice = (2.0 * diag) / (gt_sum + pred_sum + eps)
    pixel_acc = diag.sum() / (conf_mat.sum() + eps)

    idx = EVAL_METRIC_INDICES
    miou = float(np.nanmean(iou[idx]))
    mdice = float(np.nanmean(dice[idx]))

    rows = {
        "Class": [EVAL_INDEX_TO_NAME[i] for i in idx],
        "IoU": iou[idx],
        "Dice": dice[idx],
        "GT Pixels": gt_sum[idx].astype(np.int64),
        "Pred Pixels": pred_sum[idx].astype(np.int64),
    }
    if hybrid_label:
        rows = {"Class": rows["Class"],
                "Source": [hybrid_label if EVAL_INDEX_TO_NAME[i] in target_classes else "SAM3 (baseline)" for i in idx],
                **{k: v for k, v in rows.items() if k != "Class"}}
    class_df = pd.DataFrame(rows).sort_values("IoU", ascending=True).reset_index(drop=True)

    metric_names = ["Pixel Accuracy", "mIoU", "Mean Dice"]
    values = [pixel_acc, miou, mdice]
    if target_classes:
        w = class_df[class_df["Class"].isin(target_classes)]["IoU"]
        s = class_df[~class_df["Class"].isin(target_classes)]["IoU"]
        metric_names += [f"mIoU_w ({len(w)} target classes)", f"mIoU_s ({len(s)} other classes)"]
        values += [float(w.mean()), float(s.mean())]
    summary_df = pd.DataFrame({"Metric": metric_names, "Value": values})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    class_df.to_csv(out_dir / "per_class_metrics.csv", index=False)

    banner(f"SUMMARY -- {title}")
    print(summary_df.to_string(index=False, formatters={"Value": lambda x: f"{x:.4f}"}))
    print(f"\nPER-CLASS ({len(idx)} classes, IoU ascending)")
    print(class_df.to_string(index=False, formatters={"IoU": lambda x: f"{x:.4f}", "Dice": lambda x: f"{x:.4f}"}))
    print(f"\nSaved: {out_dir / 'summary_metrics.csv'}\nSaved: {out_dir / 'per_class_metrics.csv'}")
    return iou, class_df, miou


def plot_bar_chart(class_df, miou, out_path, title, hybrid_label=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(class_df)
    fig, ax = plt.subplots(figsize=(10, max(7, n * 0.22)))
    bars = ax.barh(class_df["Class"], class_df["IoU"], color=plt.cm.RdYlGn(class_df["IoU"].values))
    handles = [plt.Line2D([0], [0], color="blue", linestyle="--", label=f"mIoU = {miou:.4f}")]
    if hybrid_label and "Source" in class_df:
        for bar, src in zip(bars, class_df["Source"].values):
            if src == hybrid_label:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.8)
                bar.set_hatch("//")
        n_t = int((class_df["Source"] == hybrid_label).sum())
        handles = [Patch(facecolor="white", edgecolor="black", hatch="//", label=f"{hybrid_label} ({n_t} target classes)"),
                   Patch(facecolor="white", edgecolor="gray", label="SAM3 baseline, unchanged")] + handles
    ax.set_xlabel("IoU", fontsize=12)
    ax.set_title(f"{title} (mIoU={miou:.4f})", fontsize=12)
    ax.set_xlim(0, 1)
    ax.axvline(x=miou, color="blue", linestyle="--", linewidth=1)
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    for bar, val in zip(bars, class_df["IoU"].values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _colormap(n):
    """Deterministic VOC-style bit-interleaved palette."""
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


def _colorize(mask, cmap):
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label_id in range(cmap.shape[0]):
        colored[mask == label_id] = cmap[label_id]
    colored[mask == EVAL_IGNORE] = [128, 128, 128]
    return colored


def plot_class_visualizations(class_samples, iou, out_dir, pred_title, target_classes=None, hybrid_tag=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = _colormap(EVAL_NUM_CLASSES)
    for cls_idx in EVAL_METRIC_INDICES:
        samples = class_samples.get(cls_idx, [])
        if not samples:
            continue
        name = EVAL_INDEX_TO_NAME[cls_idx]
        n_show = min(2, len(samples))
        fig, axes = plt.subplots(n_show, 4, figsize=(18, 4.5 * n_show))
        if n_show == 1:
            axes = axes[np.newaxis, :]
        tag = ""
        if target_classes is not None:
            tag = f"  [{hybrid_tag if name in target_classes else 'SAM3 baseline'}]"
        fig.suptitle(f"Class: {name} (IoU = {iou[cls_idx]:.4f}){tag}", fontsize=14, fontweight="bold")
        for row in range(n_show):
            s = samples[row]
            img = np.array(Image.open(s["image_path"]).convert("RGB"))
            err = (s["pred"] != s["gt"]) & (s["gt"] != EVAL_IGNORE)
            panels = [(img, f"Image: {s['id']}"), (_colorize(s["gt"], cmap), "Ground Truth"),
                      (_colorize(s["pred"], cmap), pred_title)]
            for col, (im, t) in enumerate(panels):
                axes[row, col].imshow(im)
                axes[row, col].set_title(t, fontsize=10)
                axes[row, col].axis("off")
            axes[row, 3].imshow(img)
            axes[row, 3].imshow(err.astype(np.float32), alpha=0.55, cmap="Reds")
            axes[row, 3].set_title("Error Overlay", fontsize=10)
            axes[row, 3].axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name.replace(' ', '_')}.png", dpi=100)
        plt.close(fig)
    print(f"Saved per-class visualizations to {out_dir}")


def collect_sample(class_samples, img_id, img_path, gt, pred, max_per_class=2):
    present = {int(c) for c in np.unique(gt) if c != EVAL_IGNORE and c in EVAL_INDEX_TO_NAME}
    for c in present:
        lst = class_samples.setdefault(c, [])
        if len(lst) < max_per_class:
            lst.append({"id": img_id, "image_path": str(img_path), "gt": gt, "pred": pred})
    return present


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------
def _read_result(d, target_classes):
    import pandas as pd
    d = Path(d)
    s_csv, c_csv = d / "summary_metrics.csv", d / "per_class_metrics.csv"
    if not (s_csv.exists() and c_csv.exists()):
        return None
    summ = pd.read_csv(s_csv)
    cls = pd.read_csv(c_csv)

    def pick(pred):
        for m, v in zip(summ["Metric"], summ["Value"]):
            if pred(str(m).lower()):
                return float(v)
        return float("nan")

    is_t = cls["Class"].isin(target_classes)
    return {
        "Pixel Accuracy": pick(lambda m: m.startswith("pixel")),
        "mIoU": pick(lambda m: m.startswith("miou") and not m.startswith("miou_")),
        "Mean Dice": pick(lambda m: m.startswith("mean dice")),
        "mIoU_w": float(cls[is_t]["IoU"].mean()),
        "mIoU_s": float(cls[~is_t]["IoU"].mean()),
    }


def write_ablation_table(out_root=None):
    import pandas as pd
    targets = load_target_classes()
    out_root = Path(out_root) if out_root else DEFAULT_OUT_ROOT
    rows = []
    for arm in ARMS:
        p = Paths(arm, out_root)
        base = p.baseline_dir
        if arm == "nollm" and not (base / "summary_metrics.csv").exists():
            base = p.old_baseline_nollm
        candidates = [("SAM3 baseline", base)] + [(f"SAM3 + {ARCH_TITLES[a]} (hybrid)", p.val_dir(a)) for a in ARCHS]
        for method, d in candidates:
            r = _read_result(d, targets)
            if r:
                rows.append({"Arm": arm, "Method": method, **r, "Source dir": str(Path(d).relative_to(out_root))})
    if not rows:
        print("[ablation] no results yet")
        return None
    df = pd.DataFrame(rows)
    out = out_root / "ablation_table.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    banner(f"ABLATION TABLE -- {DATASET_TITLE}  ({len(targets)} target / {len(EVAL_METRIC_INDICES) - len(targets)} other classes)")
    print(df.drop(columns=["Source dir"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {out}")
    return df


def run_main(main):
    """Common entry-point wrapper: config problems print cleanly, everything else with traceback."""
    import traceback
    try:
        main()
    except PipelineError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except Exception:
        print(f"\n[FATAL] {Path(sys.argv[0]).name} failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
