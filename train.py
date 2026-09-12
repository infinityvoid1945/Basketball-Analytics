# the code used for training the model
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch
import torch.nn as nn
import torch.multiprocessing as mp

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from ultralytics import YOLO


def _de_parallel(m):
    return m.module if hasattr(m, "module") else m


DATA_YAML = "/workspace/dataset/data.yaml"
MODEL     = "yolo11x.pt"
PROJECT   = "/workspace/runs/ball"
NAME      = "cost_sensitive_x_1280"

EPOCHS  = 100
IMGSZ   = 1280
BATCH   = 16
WORKERS = 8
DEVICE  = 0

BALL_CLASS_IDX = 1
BALL_FN_WEIGHT = 4.0


def _make_patched_init_criterion(orig_init_criterion, pos_weight):
    def _patched():
        criterion = orig_init_criterion()
        if not hasattr(criterion, "bce"):
            raise RuntimeError(
                "Loss has no 'bce' attribute -- Ultralytics internals changed. "
                "Update this script for your ultralytics version."
            )
        criterion.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        return criterion
    return _patched


def apply_cost_sensitive(trainer):
    model = _de_parallel(trainer.model)
    device = next(model.parameters()).device
    nc = int(model.nc)

    if BALL_CLASS_IDX >= nc:
        raise ValueError(f"BALL_CLASS_IDX={BALL_CLASS_IDX} but model has nc={nc} classes.")
    if not hasattr(model, "init_criterion"):
        raise RuntimeError("Model has no init_criterion(); cannot apply cost-sensitive loss.")

    pos_weight = torch.ones(nc, device=device)
    pos_weight[BALL_CLASS_IDX] = float(BALL_FN_WEIGHT)

    model.init_criterion = _make_patched_init_criterion(model.init_criterion, pos_weight)
    model.criterion = None

    print(f"[cost-sensitive] per-class pos_weight = {pos_weight.tolist()} "
          f"(ball idx {BALL_CLASS_IDX} weighted x{BALL_FN_WEIGHT})")


def main():
    if not os.path.isfile(DATA_YAML):
        raise FileNotFoundError(
            f"data.yaml not found at {DATA_YAML}. Edit DATA_YAML at the top of this file."
        )
    os.makedirs(PROJECT, exist_ok=True)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    else:
        print("WARNING: no CUDA device visible -- training will be very slow.")

    model = YOLO(MODEL)
    model.add_callback("on_train_start", apply_cost_sensitive)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        amp=True,
        cache=False,
        patience=30,
        cos_lr=True,
        optimizer="auto",
        close_mosaic=10,
        seed=0,
        verbose=True,
    )

    model.val(
        data=DATA_YAML, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
        project=PROJECT, name=NAME + "_val", exist_ok=True,
    )
    print("Done. best.pt is in:", os.path.join(PROJECT, NAME, "weights", "best.pt"))

if __name__ == "__main__":
    main()
