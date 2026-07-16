"""Fine-tune v7 PropertyPredictionModel on reserved lsr_transform formulas.

Uses 29 reserved real-physics formulas (HDF5 train split) to adapt the
synthetically-pretrained v7 model, mixed with synthetic data to prevent
catastrophic forgetting.

Evaluation on 21 validation lsr_transform formulas for early stopping.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "experimental"))
sys.path.insert(0, str(ROOT / "src"))

import json
import time
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
from collections import Counter

from nn_tools.models import FloatEmbedder, DataEmbedder, PropertyPredictionModel
from nn_tools.datasets.generate_eq import BaseEqGenerator
from nn_tools.datasets.generate_data import BaseDataGenerator
from nn_tools.datasets.data_property_dataset import DataPropertyDataset, InfiniteSampler
from nn_tools.datasets.compute_labels import MONO_CLASSES, CONV_CLASSES
from nn_tools.datasets.srbench_data import load_v6_split, load_lsr_transform_items
from sr_agent.utils import setup_logging, seed_all

_logger = logging.getLogger("sr_agent.finetune_v7")

DATA_ROOT = ROOT / "data" / "llm-srbench-data"
HDF5_PATH = DATA_ROOT / "lsr_bench_data.hdf5"
LABEL_PATH = DATA_ROOT / "property_label" / "all_labels.json"


def load_reserved_items(max_var_num, sample_num, seed=42):
    """Load the 29 reserved lsr_transform formulas from HDF5 train split."""
    v6_split = load_v6_split()
    reserved_names = v6_split.get("reserved", [])
    _logger.info(f"Loading {len(reserved_names)} reserved formulas...")
    items = load_lsr_transform_items(
        reserved_names, max_var_num, sample_num,
        hdf5_splits=("train",), seed=seed,
    )
    _logger.info(f"Loaded {len(items)} reserved items (some may be filtered by n_vars)")
    return items


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    if isinstance(saved_args, dict):
        saved_args = argparse.Namespace(**saved_args)
    return ckpt, saved_args


def eval_on_items(model, float_emb, data_emb, items, device):
    """Evaluate model on a list of items, return per-property metrics."""
    model.eval(); float_emb.eval(); data_emb.eval()
    preds_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
    gts_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
    masks_all = []
    sep_preds, sep_gts = [], []

    with torch.no_grad():
        for item in items:
            data_t = item["data"].unsqueeze(0).to(device)
            B, S = data_t.shape[:2]
            val_emb = float_emb(data_t).flatten(0, 1)
            d_emb = data_emb.pool(val_emb).reshape(B, S, -1)
            out = model(d_emb)

            mask = item["var_mask"].numpy()
            masks_all.append(mask)
            for t in ("monotonicity", "convexity", "periodicity"):
                preds_dict[t].append(out[t][0].cpu().argmax(dim=-1).numpy())
                gts_dict[t].append(item[t].numpy())
            sep_preds.append(out["multiplicative_separable"][0].cpu().argmax().item())
            sep_gts.append(item["mul_sep"].item())

    results = {}
    for t, nc in [("monotonicity", MONO_CLASSES), ("convexity", CONV_CLASSES), ("periodicity", 2)]:
        pred = np.concatenate(preds_dict[t])
        gt = np.concatenate(gts_dict[t])
        mask = np.concatenate(masks_all)
        pred_m, gt_m = pred[mask], gt[mask]
        acc = float((pred_m == gt_m).mean()) if len(gt_m) > 0 else 0.0
        f1s = []
        for c in range(nc):
            tp = ((pred_m == c) & (gt_m == c)).sum()
            fp = ((pred_m == c) & (gt_m != c)).sum()
            fn = ((pred_m != c) & (gt_m == c)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            f1s.append(f1)
        results[t] = {"accuracy": acc, "macro_f1": float(np.mean(f1s))}

    sp, sg = np.array(sep_preds), np.array(sep_gts)
    results["separability"] = {"accuracy": float((sp == sg).mean())}
    mono_f1 = results["monotonicity"]["macro_f1"]
    conv_f1 = results["convexity"]["macro_f1"]
    results["target_f1"] = 2 * mono_f1 * conv_f1 / max(mono_f1 + conv_f1, 1e-8)
    return results


def train_on_batch(model, float_emb, data_emb, batch, criterion_dict, device):
    """Forward + loss on a batch from DataPropertyDataset."""
    data = batch["data"].to(device)
    B, S, _ = data.shape
    val_emb = float_emb(data).flatten(0, 1)
    d_emb = data_emb.pool(val_emb).reshape(B, S, -1)
    out = model(d_emb)

    var_mask = batch["var_mask"].to(device)
    mono_gt = batch["monotonicity"].to(device)
    conv_gt = batch["convexity"].to(device)
    period_gt = batch["periodicity"].to(device)
    sep_gt = batch["mul_sep"].to(device)

    flat_mask = var_mask.reshape(-1)
    mono_pred = out["monotonicity"].reshape(-1, MONO_CLASSES)[flat_mask]
    conv_pred = out["convexity"].reshape(-1, CONV_CLASSES)[flat_mask]
    period_pred = out["periodicity"].reshape(-1, 2)[flat_mask]

    loss = (
        criterion_dict["monotonicity"](mono_pred, mono_gt.reshape(-1)[flat_mask])
        + criterion_dict["convexity"](conv_pred, conv_gt.reshape(-1)[flat_mask])
        + criterion_dict["periodicity"](period_pred, period_gt.reshape(-1)[flat_mask])
        + criterion_dict["sep"](out["multiplicative_separable"], sep_gt)
    )
    return loss


def main(args):
    _logger.info(f"Fine-tuning v7: checkpoint={args.checkpoint}, device={args.device}")

    ckpt, saved_args = load_checkpoint(args.checkpoint, args.device)
    for attr in ("d_model", "nhead", "num_encoder_layers", "dim_feedforward",
                 "dropout", "max_var_num", "data_pooling"):
        setattr(args, attr, getattr(saved_args, attr, getattr(args, attr, None)))

    float_emb = FloatEmbedder(d_model=args.d_model).to(args.device)
    data_emb = DataEmbedder(d_model=args.d_model, pooling=args.data_pooling,
                            float_embedder=float_emb).to(args.device)
    model = PropertyPredictionModel(args).to(args.device)

    model.load_state_dict(ckpt["model"])
    float_emb.load_state_dict(ckpt["float_embedder"])
    data_emb.load_state_dict(ckpt["data_embedder"])
    _logger.info("Loaded v7 checkpoint weights")

    reserved_items = load_reserved_items(args.max_var_num, args.sample_num, seed=args.seed)

    v6_split = load_v6_split()
    val_items = load_lsr_transform_items(
        v6_split["validation"], args.max_var_num, args.sample_num,
        hdf5_splits=("test",), seed=12345,
    )
    _logger.info(f"Validation set: {len(val_items)} formulas")

    eq_gen = BaseEqGenerator.create(
        "gplearn", n_variables=args.max_var_num, random_seed=args.seed + 1000,
        const_range=None, depth_range=(1, 6), n_var_range=(1, args.max_var_num + 1),
    )
    data_gen = BaseDataGenerator.create(
        "uniform", sample_num=args.sample_num, random_seed=args.seed + 1000,
        range=(-10.0, 10.0),
    )
    synth_ds = DataPropertyDataset(
        max_var_num=args.max_var_num, eq_generator=eq_gen, data_generator=data_gen,
        sample_num=args.sample_num, n_samples=None, random_state=args.seed + 1000,
        max_per_signature=999999999, range_augment=True,
        use_sympy_labels=True,
        srbench_items=reserved_items,
        srbench_mix_ratio=args.real_mix_ratio,
        noise_std=0.01, scale_augment=True, permute_vars=True,
        reject_trivial_prob=0.3, combo_augment_prob=0.2,
    )

    train_loader = D.DataLoader(
        synth_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=synth_ds.collate_fn, sampler=InfiniteSampler(),
    )

    label_smoothing = args.label_smoothing
    mono_weights = torch.tensor([0.5, 5.0, 5.0, 5.0], device=args.device)
    conv_weights = torch.tensor([0.5, 5.0, 5.0, 5.0], device=args.device)
    period_weights = torch.tensor([1.0, 5.0], device=args.device)
    criterion_dict = {
        "monotonicity": nn.CrossEntropyLoss(weight=mono_weights, label_smoothing=label_smoothing),
        "convexity": nn.CrossEntropyLoss(weight=conv_weights, label_smoothing=label_smoothing),
        "periodicity": nn.CrossEntropyLoss(weight=period_weights),
        "sep": nn.CrossEntropyLoss(label_smoothing=label_smoothing),
    }

    all_params = list(model.parameters()) + list(float_emb.parameters()) + list(data_emb.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01,
    )

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    best_f1 = -1.0
    best_loss = float("inf")
    patience_left = args.patience
    history = []
    start_time = time.time()

    _logger.info(f"Fine-tuning: max_steps={args.max_steps}, lr={args.lr}, "
                 f"batch_size={args.batch_size}, real_mix_ratio={args.real_mix_ratio}")

    baseline = eval_on_items(model, float_emb, data_emb, val_items, args.device)
    _logger.info(f"[Baseline] mono_acc={baseline['monotonicity']['accuracy']:.3f} "
                 f"conv_acc={baseline['convexity']['accuracy']:.3f} "
                 f"period_acc={baseline['periodicity']['accuracy']:.3f} "
                 f"sep_acc={baseline['separability']['accuracy']:.3f} "
                 f"target_f1={baseline['target_f1']:.3f}")

    for step, batch in enumerate(train_loader):
        if step >= args.max_steps:
            break

        model.train(); float_emb.train(); data_emb.train()
        loss = train_on_batch(model, float_emb, data_emb, batch, criterion_dict, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if (step + 1) % args.eval_every == 0 or step == 0:
            metrics = eval_on_items(model, float_emb, data_emb, val_items, args.device)
            elapsed = time.time() - start_time
            speed = (step + 1) * args.batch_size / elapsed

            record = {"step": step + 1, "eval": metrics, "train_loss": loss.item()}
            history.append(record)

            improved = False
            markers = []
            if metrics["target_f1"] > best_f1:
                best_f1 = metrics["target_f1"]
                torch.save({
                    "step": step + 1, "args": vars(args), "base_checkpoint": args.checkpoint,
                    "model": model.state_dict(),
                    "float_embedder": float_emb.state_dict(),
                    "data_embedder": data_emb.state_dict(),
                }, save_path / "best_f1.pth")
                improved = True
                markers.append("BEST_F1")

            if loss.item() < best_loss:
                best_loss = loss.item()
                improved = True
                markers.append("BEST_LOSS")

            marker = f" *{'|'.join(markers)}*" if markers else ""
            _logger.info(
                f"[step {step+1:>5d} | {elapsed/3600:.1f}h] "
                f"loss={loss.item():.4f}  "
                f"mono_acc={metrics['monotonicity']['accuracy']:.3f}  "
                f"conv_acc={metrics['convexity']['accuracy']:.3f}  "
                f"period_acc={metrics['periodicity']['accuracy']:.3f}  "
                f"sep_acc={metrics['separability']['accuracy']:.3f}  "
                f"target_f1={metrics['target_f1']:.3f}  "
                f"speed={speed:.0f} eq/s{marker}"
            )

            if improved:
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    _logger.info(f"Early stopping at step {step+1}")
                    break

    torch.save({
        "step": step + 1, "args": vars(args), "base_checkpoint": args.checkpoint,
        "model": model.state_dict(),
        "float_embedder": float_emb.state_dict(),
        "data_embedder": data_emb.state_dict(),
    }, save_path / "last.pth")

    with open(save_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    import shutil
    best_f1_pth = save_path / "best_f1.pth"
    if best_f1_pth.exists():
        shutil.copy2(str(best_f1_pth), str(save_path / "best.pth"))

    elapsed = time.time() - start_time
    _logger.info(f"Fine-tuning finished in {elapsed:.0f}s ({elapsed/3600:.1f}h), best_f1={best_f1:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="v7 best_f1 checkpoint path")
    p.add_argument("--save_path", required=True, help="Output directory")
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--real_mix_ratio", type=float, default=0.7,
                   help="Prob of returning a reserved formula vs synthetic (0.7=70%% real)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample_num", type=int, default=200)
    p.add_argument("--max_var_num", type=int, default=5)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_encoder_layers", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--data_pooling", default="attention")
    args = p.parse_args()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    setup_logging(info_level="info", exp_name="finetune_v7",
                  save_path=Path(args.save_path) / "info.log", force=True)
    _logger.info(f"Args: {args}")
    main(args)
