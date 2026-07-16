"""Evaluate PropertyPredictionModel on formula-level test sets.

Unified evaluation script for v6+ property prediction models.

Test sets (no formula overlap with training):
  A) lsr_transform held-out formulas (formula-level split, never seen during training)
  B) Unseen synthetic formulas (different seed from training)
  C) Seen-seed synthetic formulas with new data range
  D) lsr_synth formulas grouped by domain (bio_pop_growth, chem_react, matsci, phys_osc)

All training uses pure synthetic data, so lsr_transform formulas are truly held-out.

Flags:
  --use_sympy_labels   Use SymPy-based labels for synthetic test sets (default: True)
  --output             Custom output filename (default: eval_results.json)
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "experimental"))
sys.path.insert(0, str(ROOT / "src"))

import json
import torch
import logging
import argparse
import numpy as np
from nn_tools.models import FloatEmbedder, DataEmbedder, PropertyPredictionModel
from nn_tools.datasets.generate_eq import BaseEqGenerator
from nn_tools.datasets.generate_data import BaseDataGenerator
from nn_tools.datasets.data_property_dataset import DataPropertyDataset
from nn_tools.datasets.compute_labels import MONO_CLASSES, CONV_CLASSES
from nn_tools.datasets.srbench_data import (
    load_v6_split, load_lsr_transform_items, load_lsr_synth_items,
    LSR_SYNTH_CATEGORIES,
)

_logger = logging.getLogger("sr_agent.eval_property")


def load_model(args, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = argparse.Namespace(**ckpt["args"]) if isinstance(ckpt["args"], dict) else ckpt["args"]
    for attr in ("d_model", "nhead", "num_encoder_layers", "dim_feedforward", "dropout", "max_var_num", "data_pooling"):
        setattr(args, attr, getattr(saved_args, attr, getattr(args, attr, None)))
    float_emb = FloatEmbedder(d_model=args.d_model).to(device)
    data_emb = DataEmbedder(d_model=args.d_model, pooling=args.data_pooling, float_embedder=float_emb).to(device)
    model = PropertyPredictionModel(args).to(device)
    model.load_state_dict(ckpt["model"])
    float_emb.load_state_dict(ckpt["float_embedder"])
    data_emb.load_state_dict(ckpt["data_embedder"])
    model.eval(); float_emb.eval(); data_emb.eval()
    return model, float_emb, data_emb


def predict_one(model, float_emb, data_emb, data_tensor, device):
    with torch.no_grad():
        data = data_tensor.to(device)
        B, S = data.shape[:2]
        val_emb = float_emb(data).flatten(0, 1)
        d_emb = data_emb.pool(val_emb).reshape(B, S, -1)
        out = model(d_emb)
    return {k: v.cpu() for k, v in out.items()}


def per_task_metrics(all_preds, all_gts, all_masks, n_classes):
    pred = np.concatenate(all_preds)
    gt = np.concatenate(all_gts)
    mask = np.concatenate(all_masks)
    pred_m, gt_m = pred[mask], gt[mask]
    acc = (pred_m == gt_m).mean() if len(gt_m) > 0 else 0.0
    f1s, per_class = [], {}
    for c in range(n_classes):
        tp = ((pred_m == c) & (gt_m == c)).sum()
        fp = ((pred_m == c) & (gt_m != c)).sum()
        fn = ((pred_m != c) & (gt_m == c)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(f1)
        per_class[int(c)] = {"precision": float(prec), "recall": float(rec),
                             "f1": float(f1), "support": int((gt_m == c).sum())}
    return {"accuracy": float(acc), "macro_f1": float(np.mean(f1s)), "per_class": per_class}


def sep_metrics(sep_preds, sep_gts):
    preds, gts = np.array(sep_preds), np.array(sep_gts)
    acc = float((preds == gts).mean())
    f1s, per_class = [], {}
    for c in range(2):
        tp = ((preds == c) & (gts == c)).sum()
        fp = ((preds == c) & (gts != c)).sum()
        fn = ((preds != c) & (gts == c)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(f1)
        per_class[int(c)] = {"precision": float(prec), "recall": float(rec),
                             "f1": float(f1), "support": int((gts == c).sum())}
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)), "per_class": per_class}


def eval_lsr_transform(args, model, float_emb, data_emb, formula_names, test_name, hdf5_splits=("test",)):
    """Evaluate on lsr_transform formulas from the v6 split."""
    items = load_lsr_transform_items(
        formula_names, args.max_var_num, args.sample_num,
        hdf5_splits=hdf5_splits, seed=12345,
    )
    if not items:
        _logger.warning(f"No valid formulas for {test_name}")
        return None

    preds_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
    gts_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
    masks_all = []
    sep_preds_list, sep_gts_list = [], []

    for item in items:
        data_t = item["data"].unsqueeze(0)
        out = predict_one(model, float_emb, data_emb, data_t, args.device)
        mask = item["var_mask"].numpy()
        masks_all.append(mask)
        for t in ("monotonicity", "convexity", "periodicity"):
            preds_dict[t].append(out[t][0].argmax(dim=-1).numpy())
            gts_dict[t].append(item[t].numpy())
        sep_preds_list.append(out["multiplicative_separable"][0].argmax().item())
        sep_gts_list.append(item["mul_sep"].item())

    _logger.info(f"  {test_name}: evaluated {len(items)} formulas")
    results = {}
    for t, nc in [("monotonicity", MONO_CLASSES), ("convexity", CONV_CLASSES), ("periodicity", 2)]:
        results[t] = per_task_metrics(preds_dict[t], gts_dict[t], masks_all, nc)
    results["separability"] = sep_metrics(sep_preds_list, sep_gts_list)
    results["n_formulas"] = len(items)
    return results


def eval_synthetic(args, model, float_emb, data_emb, seed, n_samples, test_name,
                   range_augment=True, data_range=None):
    """Evaluate on synthetic formulas."""
    eq_gen = BaseEqGenerator.create(
        "gplearn", n_variables=args.max_var_num, random_seed=seed,
        const_range=None, depth_range=(1, 6), n_var_range=(1, args.max_var_num + 1),
    )
    data_min, data_max = data_range if data_range else (args.data_min, args.data_max)
    data_gen = BaseDataGenerator.create(
        "uniform", sample_num=args.sample_num, random_seed=seed,
        range=(data_min, data_max),
    )
    ds = DataPropertyDataset(
        max_var_num=args.max_var_num, eq_generator=eq_gen, data_generator=data_gen,
        sample_num=args.sample_num, n_samples=n_samples, random_state=seed,
        max_per_signature=9999, range_augment=range_augment,
        use_sympy_labels=args.use_sympy_labels,
    )

    preds_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
    gts_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
    masks_all = []
    sep_preds_list, sep_gts_list = [], []

    for i in range(n_samples):
        item = ds[i]
        data_t = item["data"].unsqueeze(0)
        out = predict_one(model, float_emb, data_emb, data_t, args.device)
        mask = item["var_mask"].numpy()
        masks_all.append(mask)
        for t in ("monotonicity", "convexity", "periodicity"):
            preds_dict[t].append(out[t][0].argmax(dim=-1).numpy())
            gts_dict[t].append(item[t].numpy())
        sep_preds_list.append(out["multiplicative_separable"][0].argmax().item())
        sep_gts_list.append(item["mul_sep"].item())

    results = {}
    for t, nc in [("monotonicity", MONO_CLASSES), ("convexity", CONV_CLASSES), ("periodicity", 2)]:
        results[t] = per_task_metrics(preds_dict[t], gts_dict[t], masks_all, nc)
    results["separability"] = sep_metrics(sep_preds_list, sep_gts_list)
    results["n_formulas"] = n_samples
    return results


def eval_lsr_synth(args, model, float_emb, data_emb, hdf5_splits=("test",)):
    """Evaluate on lsr_synth formulas (Test D), grouped by category."""
    synth_items = load_lsr_synth_items(
        categories=LSR_SYNTH_CATEGORIES,
        max_var_num=args.max_var_num, sample_num=args.sample_num,
        hdf5_splits=hdf5_splits, seed=12345,
    )
    all_results = {}
    for cat, items in synth_items.items():
        if not items:
            _logger.warning(f"  No valid formulas for lsr_synth/{cat}")
            continue
        preds_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
        gts_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
        masks_all = []
        sep_preds_list, sep_gts_list = [], []

        for item in items:
            data_t = item["data"].unsqueeze(0)
            out = predict_one(model, float_emb, data_emb, data_t, args.device)
            mask = item["var_mask"].numpy()
            masks_all.append(mask)
            for t in ("monotonicity", "convexity", "periodicity"):
                preds_dict[t].append(out[t][0].argmax(dim=-1).numpy())
                gts_dict[t].append(item[t].numpy())
            sep_preds_list.append(out["multiplicative_separable"][0].argmax().item())
            sep_gts_list.append(item["mul_sep"].item())

        cat_results = {}
        for t, nc in [("monotonicity", MONO_CLASSES), ("convexity", CONV_CLASSES), ("periodicity", 2)]:
            cat_results[t] = per_task_metrics(preds_dict[t], gts_dict[t], masks_all, nc)
        cat_results["separability"] = sep_metrics(sep_preds_list, sep_gts_list)
        cat_results["n_formulas"] = len(items)
        all_results[cat] = cat_results
        _logger.info(f"  {cat}: {len(items)} formulas")

    return all_results


def main(args):
    model, float_emb, data_emb = load_model(args, args.checkpoint, args.device)
    _logger.info(f"Model loaded from {args.checkpoint}")
    _logger.info(f"use_sympy_labels={args.use_sympy_labels}")

    v6_split = load_v6_split()
    results = {}

    _logger.info("=== Test A: Held-out lsr_transform formulas ===")
    results["test_a_lsr_transform"] = eval_lsr_transform(
        args, model, float_emb, data_emb,
        v6_split["test_a"], "lsr_transform_held_out", hdf5_splits=("test",))
    if results["test_a_lsr_transform"]:
        for t in ("monotonicity", "convexity", "periodicity"):
            r = results["test_a_lsr_transform"][t]
            _logger.info(f"  {t}: acc={r['accuracy']:.3f}, macro_f1={r['macro_f1']:.3f}")
        sr = results["test_a_lsr_transform"]["separability"]
        _logger.info(f"  sep: acc={sr['accuracy']:.3f}, macro_f1={sr['macro_f1']:.3f}")

    _logger.info("=== Test B: Unseen synthetic formulas ===")
    results["test_b_unseen_synthetic"] = eval_synthetic(
        args, model, float_emb, data_emb,
        seed=9999, n_samples=args.n_test, test_name="unseen_synthetic")
    for t in ("monotonicity", "convexity", "periodicity"):
        r = results["test_b_unseen_synthetic"][t]
        _logger.info(f"  {t}: acc={r['accuracy']:.3f}, macro_f1={r['macro_f1']:.3f}")
    sr = results["test_b_unseen_synthetic"]["separability"]
    _logger.info(f"  sep: acc={sr['accuracy']:.3f}, macro_f1={sr['macro_f1']:.3f}")

    _logger.info("=== Test C: Seen-seed synthetic, new range ===")
    results["test_c_seen_new_range"] = eval_synthetic(
        args, model, float_emb, data_emb,
        seed=42, n_samples=args.n_test, test_name="seen_seed_new_range",
        range_augment=False, data_range=(-20.0, 20.0))
    for t in ("monotonicity", "convexity", "periodicity"):
        r = results["test_c_seen_new_range"][t]
        _logger.info(f"  {t}: acc={r['accuracy']:.3f}, macro_f1={r['macro_f1']:.3f}")
    sr = results["test_c_seen_new_range"]["separability"]
    _logger.info(f"  sep: acc={sr['accuracy']:.3f}, macro_f1={sr['macro_f1']:.3f}")

    _logger.info("=== Test D: lsr_synth formulas (4 categories) ===")
    results["test_d_lsr_synth"] = eval_lsr_synth(args, model, float_emb, data_emb,
                                                  hdf5_splits=("test",))
    for cat, cr in results["test_d_lsr_synth"].items():
        _logger.info(f"  [{cat}] n={cr['n_formulas']}")
        for t in ("monotonicity", "convexity", "periodicity"):
            _logger.info(f"    {t}: acc={cr[t]['accuracy']:.3f}, macro_f1={cr[t]['macro_f1']:.3f}")
        _logger.info(f"    sep: acc={cr['separability']['accuracy']:.3f}, "
                     f"macro_f1={cr['separability']['macro_f1']:.3f}")

    _logger.info("=== Validation: lsr_transform val formulas ===")
    results["val_lsr_transform"] = eval_lsr_transform(
        args, model, float_emb, data_emb,
        v6_split["validation"], "lsr_transform_val", hdf5_splits=("test",))
    if results["val_lsr_transform"]:
        for t in ("monotonicity", "convexity", "periodicity"):
            r = results["val_lsr_transform"][t]
            _logger.info(f"  {t}: acc={r['accuracy']:.3f}, macro_f1={r['macro_f1']:.3f}")

    out_name = args.output if args.output else "eval_results.json"
    out_path = Path(args.checkpoint).parent / out_name
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _logger.info(f"Results saved to {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_test", type=int, default=500)
    p.add_argument("--sample_num", type=int, default=200)
    p.add_argument("--data_min", type=float, default=-10.0)
    p.add_argument("--data_max", type=float, default=10.0)
    p.add_argument("--max_var_num", type=int, default=5)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_encoder_layers", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--data_pooling", default="attention")
    p.add_argument("--use_sympy_labels", action=argparse.BooleanOptionalAction, default=True,
                   help="Use SymPy-based labels for synthetic test sets (default: True)")
    p.add_argument("--output", default=None,
                   help="Output filename (default: eval_results.json)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(args)
