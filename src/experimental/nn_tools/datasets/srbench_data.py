"""Unified SRBench HDF5 data-loading utilities.

Provides functions to load lsr_transform and lsr_synth formulas from the
LLM-SRBench HDF5 file, returning item dicts suitable for property prediction
model evaluation and training:

    {data, var_mask, monotonicity, convexity, periodicity, mul_sep}

Also provides helpers for loading the v6 formula-level split.
"""
from __future__ import annotations

import json
import logging
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple

_logger = logging.getLogger(f"sr_agent.{__name__}")

DATA_ROOT = Path(__file__).resolve().parents[4] / "data" / "llm-srbench-data"
HDF5_PATH = DATA_ROOT / "lsr_bench_data.hdf5"
V6_SPLIT_PATH = DATA_ROOT / "property_label" / "v6_split.json"
LABEL_PATH = DATA_ROOT / "property_label" / "all_labels.json"

LSR_SYNTH_CATEGORIES = ["bio_pop_growth", "chem_react", "matsci", "phys_osc"]


# ---------------------------------------------------------------------------
# v6 formula-level split helpers
# ---------------------------------------------------------------------------

def load_v6_split() -> dict:
    """Load the v6 formula-level split (validation / test_a / reserved)."""
    with open(V6_SPLIT_PATH) as f:
        return json.load(f)


def _labels_from_entry(entry: dict, n_vars: int):
    """Extract padded label arrays from a v6_split formula_details entry."""
    lab = entry.get("labels") or entry.get("original_labels")
    vars_list = lab.get("variables") if "variables" in lab else entry.get("variables", [])

    mono_raw = lab["monotonicity"]
    conv_raw = lab["convexity"]
    period = lab["periodicity"]

    mono_labels = np.array(
        [mono_raw.get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64
    )
    conv_labels = np.array(
        [conv_raw.get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64
    )
    mono_labels = np.clip(mono_labels, 0, 3)
    conv_labels = np.clip(conv_labels, 0, 3)

    period = np.array(
        [period.get(v, 0) if isinstance(period, dict) else int(period)
         for v in vars_list[:n_vars]], dtype=np.int64
    )
    period = np.clip(period, 0, 1)

    sep = int(lab.get("multiplicative_separable", lab.get("group_separable", 0)))

    return mono_labels, conv_labels, period, sep


# ---------------------------------------------------------------------------
# Load lsr_transform formulas (with pre-computed labels from v6_split.json)
# ---------------------------------------------------------------------------

def load_lsr_transform_items(
    formula_names: List[str],
    max_var_num: int,
    sample_num: int,
    hdf5_splits: Tuple[str, ...] = ("validation", "test_a", "reserved"),
    seed: int = 12345,
) -> List[Dict]:
    """Load lsr_transform formulas with v6 domain-specific labels.

    Returns list of item dicts ready for model evaluation.
    """
    v6_split = load_v6_split()
    details = v6_split["formula_details"]

    split_name_to_entries = {}
    for split_name in details:
        for entry in details[split_name]:
            split_name_to_entries[entry["name"]] = entry

    items = []
    with h5py.File(HDF5_PATH, "r") as f:
        for name in formula_names:
            entry = split_name_to_entries.get(name)
            if entry is None:
                continue
            n_vars = entry["n_variables"]
            if n_vars > max_var_num:
                continue

            grp = f["lsr_transform"].get(name)
            if grp is None:
                continue

            arrays = []
            for split in hdf5_splits:
                if split in grp:
                    raw = grp[split]
                    if isinstance(raw, h5py.Dataset):
                        arrays.append(raw[:])
                    else:
                        for k in sorted(raw.keys()):
                            arrays.append(raw[k][:])

            if not arrays:
                continue

            raw = np.concatenate(arrays, axis=0)
            mask_fin = np.all(np.isfinite(raw), axis=1)
            raw = raw[mask_fin]
            if raw.shape[0] < 20:
                continue

            rng = np.random.default_rng(seed)
            S = min(sample_num, raw.shape[0])
            idx = rng.choice(raw.shape[0], S, replace=False)
            sampled = raw[idx]

            data = np.zeros((S, max_var_num + 1), dtype=np.float32)
            for i in range(n_vars):
                data[:, i] = sampled[:, i].astype(np.float32)
            data[:, -1] = sampled[:, -1].astype(np.float32)

            mono_labels, conv_labels, period, sep = _labels_from_entry(entry, n_vars)

            mono_padded = np.zeros(max_var_num, dtype=np.int64)
            conv_padded = np.zeros(max_var_num, dtype=np.int64)
            period_padded = np.zeros(max_var_num, dtype=np.int64)
            mono_padded[:n_vars] = mono_labels
            conv_padded[:n_vars] = conv_labels
            period_padded[:n_vars] = period

            var_mask = np.zeros(max_var_num, dtype=bool)
            var_mask[:n_vars] = True

            items.append({
                "data": torch.from_numpy(data),
                "var_mask": torch.from_numpy(var_mask),
                "monotonicity": torch.tensor(mono_padded, dtype=torch.long),
                "convexity": torch.tensor(conv_padded, dtype=torch.long),
                "periodicity": torch.tensor(period_padded, dtype=torch.long),
                "mul_sep": torch.tensor(sep, dtype=torch.long),
            })

    return items


# ---------------------------------------------------------------------------
# Load lsr_synth formulas (compute labels numerically on the fly)
# ---------------------------------------------------------------------------

def load_lsr_synth_items(
    categories: Optional[List[str]] = None,
    max_var_num: int = 5,
    sample_num: int = 200,
    hdf5_splits: Tuple[str, ...] = ("test",),
    seed: int = 12345,
) -> Dict[str, List[Dict]]:
    """Load lsr_synth formulas grouped by category, computing labels numerically.

    Returns dict mapping category name -> list of item dicts.
    """
    from .compute_labels import compute_all_labels

    if categories is None:
        categories = LSR_SYNTH_CATEGORIES

    result: Dict[str, List[Dict]] = {cat: [] for cat in categories}

    with h5py.File(HDF5_PATH, "r") as f:
        synth_grp = f["lsr_synth"]
        for cat in sorted(synth_grp.keys()):
            if cat not in result:
                continue
            items = []
            for name in sorted(synth_grp[cat].keys()):
                grp = synth_grp[cat][name]

                arrays = []
                for split in hdf5_splits:
                    if split in grp:
                        raw = grp[split]
                        if isinstance(raw, h5py.Dataset):
                            arrays.append(raw[:])
                        else:
                            for k in sorted(raw.keys()):
                                arrays.append(raw[k][:])

                if not arrays:
                    continue

                raw = np.concatenate(arrays, axis=0)
                mask_fin = np.all(np.isfinite(raw), axis=1)
                raw = raw[mask_fin]
                if raw.shape[0] < 20:
                    continue

                n_vars = raw.shape[1] - 1
                if n_vars > max_var_num:
                    continue

                rng = np.random.default_rng(seed)
                S = min(sample_num, raw.shape[0])
                idx = rng.choice(raw.shape[0], S, replace=False)
                sampled = raw[idx]

                X = np.zeros((S, max_var_num), dtype=np.float32)
                for i in range(n_vars):
                    X[:, i] = sampled[:, i].astype(np.float32)
                y = sampled[:, -1].astype(np.float32)

                labels = compute_all_labels(X, y, n_vars)

                data = np.zeros((S, max_var_num + 1), dtype=np.float32)
                data[:, :max_var_num] = X
                data[:, -1] = y

                mono_padded = np.zeros(max_var_num, dtype=np.int64)
                conv_padded = np.zeros(max_var_num, dtype=np.int64)
                period_padded = np.zeros(max_var_num, dtype=np.int64)
                mono_padded[:n_vars] = np.array(labels["monotonicity"][:n_vars], dtype=np.int64)
                conv_padded[:n_vars] = np.array(labels["convexity"][:n_vars], dtype=np.int64)
                period_padded[:n_vars] = np.array(labels["periodicity"][:n_vars], dtype=np.int64)

                var_mask = np.zeros(max_var_num, dtype=bool)
                var_mask[:n_vars] = True

                sep = int(labels["multiplicative_separable"])

                items.append({
                    "data": torch.from_numpy(data),
                    "var_mask": torch.from_numpy(var_mask),
                    "monotonicity": torch.tensor(mono_padded, dtype=torch.long),
                    "convexity": torch.tensor(conv_padded, dtype=torch.long),
                    "periodicity": torch.tensor(period_padded, dtype=torch.long),
                    "mul_sep": torch.tensor(sep, dtype=torch.long),
                    "formula_name": cat + "/" + name,
                })

            result[cat] = items

    return result


# ---------------------------------------------------------------------------
# Load SRBench items from all_labels.json (used by training data mixing)
# ---------------------------------------------------------------------------

def load_srbench_items(
    hdf5_path: Optional[str] = None,
    label_path: Optional[str] = None,
    max_var_num: int = 5,
    sample_num: int = 200,
    splits: Tuple[str, ...] = ("train",),
    seed: int = 42,
) -> List[Dict]:
    """Pre-load LLM-SRBench HDF5 data as list of dicts matching dataset format.

    Uses all_labels.json for label lookup. Data columns in HDF5 are ordered as
    [y, x0, x1, ...], which is transposed to [x0, x1, ..., y] in the output.
    """
    hdf5_p = Path(hdf5_path) if hdf5_path else HDF5_PATH
    label_p = Path(label_path) if label_path else LABEL_PATH
    if not hdf5_p.exists() or not label_p.exists():
        _logger.warning("SRBench data not found, skipping HDF5 mixing.")
        return []

    labels = json.load(open(label_p))
    ds_map = {
        "lsr_synth_chem_react": "chem_react",
        "lsr_synth_phys_osc": "phys_osc",
        "lsr_synth_matsci": "matsci",
    }

    items: List[Dict] = []
    rng = np.random.default_rng(seed)

    with h5py.File(hdf5_p, "r") as f:
        for lab in labels:
            ds_name = lab["dataset"]
            name = lab["name"]
            n_vars = lab["n_variables"]
            if n_vars > max_var_num or n_vars == 0:
                continue

            try:
                if ds_name == "lsr_transform":
                    grp = f["lsr_transform"][name]
                else:
                    grp = f["lsr_synth"][ds_map[ds_name]][name]

                arrays = []
                for split in splits:
                    if split in grp and isinstance(grp[split], h5py.Dataset):
                        arrays.append(grp[split][:])
                if not arrays:
                    continue
                raw = np.concatenate(arrays, axis=0)
            except Exception:
                continue

            mask_fin = np.all(np.isfinite(raw), axis=1)
            raw = raw[mask_fin]
            if raw.shape[0] < 20:
                continue

            S = min(raw.shape[0], sample_num)
            idx = rng.choice(raw.shape[0], S, replace=False)
            sampled = raw[idx]

            data = np.zeros((S, max_var_num + 1), dtype=np.float32)
            for i in range(min(n_vars, sampled.shape[1] - 1)):
                data[:, i] = sampled[:, i + 1].astype(np.float32)
            data[:, -1] = sampled[:, 0].astype(np.float32)

            vars_list = lab["variables"]
            mono_raw = np.array([lab["monotonicity"].get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64)
            conv_raw = np.array([lab["convexity"].get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64)
            period = np.array([lab["periodicity"].get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64)
            mono_labels = np.clip(mono_raw, 0, 3)
            mono_labels[mono_raw == 4] = 3
            conv_labels = np.clip(conv_raw, 0, 3)
            conv_labels[conv_raw == 4] = 3

            mono_padded = np.zeros(max_var_num, dtype=np.int64)
            conv_padded = np.zeros(max_var_num, dtype=np.int64)
            period_padded = np.zeros(max_var_num, dtype=np.int64)
            mono_padded[:n_vars] = mono_labels
            conv_padded[:n_vars] = conv_labels
            period_padded[:n_vars] = period

            var_mask = np.zeros(max_var_num, dtype=bool)
            var_mask[:n_vars] = True

            items.append({
                "data": torch.from_numpy(data),
                "var_mask": torch.from_numpy(var_mask),
                "monotonicity": torch.from_numpy(mono_padded),
                "convexity": torch.from_numpy(conv_padded),
                "periodicity": torch.from_numpy(period_padded),
                "mul_sep": torch.tensor(lab["multiplicative_separable"], dtype=torch.long),
            })

    _logger.info(f"Loaded {len(items)} SRBench items from splits={splits}")
    return items
