"""Load feature pickles, align targets, scale, and build tensor batches."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from catmof.sbu_quantum_ml.tasks import BpnnTask

# Per-atom-class true feature width before padding (must match featurization).
DEFAULT_CLASS_FEATURE_LENGTHS: Dict[int, int] = {
    0: 18,
    1: 75,
    2: 128,
    3: 128,
    4: 112,
    5: 264,
    6: 264,
    7: 264,
    8: 264,
    9: 264,
}

N_ATOM_CLASSES = 10

# Atom padding in :func:`prepare_data` when the feature bundle omits ``max_atoms_all_sbus``.
DEFAULT_MAX_ATOMS_PADDING = 180

# Second dimension of padded atom rows when the feature bundle omits ``max_col_length``.
DEFAULT_MAX_COL_LENGTH = 264

# Columns bundled as optional global SBU-level descriptors (from xTB parsing, etc.).
XTB_SCALAR_BLOCK = (
    "Final energy (Eh)",
    "HOMO-LUMO gap (eV)",
    "Dipole moment (Debye)",
    "Vertical IP (eV)",
    "Vertical EA (eV)",
    "Global Electrophilicity Index (eV)",
)


def _irreps_list_from_feature_bundle(data: dict) -> List[Any]:
    """Collapse per-SBU irreps into one Irreps per atom class (same as original script)."""
    atomic_irreps = data["class_based_atomic_irreps"]
    buckets = {i: set() for i in range(N_ATOM_CLASSES)}
    for sbu_irreps in atomic_irreps:
        for i, class_irreps in enumerate(sbu_irreps):
            buckets[i].update(class_irreps)
    return [list(buckets[i])[0] for i in range(N_ATOM_CLASSES)]


def _stack_xtb_block(data: dict) -> np.ndarray:
    cols = [np.asarray(data[c]) for c in XTB_SCALAR_BLOCK]
    return np.column_stack(cols)


def _max_atoms_from_feature_bundle(data: dict) -> int:
    """Use ``max_atoms_all_sbus`` from :func:`assemble_class_tiled_features` when present."""
    v = data.get("max_atoms_all_sbus")
    if v is None:
        return DEFAULT_MAX_ATOMS_PADDING
    n = int(v)
    return n if n > 0 else DEFAULT_MAX_ATOMS_PADDING


def _class_feature_lengths_from_bundle(data: dict) -> Dict[int, int]:
    """Per-tile coefficient width from assembly; fall back to :data:`DEFAULT_CLASS_FEATURE_LENGTHS`."""
    v = data.get("class_feature_lengths")
    if v is None or not isinstance(v, dict):
        return dict(DEFAULT_CLASS_FEATURE_LENGTHS)
    out = dict(DEFAULT_CLASS_FEATURE_LENGTHS)
    for key, val in v.items():
        ki = int(key)
        if 0 <= ki < N_ATOM_CLASSES:
            out[ki] = int(val)
    return out


def _max_col_length_from_bundle(data: dict) -> int:
    """Global pad width from assembly when present; else :data:`DEFAULT_MAX_COL_LENGTH`."""
    raw = data.get("max_col_length")
    if raw is not None:
        n = int(raw)
        if n > 0:
            return n
    return DEFAULT_MAX_COL_LENGTH


def geometry_from_feature_bundle(data: dict) -> Tuple[int, Dict[int, int], int]:
    """``max_atoms``, per-class coefficient widths, and padded row width — from pickle metadata."""
    max_atoms = _max_atoms_from_feature_bundle(data)
    lengths = _class_feature_lengths_from_bundle(data)
    max_col = _max_col_length_from_bundle(data)
    return max_atoms, lengths, max_col


def align_features_with_targets(
    sbus: np.ndarray,
    atomic_features: Sequence[Any],
    full_sbu_features: np.ndarray,
    targets_df: pd.DataFrame,
    task: BpnnTask,
) -> Tuple[np.ndarray, List[Any], np.ndarray, np.ndarray]:
    """Inner-join SBUs to DFT labels; drop rows with NaN targets."""
    df_t = targets_df[[task.sbu_column, task.target_column]].drop_duplicates(subset=[task.sbu_column])
    df_f = pd.DataFrame({"sbu": sbus, "_idx": np.arange(len(sbus))})
    merged = df_f.merge(df_t, left_on="sbu", right_on=task.sbu_column, how="inner")
    y = pd.to_numeric(merged[task.target_column], errors="coerce").values
    idx = merged["_idx"].to_numpy()
    mask = ~np.isnan(y)
    idx_kept = idx[mask]
    y_clean = y[mask].astype(np.float64)
    sbus_kept = sbus[idx_kept]
    atomic_kept = [atomic_features[i] for i in idx_kept]
    full_kept = full_sbu_features[idx_kept]
    return sbus_kept, atomic_kept, full_kept, y_clean


def load_or_create_splits(
    task: BpnnTask,
    features_pickle: Path,
    targets_csv: Path,
    splits_cache: Path,
    random_state: int = 42,
    test_size: float = 0.2,
    val_fraction_of_train: float = 0.125,
    force_refresh: bool = False,
) -> Tuple[Any, ...]:
    """
    Return
    (sbus_train, sbus_val, sbus_test,
     atomic_train, atomic_val, atomic_test,
     full_train, full_val, full_test,
     y_train, y_val, y_test,
     input_irreps,
     max_atoms,
     class_feature_lengths,
     max_col_length)
    where ``max_atoms``, ``class_feature_lengths``, and ``max_col_length`` are read from the
    feature pickle (see :func:`assemble_class_tiled_features`) when present, with fallbacks
    in :mod:`catmof.sbu_quantum_ml.training.data`.
    """
    if splits_cache.is_file() and not force_refresh:
        with open(splits_cache, "rb") as f:
            cached = pickle.load(f)
        with open(features_pickle, "rb") as f:
            data_refresh = pickle.load(f)
        max_atoms, class_feature_lengths, max_col_length = geometry_from_feature_bundle(data_refresh)
        if len(cached) == 13:
            return (*cached, max_atoms, class_feature_lengths, max_col_length)
        return (*cached[:-1], max_atoms, class_feature_lengths, max_col_length)

    with open(features_pickle, "rb") as f:
        data = pickle.load(f)

    max_atoms, class_feature_lengths, max_col_length = geometry_from_feature_bundle(data)
    sbus = np.asarray(data["sbu"])
    input_irreps = _irreps_list_from_feature_bundle(data)
    atomic_features = data["class_based_atomic_coeffs"]
    full_sbu_features = _stack_xtb_block(data)

    targets_df = pd.read_csv(targets_csv)
    sbus, atomic_features, full_sbu_features, y = align_features_with_targets(
        sbus, atomic_features, full_sbu_features, targets_df, task
    )

    idx = np.arange(len(sbus))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state)
    sbus_train, sbus_test = sbus[idx_train], sbus[idx_test]
    atomic_train = [atomic_features[i] for i in idx_train]
    atomic_test = [atomic_features[i] for i in idx_test]
    full_train, full_test = full_sbu_features[idx_train], full_sbu_features[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    idx_sub = np.arange(len(sbus_train))
    idx_tr, idx_va = train_test_split(
        idx_sub, test_size=val_fraction_of_train, random_state=random_state
    )
    sbus_train, sbus_val = sbus_train[idx_tr], sbus_train[idx_va]
    atomic_train = [atomic_train[i] for i in idx_tr]
    atomic_val = [atomic_train[i] for i in idx_va]
    full_train, full_val = full_train[idx_tr], full_train[idx_va]
    y_train, y_val = y_train[idx_tr], y_train[idx_va]

    bundle = (
        sbus_train,
        sbus_val,
        sbus_test,
        atomic_train,
        atomic_val,
        atomic_test,
        full_train,
        full_val,
        full_test,
        y_train,
        y_val,
        y_test,
        input_irreps,
        max_atoms,
    )
    splits_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_cache, "wb") as f:
        pickle.dump(bundle, f)
    return (*bundle, class_feature_lengths, max_col_length)


def normalize_atomic_features(
    normalizers: Dict[int, StandardScaler],
    atomic_features: List[List[np.ndarray]],
) -> List[List[np.ndarray]]:
    out: List[List[np.ndarray]] = []
    for sbu_tile_list in atomic_features:
        normalized: List[np.ndarray] = []
        for i in range(len(sbu_tile_list)):
            if sbu_tile_list[i].size == 0:
                normalized.append(np.array([]))
                continue
            normalized.append(normalizers[i].transform(sbu_tile_list[i]))
        out.append(normalized)
    return out


def normalize_features_labels(
    sbus_train: np.ndarray,
    atomic_features_train: List[List[np.ndarray]],
    atomic_features_val: List[List[np.ndarray]],
    atomic_features_test: List[List[np.ndarray]],
    target_train: np.ndarray,
    target_val: np.ndarray,
    target_test: np.ndarray,
    full_sbu_features_train: Optional[np.ndarray] = None,
    full_sbu_features_val: Optional[np.ndarray] = None,
    full_sbu_features_test: Optional[np.ndarray] = None,
) -> Tuple[
    List[List[np.ndarray]],
    List[List[np.ndarray]],
    List[List[np.ndarray]],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """Fit per-tile StandardScaler on train atoms; scale targets with a single scaler."""
    normalizers: Dict[int, StandardScaler] = {}
    n_tiles = len(atomic_features_train[0])
    for i in range(n_tiles):
        parts: List[np.ndarray] = []
        for j, _sbu in enumerate(sbus_train):
            tile = atomic_features_train[j][i]
            if tile.size:
                parts.append(tile)
        scaler = StandardScaler()
        if parts:
            scaler.fit(np.concatenate(parts, axis=0))
        normalizers[i] = scaler

    norm_train = normalize_atomic_features(normalizers, atomic_features_train)
    norm_val = normalize_atomic_features(normalizers, atomic_features_val)
    norm_test = normalize_atomic_features(normalizers, atomic_features_test)

    if full_sbu_features_train is not None:
        assert full_sbu_features_val is not None and full_sbu_features_test is not None
        full_scaler = StandardScaler()
        full_scaler.fit(full_sbu_features_train)
        full_tr = full_scaler.transform(full_sbu_features_train)
        full_va = full_scaler.transform(full_sbu_features_val)
        full_te = full_scaler.transform(full_sbu_features_test)
    else:
        full_tr = full_va = full_te = None

    t_tr = target_train.reshape(-1, 1)
    t_va = target_val.reshape(-1, 1)
    t_te = target_test.reshape(-1, 1)
    target_scaler = StandardScaler()
    target_scaler.fit(t_tr)
    y_tr = target_scaler.transform(t_tr)
    y_va = target_scaler.transform(t_va)
    y_te = target_scaler.transform(t_te)

    return (
        norm_train,
        norm_val,
        norm_test,
        full_tr,
        full_va,
        full_te,
        y_tr,
        y_va,
        y_te,
        target_scaler,
    )


def prepare_data(
    atomic_features: List[List[np.ndarray]],
    targets: np.ndarray,
    full_sbu_features: Optional[np.ndarray] = None,
    max_atoms: int = DEFAULT_MAX_ATOMS_PADDING,
    max_col_length: int = DEFAULT_MAX_COL_LENGTH,
    class_feature_lengths: Optional[Dict[int, int]] = None,
) -> Tuple[dict, torch.Tensor]:
    """Pad atoms to fixed (max_atoms, max_col_length); ghost rows use MLP index 10.

    Returns an ``inputs`` dict with ``atom_features``, ``mlp_mapping``, and optionally
    ``full_complex_features``. Per-class true widths are not stored here; pass the same
    ``class_feature_lengths`` dict into :class:`~catmof.sbu_quantum_ml.models.bp_net.FullEquivariantModel`.

    ``max_atoms``, ``class_feature_lengths``, and ``max_col_length`` default like
    :data:`DEFAULT_MAX_ATOMS_PADDING`, :data:`DEFAULT_CLASS_FEATURE_LENGTHS` (when
    ``class_feature_lengths`` is omitted), and :data:`DEFAULT_MAX_COL_LENGTH`. The training
    entrypoint passes values from :func:`geometry_from_feature_bundle` / :func:`load_or_create_splits`.
    """
    if class_feature_lengths is None:
        lengths = dict(DEFAULT_CLASS_FEATURE_LENGTHS)
    else:
        lengths = {
            i: int(class_feature_lengths.get(i, DEFAULT_CLASS_FEATURE_LENGTHS[i]))
            for i in range(N_ATOM_CLASSES)
        }
    sbu_final_features: List[np.ndarray] = []
    sbu_mlp_mapping: List[np.ndarray] = []

    for features_set in atomic_features:
        sbu_tensor = np.empty((0, max_col_length))
        mlp_mapping: List[int] = []
        for i, atom_class in enumerate(features_set):
            if atom_class.size == 0:
                continue
            mlp_mapping.extend([i] * atom_class.shape[0])
            padded = np.pad(
                atom_class,
                pad_width=((0, 0), (0, max_col_length - atom_class.shape[1])),
                mode="constant",
                constant_values=0,
            )
            sbu_tensor = np.concatenate([sbu_tensor, padded], axis=0)

        padded_sbu = np.pad(
            sbu_tensor,
            pad_width=((0, max_atoms - sbu_tensor.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        sbu_final_features.append(padded_sbu)

        mlp_mapping.extend([10] * (max_atoms - sbu_tensor.shape[0]))
        sbu_mlp_mapping.append(np.array(mlp_mapping, dtype=np.int32).reshape(-1, 1))

    inputs: dict = {
        "atom_features": torch.tensor(np.stack(sbu_final_features), dtype=torch.float32),
        "mlp_mapping": torch.tensor(np.stack(sbu_mlp_mapping), dtype=torch.int64),
    }
    if full_sbu_features is None:
        inputs["full_complex_features"] = torch.zeros((len(atomic_features), 0), dtype=torch.float32)
    else:
        inputs["full_complex_features"] = torch.tensor(np.asarray(full_sbu_features), dtype=torch.float32)

    target_tensor = torch.tensor(np.asarray(targets), dtype=torch.float32).reshape(-1, 1)
    return inputs, target_tensor


class DictDataset(Dataset):
    """Batched dict of tensors (preserves integer dtypes for masks)."""

    def __init__(self, inputs_dict: dict, targets: torch.Tensor):
        self.inputs_dict = inputs_dict
        self.targets = targets

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        inputs = {key: val[idx] for key, val in self.inputs_dict.items()}
        return inputs, self.targets[idx]
