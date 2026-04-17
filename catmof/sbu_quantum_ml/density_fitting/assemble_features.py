"""Merge per-SBU DF coefficient pickles with xTB scalars → ``density_fitting_features.pkl`` layout.

Algorithm matches the standalone MO featurization script: per-class max irrep/coeff length across
structures, pad shorter atoms to the longest in each class, join xTB CSV scalars. Identifiers and
the xTB CSV join column use ``sbu`` throughout (:data:`DEFAULT_ATOM_CLASS_GROUPS`,
``sbu_column_xtb``). Coefficient pickles use keys ``sbu``, ``atom_list``, ``atom_irreps``,
``coefficients``. The output pickle also stores ``max_atoms_all_sbus`` for atom padding in training.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

# Default atom-class tiling.
DEFAULT_ATOM_CLASS_GROUPS: Mapping[int, List[str]] = {
    1: ["H"],
    2: ["C", "Si", "Ge", "Sn"],
    3: ["N", "P", "As", "Sb"],
    4: ["O", "S", "Se", "Te"],
    5: ["F", "Cl", "Br", "I"],
    6: ["Mn"],
    7: ["Fe"],
    8: ["Co"],
    9: ["Ni"],
    10: ["Cu"],
}


def _parse_irreps(irreps: Any) -> Dict[int, int]:
    irrep_dict: Dict[int, int] = {}
    matches = re.findall(r"(\d+)x(\d+)([eo])", str(irreps))
    for count, lstr, _parity in matches:
        l = int(lstr)
        irrep_dict[l] = int(count)
    return irrep_dict


def _pad_coeffs(coeffs: Sequence[float], irreps: Any, longest_irreps: Any) -> List[float]:
    irreps_dict = _parse_irreps(irreps)
    longest_dict = _parse_irreps(longest_irreps)
    padded: List[float] = []
    coeff_index = 0
    for l in sorted(longest_dict.keys()):
        max_count = longest_dict[l]
        current_count = irreps_dict.get(l, 0)
        if current_count > 0:
            take = current_count * (2 * l + 1)
            padded.extend(coeffs[coeff_index : coeff_index + take])
            coeff_index += take
        missing = max_count - current_count
        if missing > 0:
            padded.extend([0.0] * (missing * (2 * l + 1)))
    return padded


def _load_coeff_bundle(data: dict) -> tuple:
    ids = data["sbu"]
    atom_lists = data["atom_list"]
    atom_irreps_list = data["atom_irreps"]
    coefficients_list = data["coefficients"]
    return list(ids), atom_lists, atom_irreps_list, coefficients_list


def assemble_class_tiled_features(
    coeff_pickle_path: Path,
    xtb_csv_path: Path,
    out_pickle_path: Path,
    atom_class_groups: Mapping[int, List[str]] = DEFAULT_ATOM_CLASS_GROUPS,
    sbu_column_xtb: str = "sbu",
) -> Path:
    """
    Build training pickle with keys ``sbu``, ``class_based_atomic_irreps``, ``class_based_atomic_coeffs``,
    ``max_atoms_all_sbus``, and xTB scalar columns matching :mod:`catmof.sbu_quantum_ml.training.data`.
    """
    coeff_pickle_path = Path(coeff_pickle_path)
    xtb_csv_path = Path(xtb_csv_path)
    out_pickle_path = Path(out_pickle_path)

    with open(coeff_pickle_path, "rb") as f:
        raw = pickle.load(f)

    sbu_ids, atom_lists, atom_irreps_list, coefficients_list = _load_coeff_bundle(raw)

    n_classes = len(atom_class_groups)
    max_lengths = [-np.inf] * n_classes
    max_irreps: List[Any] = [""] * n_classes
    max_at_types: List[str] = [""] * n_classes

    sbu_based_at_class_irreps: List[List[List[Any]]] = []
    sbu_based_at_class_coeffs: List[List[List[np.ndarray]]] = []
    max_atoms_all_sbus = max((len(al) for al in atom_lists), default=0)

    class_keys = sorted(atom_class_groups.keys())

    for i, _sbu in enumerate(sbu_ids):
        atom_list = atom_lists[i]
        atom_irreps = atom_irreps_list[i]
        coefficients = coefficients_list[i]

        at_class_irreps: List[List[Any]] = []
        at_class_coeffs: List[List[np.ndarray]] = []
        for j, at_class in enumerate(class_keys):
            syms = atom_class_groups[at_class]
            this_irreps: List[Any] = []
            this_coeffs: List[np.ndarray] = []
            for k, at in enumerate(atom_list):
                if at in syms:
                    this_irreps.append(atom_irreps[k])
                    ck = np.asarray(coefficients[k], dtype=float)
                    this_coeffs.append(ck)
                    clen = int(ck.size)
                    if clen > max_lengths[j]:
                        max_lengths[j] = clen
                        max_irreps[j] = atom_irreps[k]
                        max_at_types[j] = at
            at_class_irreps.append(this_irreps)
            at_class_coeffs.append(this_coeffs)

        sbu_based_at_class_irreps.append(at_class_irreps)
        sbu_based_at_class_coeffs.append(at_class_coeffs)

    final_irreps: List[List[List[Any]]] = []
    final_coeffs: List[List[np.ndarray]] = []

    for i, _sbu in enumerate(sbu_ids):
        this_ir = sbu_based_at_class_irreps[i]
        this_co = sbu_based_at_class_coeffs[i]
        final_class_irreps: List[Any] = []
        final_class_coeffs: List[np.ndarray] = []
        for j, class_irreps in enumerate(this_ir):
            longest = max_irreps[j]
            padded_ir: List[Any] = []
            padded_co: List[List[float]] = []
            for k, ir in enumerate(class_irreps):
                if ir == longest:
                    padded_ir.append(ir)
                    padded_co.append(list(this_co[j][k].ravel()))
                else:
                    padded_co.append(
                        _pad_coeffs(list(this_co[j][k].ravel()), ir, longest)
                    )
                    padded_ir.append(longest)
            final_class_irreps.append(padded_ir)
            final_class_coeffs.append(np.array(padded_co, dtype=float))
        final_irreps.append(final_class_irreps)
        final_coeffs.append(final_class_coeffs)

    xtb_df = pd.read_csv(xtb_csv_path)
    if sbu_column_xtb not in xtb_df.columns:
        raise KeyError(f"Column {sbu_column_xtb!r} not in {xtb_csv_path}")
    colmap = {
        "Final energy (Eh)": "Final energy (Eh)",
        "HOMO-LUMO gap (eV)": "HOMO-LUMO gap (eV)",
        "Dipole moment (Debye)": "Dipole moment (Debye)",
        "Vertical IP (eV)": "Vertical IP (eV)",
        "Vertical EA (eV)": "Vertical EA (eV)",
        "Global Electrophilicity Index (eV)": "Global Electrophilicity Index (eV)",
    }
    missing = [c for c in colmap if c not in xtb_df.columns]
    if missing:
        raise KeyError(f"xTB CSV missing columns: {missing}")

    mapping = {
        row[sbu_column_xtb]: tuple(row[c] for c in colmap)
        for _, row in xtb_df.iterrows()
    }

    xtb_energies, xtb_hl, xtb_dip, xtb_vip, xtb_vea, xtb_vom = [], [], [], [], [], []
    for sid in sbu_ids:
        if sid not in mapping:
            raise KeyError(f"SBU {sid!r} missing from xTB CSV")
        t = mapping[sid]
        xtb_energies.append(t[0])
        xtb_hl.append(t[1])
        xtb_dip.append(t[2])
        xtb_vip.append(t[3])
        xtb_vea.append(t[4])
        xtb_vom.append(t[5])

    features_data: Dict[str, Any] = {
        "sbu": sbu_ids,
        "class_based_atomic_irreps": final_irreps,
        "class_based_atomic_coeffs": final_coeffs,
        "max_atoms_all_sbus": int(max_atoms_all_sbus),
        "Final energy (Eh)": xtb_energies,
        "HOMO-LUMO gap (eV)": xtb_hl,
        "Dipole moment (Debye)": xtb_dip,
        "Vertical IP (eV)": xtb_vip,
        "Vertical EA (eV)": xtb_vea,
        "Global Electrophilicity Index (eV)": xtb_vom,
    }

    out_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pickle_path, "wb") as f:
        pickle.dump(features_data, f)
    return out_pickle_path


def merge_per_sbu_pickles(pickle_paths: Sequence[Path], out_path: Path) -> Path:
    """Combine many ``*_df_e3nn.pkl`` files into one bundle for :func:`assemble_class_tiled_features`."""
    sbu_l: List[str] = []
    atom_l: List[List[str]] = []
    irrep_l: List[List[Any]] = []
    coeff_l: List[List[Any]] = []
    for p in pickle_paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        sid = d.get("sbu")
        if sid is None:
            raise KeyError(f"No 'sbu' in {p}")
        sbu_l.append(str(sid))
        atom_l.append(d["atom_list"])
        irrep_l.append(d["atom_irreps"])
        coeff_l.append(d["coefficients"])
    bundle = {
        "sbu": sbu_l,
        "atom_list": atom_l,
        "atom_irreps": irrep_l,
        "coefficients": coeff_l,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    return out_path
