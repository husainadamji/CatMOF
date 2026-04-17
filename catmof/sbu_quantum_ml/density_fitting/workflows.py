"""Psi4 / density-fitting helpers. Cluster layout and xTB runs are left to the user (see ``density_fitting/examples``)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import pandas as pd

# Optional heavy imports are local to functions that need Psi4.


def symbols_from_xyz(xyz_path: Path) -> List[str]:
    with open(xyz_path, "r") as f:
        n = int(f.readline().split()[0])
        f.readline()
        return [f.readline().split()[0] for _ in range(n)]


def psi4_rohf_sto3g_wfn(
    xyz_path: Path,
    charge: int,
    spin_mult: int,
    wfn_out: Path,
    *,
    memory_gb: float = 3.0,
    num_threads: int = 4,
    maxiter: int = 1,
) -> Path:
    """Minimal ROHF/STO-3G reference (one SCF cycle by default)."""
    import psi4

    psi4.set_memory(f"{memory_gb} GB")
    psi4.set_num_threads(num_threads)

    atoms = symbols_from_xyz(Path(xyz_path))
    geom = f"{charge} {spin_mult}\n"
    with open(xyz_path, "r") as f:
        lines = f.readlines()[2 : 2 + len(atoms)]
    geom += "".join(lines)
    mol = psi4.geometry(geom)
    psi4.set_options(
        {
            "basis": "STO-3G",
            "scf_type": "pk",
            "reference": "rohf",
            "e_convergence": 1e-5,
            "d_convergence": 1e-5,
            "maxiter": maxiter,
            "fail_on_maxiter": False,
        }
    )
    _, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    wfn_out = Path(wfn_out)
    wfn_out.parent.mkdir(parents=True, exist_ok=True)
    wfn.to_file(filename=str(wfn_out))
    psi4.core.clean()
    return wfn_out


def psi4_from_xtb_molden_mos(wfn_in: Path, molden_path: Path, wfn_out: Path) -> Path:
    """Replace Ca (alpha MO coeffs) with coefficients parsed from an xTB (or compatible) Molden file."""
    from catmof.sbu_quantum_ml.density_fitting.molden_io import load_molden

    import psi4

    wfn = psi4.core.Wavefunction.from_file(str(wfn_in))
    molden_dict = load_molden(str(molden_path))
    coeff_matrix = molden_dict["orb_alpha_coeffs"]
    new_C = psi4.core.Matrix.from_array(coeff_matrix)
    wfn.Ca().copy(new_C)
    wfn_out = Path(wfn_out)
    wfn_out.parent.mkdir(parents=True, exist_ok=True)
    wfn.to_file(filename=str(wfn_out))
    return wfn_out


def run_df_e3nn(
    wfn_path: Path,
    xyz_path: Path,
    sbu_id: str,
    charge: int,
    spin_mult: int,
    out_pkl: Path,
    *,
    orbital_basis: str = "def2-universal-jfkit",
) -> Path:
    """Density-fit alpha density; dump e3nn irreps + per-atom coefficients (one SBU)."""
    from catmof.sbu_quantum_ml.density_fitting.psi4_df import DensityFitting

    import psi4

    dens = DensityFitting(str(wfn_path), str(xyz_path), orbital_basis, charge, spin_mult)
    dens.get_df_coeffs(dens.wfn.Da())
    at_irreps, coeff_features = dens.get_e3nn_features()
    symbols = symbols_from_xyz(Path(xyz_path))
    out_pkl = Path(out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "sbu": sbu_id,
                "atom_list": symbols,
                "atom_irreps": at_irreps,
                "coefficients": coeff_features,
            },
            f,
        )
    del dens
    psi4.core.clean()
    return out_pkl


def charge_spin_lookup(csv_path: Path, sbu_col: str = "sbu") -> dict:
    """Build ``{sbu_id: (charge, spin_mult)}`` from a CSV with ``charge`` and ``spin multiplicity`` columns."""
    df = pd.read_csv(csv_path)
    return {
        str(row[sbu_col]): (int(row["charge"]), int(row["spin multiplicity"]))
        for _, row in df.iterrows()
    }
