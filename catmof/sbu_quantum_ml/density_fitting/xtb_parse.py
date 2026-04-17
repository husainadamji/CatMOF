"""Parse xTB text outputs into tabular ``xtb_output_data``-style CSVs (completed runs)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _find_fileContaining(dirpath: Path, substr: str) -> Optional[str]:
    for name in os.listdir(dirpath):
        if substr in name and os.path.isfile(dirpath / name):
            return name
    return None


def parse_finished_xtb_case(case_dir: Path) -> dict:
    """Extract scalar fields from one SBU directory of xTB logs (spinpol, SP, VIP/EA, ω)."""
    case_dir = Path(case_dir)
    sbu = case_dir.name
    out: dict = {"sbu": sbu}

    e_file = _find_fileContaining(case_dir, "spinpol_Ecorr")
    sp_file = _find_fileContaining(case_dir, "_SP")
    vipea_file = _find_fileContaining(case_dir, "_vipea")
    vomega_file = _find_fileContaining(case_dir, "_vomega")

    energy = hl_gap = dipole = vip = vea = vomega = None

    if e_file:
        lines = (case_dir / e_file).read_text().splitlines()
        for line in reversed(lines):
            if "TOTAL ENERGY" in line and len(line.split()) == 6:
                energy = float(line.split()[3])
                break

    if sp_file:
        lines = (case_dir / sp_file).read_text().splitlines()
        for line in reversed(lines):
            if "HOMO-LUMO GAP" in line and len(line.split()) == 6:
                hl_gap = float(line.split()[3])
                break
        for i, line in enumerate(lines):
            if "molecular dipole:" in line and i + 3 < len(lines):
                dipole_line = lines[i + 3]
                if len(dipole_line.split()) == 5:
                    dipole = dipole_line.split()[-1]
                break

    if vipea_file:
        lines = (case_dir / vipea_file).read_text().splitlines()
        for line in lines:
            if "delta SCC IP (eV):" in line and len(line.split()) == 5:
                vip = line.split()[4]
            if "delta SCC EA (eV):" in line and len(line.split()) == 5:
                vea = line.split()[4]

    if vomega_file:
        for line in (case_dir / vomega_file).read_text().splitlines():
            if "Global electrophilicity index (eV):" in line and len(line.split()) == 5:
                vomega = line.split()[4]
                break

    out["Final energy (Eh)"] = energy
    out["HOMO-LUMO gap (eV)"] = hl_gap
    out["Dipole moment (Debye)"] = dipole
    out["Vertical IP (eV)"] = vip
    out["Vertical EA (eV)"] = vea
    out["Global Electrophilicity Index (eV)"] = vomega
    return out


def parse_xtb_root_to_dataframe(xtb_root: Path) -> pd.DataFrame:
    """Parse every immediate subdirectory under ``xtb_root`` (assumes jobs are finished)."""
    xtb_root = Path(xtb_root)
    rows: List[dict] = []
    for name in sorted(os.listdir(xtb_root)):
        p = xtb_root / name
        if not p.is_dir():
            continue
        rows.append(parse_finished_xtb_case(p))
    return pd.DataFrame(rows)
