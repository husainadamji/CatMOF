"""Parse Molden format (geometry, GTO, MO blocks) for aligning MOs with Psi4 wavefunctions."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from molSimplify.Classes.globalvars import amassdict

angstrom2bohr = 1.8897261258369282

_shell_labels = ["s", "p", "d", "f", "sp", "g"]


def load_molden(filename: str) -> Dict[str, Any]:
    """Parse a Molden file into a dict with coordinates, obasis, and MO coefficients.

    Specification: https://www.theochem.ru.nl/molden/molden_format.html
    """
    with open(filename, "r") as fin:
        lines = fin.readlines()

    sections: List[List[str]] = []
    section: List[str] = []
    for line in lines:
        if line.startswith("["):
            if section:
                sections.append(section)
            section = []
        section.append(line.rstrip("\n"))
    sections.append(section)

    results: Dict[str, Any] = {}
    obasis: Dict[str, Any] = {}

    for section in sections:
        name = section[0].lower()
        if name.startswith("[title]"):
            results["title"] = "\n".join(section[1:])
        elif name.startswith("[atoms]"):
            length_unit = angstrom2bohr if "angs" in name else 1.0
            coordinates, numbers, pseudo_numbers = _read_geometry_section(section[1:])
            results["coordinates"] = coordinates * length_unit
            results["numbers"] = numbers
            results["pseudo_numbers"] = pseudo_numbers
        elif name.startswith("[gto]"):
            obasis = _read_gto_section(section[1:])
        elif name.startswith("[mo]"):
            results.update(_read_mo_section(section[1:]))

    if "coordinates" not in results:
        raise ValueError(f"No [Atoms] block found in {filename!r}")

    obasis["centers"] = results["coordinates"]
    results["obasis"] = obasis
    results["permutation"] = None

    return results


def _read_geometry_section(lines: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates: List[List[float]] = []
    numbers: List[int] = []
    pseudo_numbers: List[int] = []

    for line in lines:
        sp = line.split()
        if len(sp) != 6:
            break
        numbers.append(int(amassdict[sp[0]][1]))
        pseudo_numbers.append(int(sp[2]))
        coordinates.append([float(x) for x in sp[3:6]])
    return np.array(coordinates), np.array(numbers), np.array(pseudo_numbers)


def _read_gto_section(lines: List[str]) -> Dict[str, Any]:
    obasis: Dict[str, Any] = {}
    shell_map: List[int] = []
    nprims: List[int] = []
    shell_types: List[int] = []
    alphas: List[float] = []
    con_coeffs: List[float] = []

    line_iter = iter(lines)
    center = -1
    for line in line_iter:
        sp = line.split()
        if len(sp) == 0:
            continue
        if len(sp) == 2 and sp[1] == "0":
            center = int(sp[0]) - 1
        elif sp[0] in _shell_labels:
            shell_types.append(_shell_labels.index(sp[0]))
            nprims.append(int(sp[1]))
            shell_map.append(center)
            for _ in range(nprims[-1]):
                line2 = next(line_iter)
                alpha, coeff = line2.split()
                alphas.append(float(alpha))
                con_coeffs.append(float(coeff))

    obasis["shell_map"] = shell_map
    obasis["nprims"] = nprims
    obasis["shell_types"] = shell_types
    obasis["alphas"] = alphas
    obasis["con_coeffs"] = con_coeffs
    return obasis


def _read_mo_section(lines: List[str]) -> Dict[str, Any]:
    orb_alpha_energies: List[float] = []
    orb_beta_energies: List[float] = []
    orb_alpha_occs: List[float] = []
    orb_beta_occs: List[float] = []
    orb_alpha_coeffs: List[List[float]] = []
    orb_beta_coeffs: List[List[float]] = []

    current_spin = "none"
    current_energy = 0.0
    current_occ = 0.0
    current_coeffs: List[float] = []
    for line in lines:
        sp = line.split()
        if len(sp) == 0:
            continue
        if "=" in line:
            if current_coeffs:
                if current_spin == "alpha":
                    orb_alpha_energies.append(current_energy)
                    orb_alpha_occs.append(current_occ)
                    orb_alpha_coeffs.append(current_coeffs)
                elif current_spin == "beta":
                    orb_beta_energies.append(current_energy)
                    orb_beta_occs.append(current_occ)
                    orb_beta_coeffs.append(current_coeffs)
            if sp[0].lower() == "ene=":
                current_energy = float(sp[1])
            elif sp[0].lower() == "spin=":
                current_spin = sp[1].lower()
            elif sp[0].lower() == "occup=":
                current_occ = float(sp[1])
            current_coeffs = []
        else:
            current_coeffs.append(float(sp[1]))

    if current_spin == "alpha":
        orb_alpha_energies.append(current_energy)
        orb_alpha_occs.append(current_occ)
        orb_alpha_coeffs.append(current_coeffs)
    else:
        orb_beta_energies.append(current_energy)
        orb_beta_occs.append(current_occ)
        orb_beta_coeffs.append(current_coeffs)

    results: Dict[str, Any] = {
        "orb_alpha": np.shape(orb_alpha_coeffs),
        "orb_alpha_energies": np.array(orb_alpha_energies),
        "orb_alpha_occs": np.array(orb_alpha_occs),
        "orb_alpha_coeffs": np.array(orb_alpha_coeffs).T,
    }
    if orb_beta_energies:
        results["orb_beta"] = np.shape(orb_beta_coeffs)
        results["orb_beta_energies"] = np.array(orb_beta_energies)
        results["orb_beta_occs"] = np.array(orb_beta_occs)
        results["orb_beta_coeffs"] = np.array(orb_beta_coeffs).T
    else:
        results["orb_alpha_occs"] /= 2.0
    return results
