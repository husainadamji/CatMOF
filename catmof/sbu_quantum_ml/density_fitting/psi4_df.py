"""Psi4 wavefunctions and atom-centered density fitting (JKFIT auxiliary basis)."""

from __future__ import annotations

import os
from collections import Counter
from typing import List, Tuple

import numpy as np
import psi4
import torch
from e3nn import o3

# Adapted from https://github.com/hjkgrp/dfa_recommender.git


def get_molecule(xyzfile: str, charge: int, spin: int, sym: str = "c1") -> Tuple[psi4.core.Molecule, list]:
    """Build a Psi4 molecule from a standard XYZ file (two-line header)."""
    wholetext = "%s %s\n" % (charge, spin)
    symbols: List[str] = []
    if not os.path.isfile(xyzfile):
        raise FileNotFoundError(xyzfile)
    with open(xyzfile, "r") as fo:
        natoms = int(fo.readline().split()[0])
        fo.readline()
        for _ in range(natoms):
            line = fo.readline()
            wholetext += line
            symbols.append(line.split()[0])
    wholetext += "\nsymmetry %s\nnoreorient\nnocom\n" % sym
    mol = psi4.geometry("""%s""" % wholetext)
    return mol, symbols


class DensityFitting:
    """Project density (or Fock/Hamiltonian) onto a JKFIT auxiliary basis."""

    def __init__(
        self,
        wfnpath: str,
        xyzfile: str,
        basis: str,
        charge: int = 0,
        spin: int = 1,
        wfnpath2: str = "NA",
    ) -> None:
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.wfnpath = wfnpath
        self.wfnpath2 = wfnpath2
        self.xyzfile = xyzfile
        self.construct_aux()
        self.get_dab()
        self.calc_utilities()

    def __str__(self) -> str:
        return f"wfnpath: {self.wfnpath}\nxyzfile: {self.xyzfile}\nbasis: {self.basis}"

    @property
    def wfnpath(self) -> str:
        return self._wfnpath

    @wfnpath.setter
    def wfnpath(self, wfnpath: str) -> None:
        self._wfnpath = wfnpath
        self.wfn = psi4.core.Wavefunction.from_file(self._wfnpath)
        assert isinstance(self.wfn, psi4.core.Wavefunction)
        self.orb = self.wfn.basisset()

    @property
    def wfnpath2(self) -> str:
        return self._wfnpath2

    @wfnpath2.setter
    def wfnpath2(self, wfnpath2: str) -> None:
        self._wfnpath2 = wfnpath2
        if wfnpath2 == "NA":
            return
        if os.path.isfile(wfnpath2):
            self.wfn2 = psi4.core.Wavefunction.from_file(wfnpath2)
            assert isinstance(self.wfn2, psi4.core.Wavefunction)
        else:
            raise FileNotFoundError(wfnpath2)

    @property
    def xyzfile(self) -> str:
        return self._xyzfile

    @xyzfile.setter
    def xyzfile(self, xyzfile: str) -> None:
        self._xyzfile = xyzfile
        self.mol, self.symbols = get_molecule(self._xyzfile, self.charge, self.spin)
        assert isinstance(self.mol, psi4.core.Molecule)

    def construct_aux(self) -> None:
        self.aux = psi4.core.BasisSet.build(self.mol, "DF_BASIS_SCF", "", "JKFIT", self.basis)

    def get_dab(self) -> None:
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
        mints = psi4.core.MintsHelper(self.orb)
        abQ = mints.ao_eri(self.aux, zero_bas, self.orb, self.orb)
        Jinv = mints.ao_eri(self.aux, zero_bas, self.aux, zero_bas)
        Jinv.power(-1.0, 1.0e-14)
        abQ = np.squeeze(abQ)
        self.Jinv = np.squeeze(Jinv)
        self.dab_P = np.einsum("Qab,QP->abP", abQ, self.Jinv, optimize=True)

    def calc_utilities(self) -> None:
        self.numfuncatom = np.zeros(self.mol.natom())
        shells: List[int] = []
        for func in range(0, self.aux.nbf()):
            current = self.aux.function_to_center(func)
            shell = self.aux.function_to_shell(func)
            shells.append(shell)
            self.numfuncatom[current] += 1

        self.shellmap: List[List[int]] = []
        ii = 0
        tmp: List[int] = []
        tmp_count = 0
        for shell in range(self.aux.nshell()):
            count = shells.count(shell)
            tmp_count += count
            tmp.append((count - 1) // 2)
            if tmp_count == self.numfuncatom[ii]:
                ii += 1
                self.shellmap.append(tmp)
                tmp = []
                tmp_count = 0

    def get_df_coeffs(self, D: psi4.core.Matrix) -> None:
        self.C_P = np.einsum("abP,ab->P", self.dab_P, D, optimize=True)

    def compensate_charges(self) -> None:
        q: List[float] = []
        counter = 0
        shellmap = np.concatenate(self.shellmap)
        for i in range(self.mol.natom()):
            for j in range(counter, counter + int(self.numfuncatom[i])):
                shell_num = self.aux.function_to_shell(j)
                shell = self.aux.shell(shell_num)
                normalization = shell.coef(0)
                exponent = shell.exp(0)
                if shellmap[shell_num] == 0:
                    integral = (1 / (4 * exponent)) * np.sqrt(np.pi / exponent)
                    q.append(4 * np.pi * normalization * integral)
                else:
                    q.append(0.0)
                counter += 1
        q_arr = np.array(q) * 0.5
        bigQ = self.wfn.nalpha()
        numer = bigQ - np.dot(q_arr, self.C_P * 2)
        denom = np.dot(np.dot(q_arr, self.Jinv), q_arr)
        lambchop = numer / denom
        self.C_P += np.dot(self.Jinv, lambchop * q_arr) * 0.5

    def calc_powerspec(self) -> np.ndarray:
        shells: List[List[int]] = []
        shells_to_at: List[int] = []
        preshell = -1
        currshell: List[int] = []
        for i_bf in range(self.aux.nbf()):
            f2s = self.aux.function_to_shell(i_bf)
            f2c = int(self.aux.function_to_center(i_bf))
            if f2s == preshell:
                currshell.append(i_bf)
            else:
                shells.append(currshell)
                currshell = [i_bf]
                preshell = f2s
                shells_to_at.append(f2c)
        shells.append(currshell)

        powerspec: List[List[float]] = []
        preat = -1
        currat: List[float] = []
        for i_s, shell in enumerate(shells[1:]):
            p = float(np.sum(self.C_P[shell] ** 2))
            if shells_to_at[i_s] == preat:
                currat.append(p)
            else:
                powerspec.append(currat)
                currat = [p]
                preat = shells_to_at[i_s]
        powerspec.append(currat)
        return np.array(powerspec[1:])

    def get_e3nn_features(self) -> Tuple[List[o3.Irreps], List[np.ndarray]]:
        psi4_2_e3nn = [
            [0],
            [2, 0, 1],
            [4, 2, 0, 1, 3],
            [6, 4, 2, 0, 1, 3, 5],
            [8, 6, 4, 2, 0, 1, 3, 5, 7],
            [10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9],
            [12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11],
        ]

        shells: List[List[int]] = []
        shells_to_at: List[int] = []
        preshell = -1
        currshell: List[int] = []
        for i_bf in range(self.aux.nbf()):
            f2s = self.aux.function_to_shell(i_bf)
            f2c = int(self.aux.function_to_center(i_bf))
            if f2s == preshell:
                currshell.append(i_bf)
            else:
                shells.append(currshell)
                currshell = [i_bf]
                preshell = f2s
                shells_to_at.append(f2c)
        shells.append(currshell)

        coeff_features: List[List[np.ndarray]] = []
        at_binned_shells: List[List[List[int]]] = []
        currat_features: List[np.ndarray] = []
        currat_shells: List[List[int]] = []
        preat = -1
        for i_s, shell in enumerate(shells[1:]):
            shell_coeffs = self.C_P[shell]
            l = (len(shell) - 1) // 2
            reordered = shell_coeffs[psi4_2_e3nn[l]]
            if shells_to_at[i_s] == preat:
                currat_shells.append(shell)
                currat_features.append(reordered)
            else:
                coeff_features.append(currat_features)
                at_binned_shells.append(currat_shells)
                currat_features = [reordered]
                currat_shells = [shell]
                preat = shells_to_at[i_s]
        coeff_features.append(currat_features)
        at_binned_shells.append(currat_shells)
        coeff_features = coeff_features[1:]
        coeff_features = [np.concatenate(x) for x in coeff_features]
        at_binned_shells = at_binned_shells[1:]

        at_irreps: List[o3.Irreps] = []
        for at_shells in at_binned_shells:
            l_counts: Counter[int] = Counter()
            for sh in at_shells:
                ll = (len(sh) - 1) // 2
                l_counts[ll] += 1
            irreps = [f"{count}x{l}{'e' if l % 2 == 0 else 'o'}" for l, count in sorted(l_counts.items())]
            at_irreps.append(o3.Irreps("+".join(irreps)))

        return at_irreps, coeff_features

    def pad_df_coeffs(self) -> None:
        self.max_shell = np.array(self.shellmap[int(np.argmax(self.numfuncatom))])
        self.n_coeff_atom = int(np.max(self.numfuncatom))
        self.max_tmplate = [0]
        self.irreps: o3.Irreps
        irreps_parts: List[str] = []
        for ii in range(int(np.max(self.max_shell)) + 1):
            self.max_tmplate.append(int((self.max_shell == ii).sum() * (2 * ii + 1) + self.max_tmplate[-1]))
            evenodd = "e" if ii % 2 == 0 else "o"
            irreps_parts += [str(int((self.max_shell == ii).sum())) + "x" + str(ii) + evenodd]
        self.irreps = o3.Irreps("+".join(irreps_parts))

        tmplates: List[List[int]] = []
        for jj in range(len(self.shellmap)):
            tmplate = [0]
            _shell = np.array(self.shellmap[jj])
            for ii in range(int(np.max(_shell))):
                tmplate.append(int((_shell == ii).sum() * (2 * ii + 1) + tmplate[-1]))
            tmplate.append(int(self.numfuncatom[jj]))
            tmplates.append(tmplate)

        self.C_P_pad = np.zeros(shape=(self.numfuncatom.shape[0], self.n_coeff_atom))
        count = 0
        for ii, tmplate in enumerate(tmplates):
            for jj in range(int(np.max(self.shellmap[ii])) + 1):
                _end = min(
                    self.max_tmplate[jj] + tmplate[jj + 1] - tmplate[jj],
                    self.max_tmplate[jj + 1],
                )
                self.C_P_pad[ii, self.max_tmplate[jj] : _end] = self.C_P[
                    (count + tmplate[jj]) : (count + tmplate[jj + 1])
                ]
            count += int(self.numfuncatom[ii])

    def convert_CP2e3nn(self) -> None:
        psi4_2_e3nn = [
            [0],
            [2, 0, 1],
            [4, 2, 0, 1, 3],
            [6, 4, 2, 0, 1, 3, 5],
            [8, 6, 4, 2, 0, 1, 3, 5, 7],
            [10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9],
            [12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11],
        ]
        self.C_P_pad_e3nn = np.zeros(shape=self.C_P_pad.shape)
        for ii in range(self.C_P_pad.shape[0]):
            for jj, irrep in enumerate(self.irreps):
                num, l = irrep.mul, irrep.ir.l
                coeffs = self.C_P_pad[ii][self.max_tmplate[jj] : self.max_tmplate[jj + 1]]
                coeffs_split = np.array_split(coeffs, num)
                coeffs_trans: List[float] = []
                for coeff in coeffs_split:
                    for k in psi4_2_e3nn[l]:
                        coeffs_trans.append(coeff[k])
                self.C_P_pad_e3nn[ii][self.max_tmplate[jj] : self.max_tmplate[jj + 1]] = coeffs_trans
        mat = np.where(np.abs(self.C_P_pad_e3nn) < 1e-8)
        self.C_P_pad_e3nn[mat[0], mat[1]] = 0

        change_of_basis = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        D_change = self.irreps.D_from_matrix(change_of_basis).numpy()
        self.C_P_pad_e3nn = self.C_P_pad_e3nn @ D_change.T


def get_spectra(
    densfit: DensityFitting,
    fock: bool = False,
    H: bool = False,
    t: str = "alpha",
) -> np.ndarray:
    if not H:
        if not fock:
            D = densfit.wfn.Da() if t == "alpha" else densfit.wfn.Db()
        else:
            D = densfit.wfn.Fa() if t == "alpha" else densfit.wfn.Fb()
    else:
        D = densfit.wfn.H()
    densfit.get_df_coeffs(D)
    return densfit.calc_powerspec()


def get_subtracted_spectra(densfit: DensityFitting, fock: bool = False, t: str = "alpha") -> np.ndarray:
    if not fock:
        D1 = densfit.wfn.Da() if t == "alpha" else densfit.wfn.Db()
        D2 = densfit.wfn2.Da() if t == "alpha" else densfit.wfn2.Db()
    else:
        D1 = densfit.wfn.Fa() if t == "alpha" else densfit.wfn.Fb()
        D2 = densfit.wfn2.Fa() if t == "alpha" else densfit.wfn2.Fb()
    delta_D = D1.clone()
    delta_D = delta_D.from_array(D1.to_array() - D2.to_array())
    densfit.get_df_coeffs(delta_D)
    return densfit.calc_powerspec()
