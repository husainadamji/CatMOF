import os
import numpy as np
import sbu_analysis as sbu_funs
from catmof.paths import get_paths
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.atom3D import atom3D
import molSimplify.Scripts.geometry as geom_funs
import networkx as nx
import pandas as pd
from data.atomic_data import BL_BA
import copy

def replace_wrong_H(temp_mol, graph):

    # Check for H atoms in rings:
    G = nx.from_numpy_matrix(graph)
    cycles = nx.minimum_cycle_basis(G)
    no_metal_cycles = [cycle for cycle in cycles if not any(metal in cycle for metal in temp_mol.findMetal(transition_metals_only=False))]
    for cycle in no_metal_cycles:
        for idx in cycle:
            if temp_mol.getAtom(idx).symbol() == "H":
                temp_mol.getAtom(idx).mutate(newType="C")

    return


def get_no_metal_cycles(temp_mol, graph, bond_threshold=2.1):

    G = nx.from_numpy_matrix(graph)
    cycles = nx.minimum_cycle_basis(G)
    metals = temp_mol.findMetal(transition_metals_only=False)
    no_metal_cycles = [cycle for cycle in cycles if not any(metal in cycle for metal in metals)]

    atom_bonds_severed = set()
    for cycle in no_metal_cycles:
        for at in cycle:
            bound_atoms_dists = [
                (x, temp_mol.getAtom(at).distance(temp_mol.getAtom(x))) 
                for x in temp_mol.getBondedAtomsSmart(at, oct=False)
                ]
            bond_to_break = [
                x for x, dist in bound_atoms_dists
                if dist > bond_threshold and x not in temp_mol.findMetal(transition_metals_only=False)
            ]
            if bond_to_break:
                for bound_atom in bond_to_break:
                    graph[at, bound_atom] = graph[bound_atom, at] = 0
                    atom_bonds_severed.add(at)
                    atom_bonds_severed.add(bound_atom)

    G = nx.from_numpy_matrix(graph)
    cycles = nx.minimum_cycle_basis(G)
    final_cycles = [cycle for cycle in cycles if not any(metal in cycle for metal in metals)]

    temp_mol.graph = graph
    
    return final_cycles, graph, atom_bonds_severed


def check_planarity_bound_atoms(temp_mol, central_atom, bound_atoms, threshold=1.05e-1):

    # Functionality only for 3 bound atoms for now
    if len(bound_atoms) != 3:
        return False

    central_atom_coords = temp_mol.getAtomCoords(central_atom)
    bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
    
    u = np.subtract(bound_coords[0], central_atom_coords)
    v = np.subtract(bound_coords[1], central_atom_coords)
    n = np.array(geom_funs.normalize(-np.add(u, v)))
    w = np.array(geom_funs.normalize(np.subtract(bound_coords[2], central_atom_coords)))

    vec_similarity = np.dot(n, w)

    return np.abs(vec_similarity - 1) < threshold


def check_linearity_bound_atoms(temp_mol, central_atom, bound_atoms, threshold=1.05e-1):
    
    # Functionality only for 2 bound atoms for now
    if len(bound_atoms) != 2:
        return False

    central_atom_coords = temp_mol.getAtomCoords(central_atom)
    bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]

    u = np.array(geom_funs.normalize(np.subtract(bound_coords[0], central_atom_coords)))
    v = -np.array(geom_funs.normalize(np.subtract(bound_coords[1], central_atom_coords)))

    vec_similarity = np.dot(u, v)

    return np.abs(vec_similarity - 1) < threshold


def check_path_length(graph, at1, at2_list, desired_path_length):

    G = nx.from_numpy_matrix(graph)
    shortest_path_length = np.inf
    matching_paths_count = 0
    for at2 in at2_list:
        if nx.has_path(G, at1, at2):
            path_length = nx.shortest_path_length(G, source=at1, target=at2)
            if path_length < shortest_path_length:
                shortest_path_length = path_length
                matching_paths_count = 0 # Reset the count if a shorter path is found
            if path_length == desired_path_length:
                matching_paths_count += 1
    # Return tuple: (True if a path exists with the desired length, count of such paths)
    return (shortest_path_length == desired_path_length, matching_paths_count)


def break_periodic_bonds(temp_mol, graph, ref_at):

    bound_atoms = temp_mol.getBondedAtomsSmart(ref_at, oct=False)
    bound_dists = [temp_mol.getAtom(x).distance(temp_mol.getAtom(ref_at)) for x in bound_atoms]
    distant_atoms = [bound_atoms[i] for i, dist in enumerate(bound_dists) if dist > 4]
    if distant_atoms:
        for at in distant_atoms:
            graph[ref_at, at] = graph[at, ref_at] = 0

    temp_mol.graph = graph
    bound_atoms = temp_mol.getBondedAtomsSmart(ref_at, oct=False)

    return graph, bound_atoms


def get_C_for_capping(temp_mol, graph, netnegcharge):

    cycles, graph, atom_bonds_severed = get_no_metal_cycles(temp_mol, graph)
    atoms_in_cycles = set([at for subcycle in cycles for at in subcycle])
    C_atoms_in_cycles = [at for at in atoms_in_cycles if temp_mol.getAtom(at).symbol() == "C"]

    keys = ["methyl", "acetate", "H"]
    C_indices = {key: ([], []) for key in keys}
    for C_atom in temp_mol.findAtomsbySymbol("C"):
        graph, bound_atoms = break_periodic_bonds(temp_mol, graph, C_atom)
        if (len(bound_atoms) == 2 and
            all(temp_mol.getAtom(x).symbol() == "O" for x in bound_atoms)): # methyl addition
            netnegcharge += 1
            C_indices["methyl"][0].append(C_atom)
        elif len(bound_atoms) == 2 and C_atom in C_atoms_in_cycles: # H addition to ring carbons
            C_indices["H"][0].append(C_atom)
            C_indices["H"][1].append(1)
        elif (len(bound_atoms) <= 2 and
            sum(temp_mol.getAtom(x).symbol() == "O" for x in bound_atoms) == 1): # acetate addition
            if "N" not in [temp_mol.getAtom(x).symbol() for x in bound_atoms]:
                C_indices["acetate"][0].append(C_atom)
                O_atom = next(x for x in bound_atoms if temp_mol.getAtom(x).symbol() == "O")
                if all(temp_mol.getAtom(x).symbol() != "H" for x in temp_mol.getBondedAtomsSmart(O_atom, oct=False)):
                    netnegcharge += 1
            else: # bound DMF or other N-containing ligands coordinating to carbonyl
                C_indices["H"][0].append(C_atom)
                C_indices["H"][1].append(1)
        elif (len(bound_atoms) == 2 and
              all(x not in temp_mol.findMetal(transition_metals_only=False) for x in bound_atoms)): # H addition to sp3 carbon deficient by 2 H's
            C_indices["H"][0].append(C_atom)
            C_indices["H"][1].append(2)
        elif (len(bound_atoms) == 2 and
              check_linearity_bound_atoms(temp_mol, C_atom, bound_atoms, threshold=1.5e-1) and
              any(x in temp_mol.findMetal(transition_metals_only=False) for x in bound_atoms) and
              any(temp_mol.getAtom(x).symbol() == "N" for x in bound_atoms)): # Update charge for C-coordinating metal-bound cyanide
            netnegcharge += 1
        elif (len(bound_atoms) == 1 and
              temp_mol.getAtom(bound_atoms[0]).symbol() == "C" and
              len(temp_mol.getBondedAtomsSmart(bound_atoms[0], oct=False)) == 2 and
              len(set(temp_mol.getBondedAtomsSmart(bound_atoms[0], oct=False)).intersection(temp_mol.findMetal(transition_metals_only=False))) == 1): # H addition to terminal alkyne bound to metal
            C_indices["H"][0].append(C_atom)
            C_indices["H"][1].append(1)
            netnegcharge += 1
        elif (len(bound_atoms) == 1 and
              not check_linearity_bound_atoms(temp_mol, bound_atoms[0], temp_mol.getBondedAtomsSmart(bound_atoms[0], oct=False))): # H addition to sp3 carbon deficient by 3 H
            C_indices["H"][0].append(C_atom)
            C_indices["H"][1].append(3)
            if check_path_length(graph, C_atom, temp_mol.findAtomsbySymbol("O"), 2) == (True, 2): # Update charge for terminal C's that are part of acetates
                netnegcharge += 1
        elif (len(bound_atoms) == 1 and
              temp_mol.getAtom(bound_atoms[0]).symbol() == "N" and
              check_linearity_bound_atoms(temp_mol, bound_atoms[0], temp_mol.getBondedAtomsSmart(bound_atoms[0], oct=False), threshold=1.5e-1)): # Update charge for N-coordinating metal-bound cyanide
              netnegcharge += 1
        elif (len(bound_atoms) == 3 and
               C_atom not in C_atoms_in_cycles and
               "O" not in [temp_mol.getAtom(x).symbol() for x in bound_atoms] and
                not check_planarity_bound_atoms(temp_mol, C_atom, bound_atoms)): # H addition to sp3 carbon deficient by 1 H
            C_indices["H"][0].append(C_atom)
            C_indices["H"][1].append(1)
        elif (len(bound_atoms) == 3 and
              sum(temp_mol.getAtom(x).symbol() == "O" for x in bound_atoms) == 2): # Update charge if full carboxylate already exists
            netnegcharge += 1

    return C_indices, graph, netnegcharge


def get_O_for_capping(temp_mol, graph, netnegcharge):

    O_atom_indices = []
    for O_atom in temp_mol.findAtomsbySymbol("O"):
        graph, bound_atoms = break_periodic_bonds(temp_mol, graph, O_atom)
        bound_metals = [x for x in bound_atoms if temp_mol.getAtom(x).ismetal(transition_metals_only=False)]
        if len(bound_metals) == len(bound_atoms): # Bridging O's exist in many SBUs
            # Remove artificial M-O connectivity arising from periodic boundary conditions
            MO_BLs = [temp_mol.getAtom(O_atom).distance(temp_mol.getAtom(x)) for x in bound_metals]
            if max(MO_BLs) - min(MO_BLs) > 0.11: # Min/Max approach for bridging O's
                artificial_MO_metal = bound_metals[MO_BLs.index(max(MO_BLs))]
                graph[O_atom, artificial_MO_metal] = graph[artificial_MO_metal, O_atom] = 0
                temp_mol.graph = graph
            # Add O atom to O_atom_indices if it is terminally bound to metal
            if len(temp_mol.getBondedAtomsSmart(O_atom, oct=False)) == 1:
                O_atom_indices.append(O_atom)
            else:
                netnegcharge += 2
        else:
            count_H = sum(1 for x in bound_atoms if temp_mol.getAtom(x).symbol() == "H")
            count_metals = sum(1 for x in bound_atoms if temp_mol.getAtom(x).ismetal(transition_metals_only=False))
            if count_H == 1 and len(bound_atoms) - 1 == count_metals:
                netnegcharge += 1

    return O_atom_indices, graph, netnegcharge


def get_N_for_capping(temp_mol, graph, netnegcharge):

    cycles, graph, atom_bonds_severed = get_no_metal_cycles(temp_mol, graph)
    atoms_in_cycles = set([at for subcycle in cycles for at in subcycle])
    N_atoms_in_cycles = [at for at in atoms_in_cycles if temp_mol.getAtom(at).symbol() == "N"]

    N_atom_indices = [] # List of tupes: (N_atom, num_H_to_add)
    for N_atom in temp_mol.findAtomsbySymbol("N"):
        graph, bound_atoms = break_periodic_bonds(temp_mol, graph, N_atom)
        if (len(bound_atoms) == 1 and
            # N_atom not in N_atoms_in_cycles and
            not check_linearity_bound_atoms(temp_mol, bound_atoms[0], temp_mol.getBondedAtomsSmart(bound_atoms[0]), threshold=1.5e-1)):
            if bound_atoms[0] in temp_mol.findMetal(transition_metals_only=False): # terminal N bound to metal converted to ammonia
                N_atom_indices.append((N_atom, 3))
            else: # non-cyanide and nonmetal bound terminal N
                N_atom_indices.append((N_atom, 2))
                if (temp_mol.getAtom(bound_atoms[0]).symbol() == "O" and
                    check_path_length(graph, N_atom, temp_mol.findMetal(transition_metals_only=False), 2) == (True, 2)): # Update charge for N-O-M type connections
                    netnegcharge += 1
        elif (len(bound_atoms) == 2 and
            N_atom not in N_atoms_in_cycles and
            not check_linearity_bound_atoms(temp_mol, N_atom, bound_atoms)):
            if any(metal in bound_atoms for metal in temp_mol.findMetal(transition_metals_only=False)): # keep N bound to metal neutral
                N_atom_indices.append((N_atom, 2))
            else: # non-cyanide and nonmetal bound N
                N_atom_indices.append((N_atom, 1))
                if (sum(temp_mol.getAtom(x).symbol() == "O" for x in bound_atoms) == 1 and
                    check_path_length(graph, N_atom, temp_mol.findMetal(transition_metals_only=False), 2) == (True, 2)): # Update charge for N-O-M type connections
                    netnegcharge += 1


    return N_atom_indices, graph, netnegcharge


def add_atom(temp_mol, graph, new_symbol, new_coords, bond_length, ref_atom_idx):

    new_atom = atom3D(new_symbol, new_coords)
    temp_mol.addAtom(new_atom)
    graph = np.c_[graph, np.zeros(np.shape(graph)[0])]
    graph = np.vstack([graph, np.zeros(np.shape(graph)[1])])
    new_atom_idx = temp_mol.natoms - 1
    graph[new_atom_idx, ref_atom_idx] = 1
    graph[ref_atom_idx, new_atom_idx] = 1
    temp_mol.graph = graph
    temp_mol.BCM(new_atom_idx, ref_atom_idx, bond_length)
    
    return graph, new_atom_idx


def add_methyl(temp_mol, graph, C_idx, formate_flag=False):

    C_coords = temp_mol.getAtomCoords(C_idx)
    O_idxs = temp_mol.getBondedAtomsSmart(C_idx, oct=False)
    O_coords = [temp_mol.getAtomCoords(x) for x in O_idxs]

    u, v = np.subtract(O_coords[0], C_coords), np.subtract(O_coords[1], C_coords)
    n = np.array(geom_funs.normalize(-np.add(u, v)))

    new_C_coords = np.add(n * BL_BA["CC"], C_coords)
    if formate_flag:
        graph, _ = add_atom(temp_mol, graph, "H", new_C_coords, BL_BA["CH"], C_idx)

        return graph
    
    graph, new_C_idx = add_atom(temp_mol, graph, "C", new_C_coords, BL_BA["CC"], C_idx)

    H1_coords = np.add(n * BL_BA["CH"], new_C_coords)
    graph, H1_idx = add_atom(temp_mol, graph, "H", H1_coords, BL_BA["CH"], new_C_idx)
    temp_mol.ACM(H1_idx, new_C_idx, C_idx, BL_BA["HCC"])
    H1_coords = temp_mol.getAtomCoords(H1_idx)
    H2_coords = geom_funs.PointRotateAxis(n, new_C_coords, H1_coords, 2.0944)
    graph, _ = add_atom(temp_mol, graph, "H", H2_coords, BL_BA["CH"], new_C_idx)
    H3_coords = geom_funs.PointRotateAxis(n, new_C_coords, H2_coords, 2.0944)
    graph, _ = add_atom(temp_mol, graph, "H", H3_coords, BL_BA["CH"], new_C_idx)

    return graph


def add_acetate(temp_mol, graph, C_idx, formate_flag=False):

    C_coords = temp_mol.getAtomCoords(C_idx)
    bound_atoms = temp_mol.getBondedAtomsSmart(C_idx, oct=False)
    bound_O_idx = next(x for x in bound_atoms if temp_mol.getAtom(x).symbol() == "O")
    bound_O_coords = temp_mol.getAtomCoords(bound_O_idx)
    
    if len(bound_atoms) == 2:
        bound_non_O_idx = next(x for x in bound_atoms if temp_mol.getAtom(x).symbol() != "O")
        bound_non_O_coords = temp_mol.getAtomCoords(bound_non_O_idx)
        u, v = np.subtract(bound_O_coords, C_coords), np.subtract(bound_non_O_coords, C_coords)
        n = np.array(geom_funs.normalize(-np.add(u, v)))
        new_O_coords = np.add(n * BL_BA["CO"], C_coords)
        graph, _ = add_atom(temp_mol, graph, "O", new_O_coords, BL_BA["CO"], C_idx)

        return graph
    else:
        n = np.array(geom_funs.normalize(np.subtract(C_coords, bound_O_coords)))
        new_O_coords = np.add(n * BL_BA["CO"], C_coords)
        graph, new_O_idx = add_atom(temp_mol, graph, "O", new_O_coords, BL_BA["CO"], C_idx)
        temp_mol.ACM(new_O_idx, C_idx, bound_O_idx, BL_BA["OCO"])

        graph = add_methyl(temp_mol, graph, C_idx, formate_flag=formate_flag)

        return graph


def cap_terminal_carbons(temp_mol, graph, C_idx, num_H_to_add):

    C_coords = temp_mol.getAtomCoords(C_idx)
    bound_atoms = temp_mol.getBondedAtomsSmart(C_idx, oct=False)
    if num_H_to_add == 1:
        if len(bound_atoms) == 1: # Accounting for H-addition to a terminal alkyne
            bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
            n = np.array(geom_funs.normalize(np.subtract(C_coords, bound_coords[0])))
            H_coords = np.add(C_coords, n * BL_BA["CH"])
            graph, _ = add_atom(temp_mol, graph, "H", H_coords, BL_BA["CH"], C_idx)
        elif len(bound_atoms) == 2: # Accounting for H-addition to sp2 carbon:
            bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
            u, v = np.subtract(bound_coords[0], C_coords), np.subtract(bound_coords[1], C_coords)
            n = np.array(geom_funs.normalize(-np.add(u, v)))
            H_coords = np.add(C_coords, n * BL_BA["CH"])
            graph, _ = add_atom(temp_mol, graph, "H", H_coords, BL_BA["CH"], C_idx)
        elif len(bound_atoms) == 3: # Accounting for H-addition to sp3 carbon with missing H:
            bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
            n = np.array(geom_funs.normalize(np.subtract(C_coords, bound_coords[0])))
            for coord in bound_coords[1:]:
                H_coords = geom_funs.PointRotateAxis(n, C_coords, coord, 2.0944)
                if all(geom_funs.distance(H_coords, c) >= 0.6 for c in bound_coords):
                    break
            graph, _ = add_atom(temp_mol, graph, "H", H_coords, BL_BA["CH"], C_idx)
    if num_H_to_add == 2:
        if len(bound_atoms) == 1: # Accounting for H-addition to a terminal alkene
            bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
            n = np.array(geom_funs.normalize(np.subtract(C_coords, bound_coords[0])))
            H1_coords = np.add(C_coords, n * BL_BA["CH"])
            graph, H1_idx = add_atom(temp_mol, graph, "H", H1_coords, BL_BA["CH"], C_idx)
            temp_mol.ACM(H1_idx, C_idx, bound_atoms[0], 120)
            H1_coords = temp_mol.getAtomCoords(H1_idx)
            H2_coords = geom_funs.PointRotateAxis(n, C_coords, H1_coords, 3.14159)
            graph, _ = add_atom(temp_mol, graph, "H", H2_coords, BL_BA["CH"], C_idx)
        elif len(bound_atoms) == 2: # Accounting for H-addition to an sp3 carbon with missing Hs:
            bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
            n = np.array(geom_funs.normalize(np.subtract(C_coords, bound_coords[0])))
            H1_coords = geom_funs.PointRotateAxis(n, C_coords, bound_coords[1], 2.0944)
            graph, H1_idx = add_atom(temp_mol, graph, "H", H1_coords, BL_BA["CH"], C_idx)
            H2_coords = geom_funs.PointRotateAxis(n, C_coords, H1_coords, 2.0944)
            graph, _ = add_atom(temp_mol, graph, "H", H2_coords, BL_BA["CH"], C_idx)
    if num_H_to_add == 3:
        bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
        n = np.array(geom_funs.normalize(np.subtract(C_coords, bound_coords[0])))
        H1_coords = np.add(C_coords, n * BL_BA["CH"])
        graph, H1_idx = add_atom(temp_mol, graph, "H", H1_coords, BL_BA["CH"], C_idx)
        temp_mol.ACM(H1_idx, C_idx, bound_atoms[0], 120)
        H1_coords = temp_mol.getAtomCoords(H1_idx)
        H2_coords = geom_funs.PointRotateAxis(n, C_coords, H1_coords, 2.0944)
        graph, _ = add_atom(temp_mol, graph, "H", H2_coords, BL_BA["CH"], C_idx)
        H3_coords = geom_funs.PointRotateAxis(n, C_coords, H2_coords, 2.0944)
        graph, _ = add_atom(temp_mol, graph, "H", H3_coords, BL_BA["CH"], C_idx)

    return graph

        
def water_cap(temp_mol, graph, O_idx):

    O_coords = temp_mol.getAtomCoords(O_idx)
    bound_atoms = temp_mol.getBondedAtomsSmart(O_idx, oct=False)

    metal_coords = temp_mol.getAtomCoords(bound_atoms[0])
    n = np.array(geom_funs.normalize(np.subtract(O_coords, metal_coords)))
    H1_coords = np.add(O_coords, n * BL_BA["OH"])
    graph, H1_idx = add_atom(temp_mol, graph, "H", H1_coords, BL_BA["OH"], O_idx)
    temp_mol.ACM(H1_idx, O_idx, bound_atoms[0], BL_BA["MOH"])
    H1_coords = temp_mol.getAtomCoords(H1_idx)
    H2_coords = geom_funs.PointRotateAxis(n, O_coords, H1_coords, 1.937315)
    graph, _ = add_atom(temp_mol, graph, "H", H2_coords, BL_BA["OH"], O_idx)

    return graph


def cap_nitrogen(temp_mol, graph, N_idx, num_H_to_add):

    N_coords = temp_mol.getAtomCoords(N_idx)
    bound_atoms = temp_mol.getBondedAtomsSmart(N_idx, oct=False)

    H_added = 0
    while H_added < num_H_to_add:
        if len(bound_atoms) == 1:
            bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
            n = np.array(geom_funs.normalize(np.subtract(N_coords, bound_coords[0])))
            H1_coords = np.add(N_coords, n * BL_BA["NH"])
            graph, H1_idx = add_atom(temp_mol, graph, "H", H1_coords, BL_BA["NH"], N_idx)
            temp_mol.ACM(H1_idx, N_idx, bound_atoms[0], 120)
            H_added += 1
            bound_atoms = temp_mol.getBondedAtomsSmart(N_idx, oct=False)
            continue
        bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
        n = np.array(geom_funs.normalize(np.subtract(N_coords, bound_coords[0])))
        for coord in bound_coords[1:]:
            new_coords = geom_funs.PointRotateAxis(n, N_coords, coord, 2.0944)
            if all(geom_funs.distance(new_coords, c) >= 0.6 for c in bound_coords):
                break
        graph, _ = add_atom(temp_mol, graph, "H", new_coords, BL_BA["NH"], N_idx)
        bound_atoms = temp_mol.getBondedAtomsSmart(N_idx, oct=False)
        H_added += 1

    return graph


def add_phosphite(temp_mol, graph, P_idx, netnegcharge):
    
    try:
        P_coords = temp_mol.getAtomCoords(P_idx)
        graph, bound_atoms = break_periodic_bonds(temp_mol, graph, P_idx)
        assert (
            len(bound_atoms) <= 3 and
                all(temp_mol.getAtom(x).symbol() == "O" for x in bound_atoms)
        ), "cannot cap phosphite"

        while len(bound_atoms) < 4:
            if len(bound_atoms) == 1:
                bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
                n = np.array(geom_funs.normalize(np.subtract(P_coords, bound_coords[0])))
                O_coords = np.add(P_coords, n * BL_BA["PO"])
                graph, O_idx = add_atom(temp_mol, graph, "O", O_coords, BL_BA["PO"], P_idx)
                temp_mol.ACM(O_idx, P_idx, bound_atoms[0], 110)
                bound_atoms = temp_mol.getBondedAtomsSmart(P_idx, oct=False)
                continue
            bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
            n = np.array(geom_funs.normalize(np.subtract(P_coords, bound_coords[0])))
            for coord in bound_coords[1:]:
                new_coords = geom_funs.PointRotateAxis(n, P_coords, coord, 2.0944)
                if all(geom_funs.distance(new_coords, c) >= 0.6 for c in bound_coords):
                    break
            if len(bound_atoms) == 3:
                graph, _ = add_atom(temp_mol, graph, "H", new_coords, BL_BA["PH"], P_idx)
                bound_atoms = temp_mol.getBondedAtomsSmart(P_idx, oct=False)
                continue
            graph, _ = add_atom(temp_mol, graph, "O", new_coords, BL_BA["PO"], P_idx)
            bound_atoms = temp_mol.getBondedAtomsSmart(P_idx, oct=False)

        return graph, netnegcharge + 2
    except AssertionError:
        if (len(bound_atoms) == 4 and
            sum(temp_mol.getAtom(x).symbol() == "O" for x in bound_atoms) == 3):

            return graph, netnegcharge + 2
        else:

            return graph, netnegcharge
    

def add_bisulfite(temp_mol, graph, S_idx, netnegcharge):

    try:
        S_coords = temp_mol.getAtomCoords(S_idx)
        graph, bound_atoms = break_periodic_bonds(temp_mol, graph, S_idx)
        assert(
            len(bound_atoms) <= 2 and
            all(temp_mol.getAtom(x).symbol() == "O" for x in bound_atoms)
        ), "cannot add bisulfite"
        
        while len(bound_atoms) < 3:
            if len(bound_atoms) == 1:
                bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
                n = np.array(geom_funs.normalize(np.subtract(S_coords, bound_coords[0])))
                O_coords = np.add(S_coords, n * BL_BA["SO"])
                graph, O_idx = add_atom(temp_mol, graph, "O", O_coords, BL_BA["SO"], S_idx)
                temp_mol.ACM(O_idx, S_idx, bound_atoms[0], BL_BA["OSO"])
                bound_atoms = temp_mol.getBondedAtomsSmart(S_idx, oct=False)
                continue
            bound_coords = [temp_mol.getAtomCoords(x) for x in bound_atoms]
            n = np.array(geom_funs.normalize(np.subtract(S_coords, bound_coords[0])))
            new_coords = geom_funs.PointRotateAxis(n, S_coords, bound_coords[1], 2.1485)
            graph, O_idx = add_atom(temp_mol, graph, "O", new_coords, BL_BA["SO"], S_idx)
            bound_atoms = temp_mol.getBondedAtomsSmart(S_idx, oct=False)
        
        n = np.array(geom_funs.normalize(np.subtract(new_coords, S_coords)))
        H_coords = np.add(new_coords, n * BL_BA["OH"])
        graph, H_idx = add_atom(temp_mol, graph, "H", H_coords, BL_BA["OH"], O_idx)
        temp_mol.ACM(H_idx, O_idx, S_idx, BL_BA["SOH"])
        
        return graph, netnegcharge + 1
    except AssertionError:
        if (len(bound_atoms) == 3 and
            sum(temp_mol.getAtom(x).symbol() == "O" for x in bound_atoms) == 3):

            return graph, netnegcharge + 1
        else:

            return graph, netnegcharge
        

def account_for_cycle_charges(temp_mol, graph, netnegcharge):

    cycles, graph, _ = get_no_metal_cycles(temp_mol, graph)

    for cycle in cycles:
        if len(cycle) != 5:
            continue
        count_N = sum(1 for elem in cycle if temp_mol.getAtom(elem).symbol() == "N")
        atoms_ring_bound = [
            bound_atom for elem in cycle
            for bound_atom in temp_mol.getBondedAtomsSmart(elem, oct=False)
            if bound_atom not in cycle
        ]
        count_non_metal_bonds = sum(1 for at in atoms_ring_bound
                                    if at not in temp_mol.findMetal(transition_metals_only=False))
        if (count_N, count_non_metal_bonds) in [(2, 3), (3, 2), (4, 1)]:
            netnegcharge += 1
        elif count_N == 3 and count_non_metal_bonds == 1:
            netnegcharge += 2
    
    return netnegcharge


def overlapping_acetate_check(temp_mol, graph):

    atom_coords = np.array([temp_mol.getAtomCoords(x) for x in range(temp_mol.natoms)])
    dist_matrix = np.linalg.norm(atom_coords[:, np.newaxis] - atom_coords, axis=2)

    cycles, graph, _ = get_no_metal_cycles(temp_mol, graph)
    atoms_to_ignore_for_overlap = set(
        bound_at for cycle in cycles for at in cycle
        for bound_at in temp_mol.getBondedAtomsSmart(at, oct=False)
        if bound_at not in temp_mol.findMetal(transition_metals_only=False)
    )
    ignore_mask = np.isin(np.arange(temp_mol.natoms), list(atoms_to_ignore_for_overlap))

    non_bonded_dist_matrix = np.where(
        (graph == 0) & ~ignore_mask[:, None] & ~ignore_mask[None, :],
        dist_matrix,
        0
    )

    overlapping_atoms = np.where(
        (non_bonded_dist_matrix < 0.75) &
        (non_bonded_dist_matrix != 0)
    )

    overlapping_atoms_dists = [(i + 1, j + 1, non_bonded_dist_matrix[i, j]) for i, j in zip(*overlapping_atoms)]

    has_overlap = bool(overlapping_atoms_dists)
    
    return has_overlap, overlapping_atoms_dists


def get_constraints(temp_mol):

    metals = set(temp_mol.findMetal(transition_metals_only=False))
    constraint_atoms = set()

    for metal in metals:
        first_shell_atoms = temp_mol.getBondedAtomsSmart(metal, oct=False)
        constraint_atoms.update(first_shell_atoms)
        for at in first_shell_atoms:
            second_shell = temp_mol.getBondedAtomsSmart(at, oct=False)
            constraint_atoms.update(second_shell)
    
    constraint_atoms -= metals # Remoe the metals from the list of constrained atoms
    
    return constraint_atoms


def cap_sbu(temp_mol, graph, netnegcharge, formate_flag=False):

    replace_wrong_H(temp_mol, graph)
    C_indices, graph, netnegcharge = get_C_for_capping(temp_mol, graph, netnegcharge)
    O_indices, graph, netnegcharge = get_O_for_capping(temp_mol, graph, netnegcharge)
    for C_idx in C_indices["methyl"][0]:
        graph = add_methyl(temp_mol, graph, C_idx, formate_flag=formate_flag)
    for C_idx in C_indices["acetate"][0]:
        graph = add_acetate(temp_mol, graph, C_idx, formate_flag=formate_flag)
    for i, C_idx in enumerate(C_indices["H"][0]):
        graph = cap_terminal_carbons(temp_mol, graph, C_idx, C_indices["H"][1][i])
    for O_idx in O_indices:
        graph = water_cap(temp_mol, graph, O_idx)
    N_indices, graph, netnegcharge = get_N_for_capping(temp_mol, graph, netnegcharge)
    for N_idx, num_H_to_add in N_indices:
        graph = cap_nitrogen(temp_mol, graph, N_idx, num_H_to_add)
    for P_idx in temp_mol.findAtomsbySymbol("P"):
        graph, netnegcharge = add_phosphite(temp_mol, graph, P_idx, netnegcharge)
    for S_idx in temp_mol.findAtomsbySymbol("S"):
        graph, netnegcharge = add_bisulfite(temp_mol, graph, S_idx, netnegcharge)
    netnegcharge = account_for_cycle_charges(temp_mol, graph, netnegcharge)

    return graph, netnegcharge


def identify_bad_sbus(base_dir):

    # Identifies only SBUs with free atoms
    problematic_sbus = []
    for sbu in os.listdir(base_dir + "/capped_sbus/sbus/"):
        temp_mol = mol3D()
        temp_mol.readfromxyz(base_dir + "/capped_sbus/sbus/" + sbu)
        if any(len(temp_mol.getBondedAtomsSmart(x, oct=False)) == 0 for x in range(temp_mol.natoms)):
            problematic_sbus.append(sbu[:-4])
    
    df =pd.DataFrame({
        "name": problematic_sbus
    })
    df.to_csv(base_dir + "/problematic_sbus_after_capping.csv", index=False)
        
    return


def main():
    p = get_paths()
    base_dir = p.sbu_clusters_dir
    os.makedirs(base_dir + "/capped_sbus", exist_ok=True)
    os.makedirs(base_dir + "/capped_sbus/sbus", exist_ok=True)
    os.makedirs(base_dir + "/capped_sbus/nets", exist_ok=True)
    path_to_sbus = base_dir + "/final_sbus/sbus"
    path_to_nets = base_dir + "/final_sbus/nets"
    all_sbus, all_atomic_constraints, all_netnegcharges = [], [], []
    for sbu in os.listdir(path_to_sbus):
        print(sbu)
        temp_mol = mol3D()
        temp_mol.readfromxyz(path_to_sbus + "/" + sbu)
        _, graph = sbu_funs.read_net(path_to_nets + "/" + sbu[:-4] + ".net")
        netnegcharge = 0
        temp_mol.graph = graph
        temp_mol_copy = copy.deepcopy(temp_mol)
        graph_copy = copy.deepcopy(graph)
        graph_copy, netnegcharge = cap_sbu(temp_mol_copy, graph_copy, netnegcharge)
        has_overlap, overlapping_atoms_dists = overlapping_acetate_check(temp_mol_copy, graph_copy)
        if has_overlap:
            print(f"{sbu} has non-bonding overlapping atoms: {overlapping_atoms_dists}, "
                  "capping with formates instead of acetates")
            graph, _ = cap_sbu(temp_mol, graph, netnegcharge, formate_flag=True)
        else:
            temp_mol, graph = temp_mol_copy, graph_copy
        temp_mol.graph = graph
        temp_mol.writexyz(base_dir + "/capped_sbus/sbus/" + sbu)
        with open(base_dir + "/capped_sbus/nets/" + sbu[:-4] + ".net", "w") as net:
            np.savetxt(net, np.array(temp_mol.graph), delimiter=',', fmt='%d', newline='\n')
        constraint_atoms = get_constraints(temp_mol)
        all_sbus.append(sbu[:-4])
        all_atomic_constraints.append(constraint_atoms)
        all_netnegcharges.append(netnegcharge)
    
    df = pd.DataFrame({
        "name": all_sbus,
        "atoms_to_constrain": all_atomic_constraints,
        "net_negative_charge": all_netnegcharges,
    })
    df.to_csv(base_dir + "/capped_sbus_constraints_netnegativecharges.csv", index=False)
    
    identify_bad_sbus(base_dir)

    return


if __name__ == "__main__":
    main()
