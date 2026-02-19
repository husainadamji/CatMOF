import os
import pandas as pd
import shutil
import cluster_extraction as cluster_funs
import featurization as accessory_funs
from molSimplify.Classes.mol3D import mol3D
import numpy as np
import networkx as nx
from molSimplify.Classes.globalvars import globalvars
import molSimplify.Informatics.MOF.PBC_functions as pbc_funs
from itertools import combinations
from data.atomic_data import metal_atom_radii
from catalmof.paths import get_paths, get_sbu_input_csv, get_config, DEFAULT_METALS

def populate_cifs(base_dir, filtered_cifs_csv_path, core_cifs_dir):

    df = pd.read_csv(filtered_cifs_csv_path)
    mof_names = df['name'].values

    for mof_name in mof_names:
        path_to_cif = os.path.join(core_cifs_dir, mof_name + '.cif')
        shutil.copy(path_to_cif, base_dir + '/cif')

    return


def get_sbu_clusters(base_dir):

    for cif_file in os.listdir(base_dir + '/cif/'):
        cluster_funs.get_primitive(base_dir + '/cif/' + cif_file, base_dir + '/primitive/' + cif_file)
        full_names, full_descriptors = cluster_funs.get_MOF_descriptors(base_dir + '/primitive/' + cif_file, 
                                                                        3, path = base_dir, 
                                                                        xyzpath = base_dir + '/xyz/' + cif_file.replace('cif', 'xyz'))

    return

def get_sbus_w_duplicate_atoms(base_dir, debug=True):

    sbus = [sbu[:-4] for sbu in os.listdir(base_dir + '/sbus') if sbu.endswith('.xyz')]
    duplicate_pairs = []
    for sbu in sbus:
        temp_mol = mol3D()
        temp_mol.readfromxyz(base_dir + '/sbus/' + sbu + '.xyz')
        temp_duplicate_pairs = []
        for i in list(range(temp_mol.natoms)):
            for j in list(range(i + 1, temp_mol.natoms)):
                dist = temp_mol.getAtom(i).distance(temp_mol.getAtom(j))
                if dist == 0.0:
                    temp_duplicate_pairs.append((i + 1, j + 1))
        duplicate_pairs.append(temp_duplicate_pairs)
    
    duplicate_df = pd.DataFrame({
        "sbu": sbus,
        "duplicate_atom_indices": duplicate_pairs
    })
    if debug:
        duplicate_df.to_csv(base_dir + '/sbus_w_overlapping_atoms.csv', index=False)

    return duplicate_df

def read_net(net_path, skip_first_line=True):

    with open(net_path, 'r') as net:
        lines = net.readlines()

    if skip_first_line:
        atom_list = lines[0].strip().split(',')[1:] # Skip the first element ('#')
        lines = lines[1:]
    else:
        atom_list = None

    graph = np.array([list(map(int, line.strip().split(','))) for line in lines])

    if skip_first_line:
        return atom_list, graph
    else:
        return graph

def fix_sbus_w_duplicate_atoms(base_dir, duplicate_df):

    filtered_df = duplicate_df[[len(indices) > 0 for indices in duplicate_df["duplicate_atom_indices"]]]
    problematic_sbus = filtered_df["sbu"].tolist()
    duplicate_at_indices = filtered_df["duplicate_atom_indices"].tolist()

    for i, problematic_sbu in enumerate(problematic_sbus):
        temp_mol = mol3D()
        temp_mol.readfromxyz(base_dir + '/sbus/' + problematic_sbu + '.xyz')
        _, graph = read_net(base_dir +'/sbus/' + problematic_sbu + '.net')
        temp_mol.graph = graph
        dupes = duplicate_at_indices[i]
        atoms_to_discard = sorted(set([x[1] - 1 for x in dupes]))
        temp_mol.deleteatoms(atoms_to_discard)
        revised_atom_list = [x.symbol() for x in temp_mol.getAtoms()]
        revised_atom_line = '# ' + ','.join(revised_atom_list)

        with open(base_dir +'/sbus/' + problematic_sbu + '.net', 'w') as revised_net:
            revised_net.write(revised_atom_line + '\n')
        with open(base_dir +'/sbus/' + problematic_sbu + '.net', 'a') as revised_net:
            np.savetxt(revised_net, np.array(temp_mol.graph), delimiter=',', fmt='%d', newline='\n')

        temp_mol.writexyz(base_dir + '/sbus/' + problematic_sbu + '.xyz')

    return

def get_unique_sbus(base_dir, save_graph_hashes=True):

    globs = globalvars()
    amassdict = globs.amass()
    sbus = [sbu[:-4] for sbu in os.listdir(base_dir + '/sbus') if sbu.endswith('.xyz')]

    graph_hashes = []
    for sbu in sbus:
        _, graph = read_net(base_dir + '/sbus/' + sbu + '.net')
        temp_mol = mol3D()
        temp_mol.readfromxyz(base_dir + '/sbus/' + sbu + '.xyz')
        temp_mol.graph = graph
        temp_graph = nx.Graph()
        for i, row in enumerate(graph):
            for j, column in enumerate(row):
                if graph[i][j] == 1:
                    temp_graph.add_edge(i, j, weight=str(amassdict[temp_mol.getAtom(i).symbol()][0]*amassdict[temp_mol.getAtom(j).symbol()][0]))
                if i == j:
                    temp_graph.add_edge(i, j, weight=str(amassdict[temp_mol.getAtom(i).symbol()][0]))
        temp_graph_hash = nx.weisfeiler_lehman_graph_hash(temp_graph, edge_attr='weight')
        graph_hashes.append(temp_graph_hash)

    if save_graph_hashes:
        hash_df = pd.DataFrame({
            "sbu": sbus,
            "graph_hash": graph_hashes
        })

        hash_df.to_csv(base_dir + '/weisfeiler_lehman_graph_hashes_all_sbus.csv', index=False)

    path_to_unique_sbus = base_dir + '/unique_sbus'
    os.makedirs(path_to_unique_sbus, exist_ok=True)
    os.makedirs(path_to_unique_sbus + "/sbus", exist_ok=True)
    os.makedirs(path_to_unique_sbus + "/nets", exist_ok=True)

    unique_sbus = []
    unique_hashes = []
    for sbu, graph_hash in zip(sbus, graph_hashes):
        if graph_hash not in unique_hashes:
            unique_sbus.append(sbu)
            unique_hashes.append(graph_hash)
    for unique_sbu in unique_sbus:
        shutil.copy(base_dir + '/sbus/' + unique_sbu + '.xyz', path_to_unique_sbus + '/sbus/' + unique_sbu + '.xyz')
        shutil.copy(base_dir + '/sbus/' + unique_sbu + '.net', path_to_unique_sbus + '/nets/' + unique_sbu + '.net')

    return

def compute_geometry_index(cn, angle_list):

    angle_list.sort()
    max_angle = angle_list[-1]
    min_angle = angle_list[-2]
    if cn == 5:
        tau = (max_angle - min_angle) / 60
    elif cn == 4:
        tau = ((max_angle - min_angle) / (360 - 109.5)) + ((180 - max_angle) / (180 - 109.5))
    
    return tau

def sbu_cn_analysis(sbu, metals_of_interest, path_to_xyz, path_to_net):

    mof = '_'.join(sbu.split('_')[:sbu.split('_').index('sbu')])
    _, graph = read_net(path_to_net + '/' + sbu[:-4] + '.net')
    temp_mol = mol3D()
    temp_mol.readfromxyz(path_to_xyz + '/' + sbu)
    temp_mol.graph = graph
    metal_inds = temp_mol.findMetal(transition_metals_only=False)
    filtered_metal_inds = [idx for idx in metal_inds if temp_mol.getAtom(idx).symbol() in metals_of_interest]
    metals = temp_mol.getAtomwithinds(filtered_metal_inds)

    results = []
    for i, metal in enumerate(metals):
        metal_sym = metal.symbol()
        coord_sphere_inds = temp_mol.getBondedAtomsSmart(filtered_metal_inds[i], oct=False)
        cn = len(coord_sphere_inds)
        if cn in [4, 5]:
            angle_list = [temp_mol.getAngle(pair[0], filtered_metal_inds[i], pair[1]) for pair in list(combinations(coord_sphere_inds, 2))]
            tau = compute_geometry_index(cn, angle_list)
        else:
            tau = np.nan
        results.append((mof, sbu[:-4], metal_sym, cn, tau))

    return results

def cn_analysis(base_dir, metals_of_interest, path_to_xyz, path_to_net):

    mofs, sbus, full_metal_list, full_cn_list, full_tau_list = [], [], [], [], []
    for sbu in os.listdir(path_to_xyz):
        results = sbu_cn_analysis(sbu, metals_of_interest, path_to_xyz, path_to_net)
        for mof, sbu_name, metal, cn, tau in results:
            mofs.append(mof)
            sbus.append(sbu_name)
            full_metal_list.append(metal)
            full_cn_list.append(cn)
            full_tau_list.append(tau)

    df = pd.DataFrame({
        'mof': mofs,
        'sbu': sbus,
        'metal': full_metal_list,
        'cn': full_cn_list,
        'tau': full_tau_list
    })

    df.to_csv(base_dir + '/metal_cn_distribution.csv', index=False)

    return df

def filter_oms_sbus(base_dir, metal_cn_df, tau_4_cutoff=0.6, tau_5_cutoff=0.45):

    mofs, uniq_sbus, metals = [], [], []
    for i, row in metal_cn_df.iterrows():
        cn, tau, sbu = int(row['cn']), float(row['tau']), row['sbu']
        if (cn < 4 or (cn == 4 and tau < tau_4_cutoff) or (cn == 5 and tau < tau_5_cutoff)) and sbu not in uniq_sbus:
            mofs.append(row['mof'])
            uniq_sbus.append(sbu)
            # metals.append(row['metal'])
            temp_mol = mol3D()
            temp_mol.readfromxyz(base_dir + '/unique_sbus/sbus/' + sbu + '.xyz')
            metals.append(set([temp_mol.getAtom(x).symbol() for x in temp_mol.findMetal(transition_metals_only=False)]))
    
    df = pd.DataFrame({
        'mof': mofs,
        'sbu': uniq_sbus,
        'metal': metals
    })
    df.to_csv(base_dir + '/oms_sbus.csv', index=False)

    path_to_oms_sbus = base_dir + '/oms_sbus'
    os.makedirs(path_to_oms_sbus, exist_ok=True)
    os.makedirs(path_to_oms_sbus + "/sbus", exist_ok=True)
    os.makedirs(path_to_oms_sbus + "/nets", exist_ok=True)

    for sbu in uniq_sbus:
        shutil.copy(base_dir + '/unique_sbus/sbus/' + sbu +'.xyz', path_to_oms_sbus + '/sbus/' + sbu + '.xyz')
        shutil.copy(base_dir + '/unique_sbus/nets/' + sbu +'.net', path_to_oms_sbus + '/nets/' + sbu + '.net')

    return

def compute_fsr(temp_mol, metal_inds, metal_atom_radii, fsr_cutoff=1.1):

    distances, fsrs = [], []
    metal_inds_combs = list(combinations(metal_inds, 2))
    for comb in metal_inds_combs:
        coords1 = np.array(temp_mol.getAtomCoords(comb[0]))
        coords2 = np.array(temp_mol.getAtomCoords(comb[1]))
        dist = np.linalg.norm(coords2 - coords1)
        distances.append(dist)

        metal_rad1 = next(element["metallic_radius"] for element in metal_atom_radii if element["symbol"] == temp_mol.getAtom(comb[0]).symbol())
        metal_rad2 = next(element["metallic_radius"] for element in metal_atom_radii if element["symbol"] == temp_mol.getAtom(comb[1]).symbol())
        fsr = dist / (metal_rad1 + metal_rad2)
        fsrs.append(fsr)

    mm_bond = any(fsr < fsr_cutoff for fsr in fsrs)

    return metal_inds_combs, distances, fsrs, mm_bond

def fsr_analysis(path_to_sbus, metal_atom_radii):

    results = {
        'sbu': [],
        'metal_indices': [],
        'metal_pairs': [],
        'metal_pair_distances': [],
        'formal_shortness_ratio': [],
        'metal-metal_bond?': []
    }

    for sbu in os.listdir(path_to_sbus):
        results['sbu'].append(sbu[:-4])
        temp_mol = mol3D()
        temp_mol.readfromxyz(path_to_sbus + '/' + sbu)
        metal_inds = temp_mol.findMetal(transition_metals_only=False)
        results['metal_indices'].append(metal_inds)

        if len(metal_inds) == 1:
            results['metal_pairs'].append('')
            results['metal_pair_distances'].append('')
            results['formal_shortness_ratio'].append('')
            results['metal-metal_bond?'].append(False)
        else:
            metal_inds_combs, distances, fsrs, mm_bond = compute_fsr(temp_mol, metal_inds, metal_atom_radii)
            results['metal_pairs'].append(metal_inds_combs)
            results['metal_pair_distances'].append(distances)
            results['formal_shortness_ratio'].append(fsrs)
            results['metal-metal_bond?'].append(mm_bond)

    df = pd.DataFrame(results)
    df.to_csv(os.path.abspath(os.path.join(path_to_sbus, os.pardir, os.pardir)) + '/oms_sbus_metal_distances.csv', index=False)

    return df

def filter_sbus_post_fsr(base_dir, metal_dist_df):

    filtered_df = metal_dist_df[metal_dist_df['metal-metal_bond?'] == False]
    sbus = filtered_df['sbu'].values

    path_to_post_fsr_sbus = base_dir + '/sbus_post_fsr'
    os.makedirs(path_to_post_fsr_sbus, exist_ok=True)
    os.makedirs(path_to_post_fsr_sbus + "/sbus", exist_ok=True)
    os.makedirs(path_to_post_fsr_sbus + "/nets", exist_ok=True)

    for sbu in sbus:
        shutil.copy(base_dir + '/oms_sbus/sbus/' + sbu +'.xyz', path_to_post_fsr_sbus + '/sbus/' + sbu + '.xyz')
        shutil.copy(base_dir + '/oms_sbus/nets/' + sbu +'.net', path_to_post_fsr_sbus + '/nets/' + sbu + '.net')

    return

def check_crystal_quality(base_dir, rfactor_data, rfactor_cutoff=10, debug=True):

    all_rfactor_df = pd.read_csv(rfactor_data)
    mof_names = all_rfactor_df['name'].values
    all_rfactors = all_rfactor_df['R-factor'].values


    sbus_post_fsr, mofs_post_fsr, rfactors_post_fsr = [], [], []
    for sbu in os.listdir(base_dir + '/sbus_post_fsr/sbus'):
        sbus_post_fsr.append(sbu[:-4])
        match_found = False
        for mof_name, rfactor in zip(mof_names, all_rfactors):
            if mof_name in sbu:
                mofs_post_fsr.append(mof_name)
                rfactors_post_fsr.append(rfactor)
                match_found = True
                break
        if not match_found:
            mofs_post_fsr.append(np.nan)
            rfactors_post_fsr.append(np.nan)

    rfactor_df = pd.DataFrame({
        'sbu': sbus_post_fsr,
        'mof': mofs_post_fsr,
        'R-factor': rfactors_post_fsr
    })

    if debug:
        rfactor_df.to_csv(base_dir + '/post_fsr_sbu_rfactors.csv', index=False)
    
    filtered_rfactor_df = rfactor_df[rfactor_df['R-factor'].astype(float) < rfactor_cutoff]
    
    return filtered_rfactor_df

def filter_mof_sbus_w_halides(base_dir, rfactor_df, debug=True):

    halogens = {'F', 'Cl', 'Br', 'I', 'At', 'Ts'}
    # rfactor_df can have 'mof' column (from R-factor check) or 'sbu' column (if R-factor bypassed)
    if 'mof' in rfactor_df.columns:
        mofs = rfactor_df['mof'].values
    else:
        # Extract MOF name from SBU name (format matches sbu_cn_analysis: mofname_sbu...)
        mofs = ['_'.join(sbu.split('_')[:sbu.split('_').index('sbu')]) for sbu in rfactor_df['sbu'].values]

    has_halogen = []
    for mof in mofs:
        if pd.isna(mof) or mof == '':
            has_halogen.append(False)  # Skip if MOF name is missing
            continue
        try:
            _, atomlist, _ = pbc_funs.readcif(base_dir + '/cif/' + str(mof) + '.cif')
            has_halogen.append(any(atom in halogens for atom in atomlist))
        except (FileNotFoundError, ValueError):
            has_halogen.append(False)  # Skip if CIF not found
    
    halide_df = rfactor_df.copy()
    halide_df['has_halogen'] = has_halogen
    if debug:
        halide_df.to_csv(base_dir + '/final_mofs_w_halogen_info.csv', index=False)
    
    path_to_final_sbus = base_dir + '/final_sbus'
    os.makedirs(path_to_final_sbus, exist_ok=True)
    os.makedirs(path_to_final_sbus + "/sbus", exist_ok=True)
    os.makedirs(path_to_final_sbus + "/nets", exist_ok=True)

    # Get SBU column name (could be 'sbu' or from rfactor_df structure)
    sbu_col = 'sbu' if 'sbu' in halide_df.columns else halide_df.columns[0]  # fallback to first column if 'sbu' not present
    final_sbus = halide_df[halide_df['has_halogen'] == False][sbu_col].tolist()
    for sbu in final_sbus:
        shutil.copy(base_dir + '/sbus_post_fsr/sbus/' + sbu +'.xyz', path_to_final_sbus + '/sbus/' + sbu + '.xyz')
        shutil.copy(base_dir + '/sbus_post_fsr/nets/' + sbu +'.net', path_to_final_sbus + '/nets/' + sbu + '.net')

    return

def main():
    p = get_paths()
    env_metals = os.environ.get("CATALMOF_METALS")
    if env_metals:
        metals_of_interest = [m.strip() for m in env_metals.split(",") if m.strip()]
    else:
        metals_of_interest = DEFAULT_METALS
    base_dir = p.sbu_clusters_dir
    os.makedirs(base_dir, exist_ok=True)
    path_to_primitive = accessory_funs.build_dir_hierarchy(base_dir)
    populate_cifs(base_dir, get_sbu_input_csv(), p.core_cifs_dir)
    get_sbu_clusters(base_dir)
    duplicate_df = get_sbus_w_duplicate_atoms(base_dir)
    fix_sbus_w_duplicate_atoms(base_dir, duplicate_df)
    get_unique_sbus(base_dir)
    metal_cn_df = cn_analysis(base_dir, metals_of_interest, base_dir + '/unique_sbus/sbus', base_dir + '/unique_sbus/nets')
    filter_oms_sbus(base_dir, metal_cn_df)
    metal_dist_df = fsr_analysis(base_dir + '/oms_sbus/sbus', metal_atom_radii)
    filter_sbus_post_fsr(base_dir, metal_dist_df)
    
    # R-factor check: optional, requires CSV with 'name' and 'R-factor' columns
    config = get_config()
    run_rfactor_check = config.get("run_rfactor_check", False)
    
    if run_rfactor_check:
        rfactor_csv = p.core_rfactors_csv
        if not rfactor_csv or not os.path.isfile(rfactor_csv):
            raise ValueError(
                "run_rfactor_check is True but core_rfactors_csv not found. "
                "Set paths.core_rfactors_csv in config to your R-factor CSV file. "
                "Required columns: 'name' (MOF refcode), 'R-factor' (numeric). "
                "You can obtain R-factor data separately via the CCDC/CSD API (CSD license required)."
            )
        rfactor_cutoff = config.get("rfactor_cutoff", 10)
        filtered_rfactor_df = check_crystal_quality(base_dir, rfactor_csv, rfactor_cutoff=rfactor_cutoff)
        filter_mof_sbus_w_halides(base_dir, filtered_rfactor_df)
    else:
        # Skip R-factor filtering; create minimal dataframe for halide filtering from SBU names
        sbus_post_fsr = [sbu[:-4] for sbu in os.listdir(base_dir + '/sbus_post_fsr/sbus')]
        sbu_df = pd.DataFrame({'sbu': sbus_post_fsr})
        filter_mof_sbus_w_halides(base_dir, sbu_df)

    return

if __name__ == "__main__":
    main()
