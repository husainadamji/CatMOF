import os
import pandas as pd
from molSimplify.Scripts.cellbuilder_tools import import_from_cif

from catmof.paths import get_paths, DEFAULT_METALS

def get_mof_metals(mof):
    metals = list(set(metal.symbol() for metal in mof.getAtomwithinds(mof.findMetal(transition_metals_only=False))))
    return metals

def process_core_version(core_path):
    cifs = []
    mof_paths = []
    mof_metals = []

    for cif in os.listdir(core_path):
        cifs.append(cif[:-4])
        mof_paths.append(core_path + '/' + cif)
        mof, _ = import_from_cif(core_path + '/' + cif)
        mof_metals.append(get_mof_metals(mof))
    
    return cifs, mof_paths, mof_metals

def get_core_mof_metals(core_cifs_dir, output_csv):

    cifs, mof_paths, mof_metals = process_core_version(core_cifs_dir)
    mof_metals_df = pd.DataFrame({'cif': cifs, 'path': mof_paths, 'metals': mof_metals})
    mof_metals_df.to_csv(output_csv, index=False)
    return cifs, mof_paths, mof_metals

def get_specific_metal_mofs(desired_metals, cifs, mof_paths, mof_metals, output_csv):

    filtered_cifs = []
    filtered_paths = []
    filtered_mof_metals = []
    for i, metals in enumerate(mof_metals):
        if any(metal in metals for metal in desired_metals):
            filtered_cifs.append(cifs[i])
            filtered_paths.append(mof_paths[i])
            filtered_mof_metals.append(metals)

    filtered_df = pd.DataFrame({'cif': filtered_cifs, 'path': filtered_paths, 'metals': filtered_mof_metals})
    filtered_df.to_csv(output_csv, index=False)
    return filtered_cifs, filtered_paths, filtered_mof_metals

def main():
    p = get_paths()
    cifs, mof_paths, mof_metals = get_core_mof_metals(p.core_cifs_dir, p.cifs_with_metals_csv)
    env_metals = os.environ.get("CATMOF_METALS")
    if env_metals:
        desired_metals = [m.strip() for m in env_metals.split(",") if m.strip()]
    else:
        desired_metals = DEFAULT_METALS
    _, _, _ = get_specific_metal_mofs(desired_metals, cifs, mof_paths, mof_metals, p.metal_filtered_cifs_csv)

if __name__ == "__main__":
    main()
