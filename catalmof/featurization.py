import os
import shutil
import molSimplify.Informatics.MOF.MOF_descriptors as mof_funs
import pandas as pd
import subprocess
import numpy as np

from catalmof.paths import get_paths

def build_dir_hierarchy(base_dir, directories=['cif', 'primitive', 'xyz']):

    for directory in directories:
        dir_path = base_dir + '/' + directory
        if directory == 'primitive':
            path_to_primitive = dir_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory created: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")
    
    return path_to_primitive

def populate_cifs(base_dir, filtered_cifs_csv_path):

    df = pd.read_csv(filtered_cifs_csv_path)
    mof_paths = df['path'].values

    for path in mof_paths:
        shutil.copy(path, base_dir + '/cif')

    return


def get_mof_racs(base_dir):

    featurization_list = []
    for cif in os.listdir(base_dir + '/cif'):
        try:
            mof_funs.get_primitive(base_dir + '/cif/' + cif, base_dir + '/primitive/' + cif)
            full_names, full_descriptors = mof_funs.get_MOF_descriptors(base_dir + '/primitive/' + cif, 3, path=base_dir, 
                                                            xyzpath=base_dir + '/xyz/' + cif.replace('cif', 'xyz'))
            full_names.append('filename')
            full_descriptors.append(cif)
            featurization = dict(zip(full_names, full_descriptors))
            featurization_list.append(featurization)
        except Exception as e:
            print(f"Error processing {cif}: {e}")
            continue

    df = pd.DataFrame(featurization_list)
    df = df.sort_values(by='filename').reset_index(drop=True)
    df = df.drop(columns=0, errors='ignore')
    df.to_csv(base_dir + '/mof_racs.csv', index=False)

    return df

def execute_geo_commands(path_to_zeo, path_to_primitive, path_to_zeo_txt_files):

    for cif in os.listdir(path_to_primitive):
        name = cif.replace('.cif', '')
        cmd1 = f"{path_to_zeo} -ha -res {path_to_zeo_txt_files}/{name}_pd.txt {path_to_primitive}/{cif}"
        cmd2 = f"{path_to_zeo} -sa 1.86 1.86 10000 {path_to_zeo_txt_files}/{name}_sa.txt {path_to_primitive}/{cif}"
        cmd3 = f"{path_to_zeo} -volpo 1.86 1.86 10000 {path_to_zeo_txt_files}/{name}_pov.txt {path_to_primitive}/{cif}"

        subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=None, shell=True).communicate()
        subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=None, shell=True).communicate()
        subprocess.Popen(cmd3, stdout=subprocess.PIPE, stderr=None, shell=True).communicate()

    return

def parse_geo_data_files(path_to_zeo_txt_files, path_to_primitive):

    dict_list = []
    
    for cif in os.listdir(path_to_primitive):
        basename = cif.replace('.cif', '')

        data = {
            "name": basename,
            "cif_file": cif,
            "Di": np.nan,
            "Df": np.nan,
            "Dif": np.nan,
            "rho": np.nan,
            "VSA": np.nan,
            "GSA": np.nan,
            "VPOV": np.nan,
            "GPOV": np.nan,
            "POAV_vol_frac": np.nan,
            "PONAV_vol_frac": np.nan,
            "GPOAV": np.nan,
            "GPONAV": np.nan,
            "POAV": np.nan,
            "PONAV": np.nan
        }

        data = parse_pore_diameter_file(f"{path_to_zeo_txt_files}/{basename}_pd.txt", data)
        data = parse_surface_area_file(f"{path_to_zeo_txt_files}/{basename}_sa.txt", data)
        data = parse_pore_volume_file(f"{path_to_zeo_txt_files}/{basename}_pov.txt", data)

        dict_list.append(data)

    return dict_list

def parse_pore_diameter_file(file_path, data):

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                line = lines[0]
                data["Di"] = float(line.split()[1])
                data["Df"] = float(line.split()[2])
                data["Dif"] = float(line.split()[3])
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
    except Exception as e:
        print(f"An error occured: {e}")

    return data

def parse_surface_area_file(file_path, data):

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                line = lines[0]
                data["rho"] = float(line.split('Unitcell_volume:')[1].split()[0])
                data["VSA"] = float(line.split('ASA_m^2/cm^3:')[1].split()[0])
                data["GSA"] = float(line.split('ASA_m^2/g:')[1].split()[0])
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
    except Exception as e:
        print(f"An error occured: {e}") 

    return data

def parse_pore_volume_file(file_path, data):

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                line = lines[0]
                data["POAV"] = float(line.split('POAV_A^3:')[1].split()[0])
                data["PONAV"] = float(line.split('PONAV_A^3:')[1].split()[0])
                data["GPOAV"] = float(line.split('POAV_cm^3/g:')[1].split()[0])
                data["GPONAV"] = float(line.split('PONAV_cm^3/g:')[1].split()[0])
                data["POAV_vol_frac"] = float(line.split('POAV_Volume_fraction:')[1].split()[0])
                data["PONAV_vol_frac"] = float(line.split('PONAV_Volume_fraction:')[1].split()[0])
                data["VPOV"] = data["POAV_vol_frac"] + data["PONAV_vol_frac"]
                data["GPOV"] = data["VPOV"] / data["rho"]
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
    except Exception as e:
        print(f"An error occured: {e}")

    return data

def get_geo_features(base_dir, path_to_zeo, path_to_primitive, path_to_zeo_txt_files):

    execute_geo_commands(path_to_zeo, path_to_primitive, path_to_zeo_txt_files)
    dict_list = parse_geo_data_files(path_to_zeo_txt_files, path_to_primitive)

    df = pd.DataFrame(dict_list)
    df = df.sort_values(by='cif_file').reset_index(drop=True)
    df.to_csv(base_dir + '/mof_geo_features.csv', index=False)

    return df

def merge_features(base_dir, racs_df, geo_df):

    features_df = pd.concat([racs_df, geo_df], axis=1)
    features_df.to_csv(base_dir + '/merged_features.csv', index=False)
    features_df = features_df.dropna()
    features_df.to_csv(base_dir + '/merged_features_featurizable_mofs.csv', index=False)

    return

def main():
    p = get_paths()
    base_dir = p.featurization_dir
    os.makedirs(base_dir, exist_ok=True)
    path_to_zeo = p.zeo_network
    if not path_to_zeo or not os.path.isfile(path_to_zeo):
        raise FileNotFoundError(
            f"Zeo++ binary not found at path: {path_to_zeo!r}. "
            "Download Zeo++ (version 0.3) from https://www.zeoplusplus.org/download.html, build the 'network' binary, "
            "then set paths.zeo_network in your config to the full path of the network executable."
        )
    path_to_zeo_txt_files = p.featurization_zeo_data
    os.makedirs(path_to_zeo_txt_files, exist_ok=True)
    path_to_primitive = build_dir_hierarchy(base_dir)
    populate_cifs(base_dir, p.metal_filtered_cifs_csv)
    racs_df = get_mof_racs(base_dir)
    geo_df = get_geo_features(base_dir, path_to_zeo, path_to_primitive, path_to_zeo_txt_files)
    merge_features(base_dir, racs_df, geo_df)

    return

if __name__ == "__main__":
    main()
