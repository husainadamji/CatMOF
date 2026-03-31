import os
import pandas as pd
import numpy as np

from catmof.paths import get_paths, get_config

def get_combined_activation_thermal_data(solvent_data_file, thermal_data_file):

    solvent_df = pd.read_csv(solvent_data_file)
    solvent_names = solvent_df['name'].values
    solvent_lse = solvent_df['lse'].values
    solvent_flags = solvent_df['predicted'].values
    solvent_flag_proba = solvent_df['probability'].values

    thermal_df = pd.read_csv(thermal_data_file)
    thermal_names = thermal_df['name'].values
    thermal_lsd = thermal_df['scaled_latent_10NN'].values
    thermal_T = thermal_df['predicted'].values

    for solvent_name, thermal_name in zip(solvent_names, thermal_names):
        if solvent_name != thermal_name:
            raise ValueError(f'refcode lists do not match')
        
    combined_df = pd.DataFrame({
        'name': solvent_names,
        'activation_lse': solvent_lse,
        'activation_flag': solvent_flags,
        'activation_flag_proba': solvent_flag_proba,
        'thermal_lsd': thermal_lsd,
        'thermal_T': thermal_T
    })

    return combined_df

def get_confident_preds(combined_df, lse_cutoff=0.19, lsd_cutoff=0.21):

    confident_df = pd.DataFrame(columns=combined_df.columns)
    for i, row in combined_df.iterrows():
        if row['activation_lse'] < lse_cutoff and row['thermal_lsd'] < lsd_cutoff:
            confident_df = pd.concat([confident_df, pd.DataFrame([row])], ignore_index=True)

    return confident_df


def get_confident_preds_activation_only(activation_df, lse_cutoff=0.19):
    """Confident = LSE below cutoff only."""
    return activation_df[activation_df["lse"] < lse_cutoff].copy()


def get_confident_preds_thermal_only(thermal_df, lsd_cutoff=0.21):
    """Confident = LSD below cutoff only."""
    return thermal_df[thermal_df["scaled_latent_10NN"] < lsd_cutoff].copy()


def get_stable_mofs(confident_df, T_threshold=300):

    stable_df = pd.DataFrame(columns=confident_df.columns)
    for i, row in confident_df.iterrows():
        if int(row['activation_flag']) == 1 and float(row['thermal_T']) > T_threshold:
            stable_df = pd.concat([stable_df, pd.DataFrame([row])], ignore_index=True)

    return stable_df


def get_stable_mofs_activation_only(confident_df):
    """Stable = activation flag == 1 (among LSE-confident)."""
    return confident_df[confident_df["predicted"].astype(int) == 1].copy()


def get_stable_mofs_thermal_only(confident_df, T_threshold=300):
    """Stable = predicted T > threshold (among LSD-confident)."""
    return confident_df[confident_df["predicted"].astype(float) > T_threshold].copy()

def check_list_equivalence(list1, list2, tol=1e-7):

    return all(np.isclose(list1_i, list2_i, atol=tol) for list1_i, list2_i in zip(list1, list2))

def _get_uniq_mcenv_representatives(stable_mofs, mc_racs_list, order_indices, tol=1e-7):
    """Given MOF names, their mc-RAC lists, and the order to iterate (indices), return the subset of
    stable_mofs that are the first occurrence of each unique mc-RAC set in that order."""
    uniq_stable_mofs = []
    uniq_mc_racs_list = []
    for idx in order_indices:
        mof = stable_mofs[idx]
        mc_racs = mc_racs_list[idx]
        if not any(check_list_equivalence(mc_racs, u, tol) for u in uniq_mc_racs_list):
            uniq_mc_racs_list.append(mc_racs)
            uniq_stable_mofs.append(mof)
    return uniq_stable_mofs


def _get_mc_racs_list(stable_mofs, solvent_data_file):
    """Load solvent CSV and return list of mc-RAC vectors (one per MOF in stable_mofs)."""
    solvent_df = pd.read_csv(solvent_data_file)
    mc_rac_names = [c for c in solvent_df.columns if 'mc' in c]
    return [
        solvent_df.loc[solvent_df['name'] == mof, mc_rac_names].values.flatten().tolist()
        for mof in stable_mofs
    ]


def get_stable_mofs_w_unique_mcenv(stable_df, solvent_data_file, tol=1e-7):
    """Among MOFs with identical mc-RACs, keep the one with highest predicted T (decomposition temp)."""
    stable_mofs = stable_df["name"].values
    predicted_T = stable_df["thermal_T"].values
    mc_racs_list = _get_mc_racs_list(stable_mofs, solvent_data_file)
    # Order by T descending so we keep highest-T representative per mc-RAC set
    order = np.argsort(-np.asarray(predicted_T, dtype=float))
    uniq_stable_mofs = _get_uniq_mcenv_representatives(stable_mofs, mc_racs_list, order, tol=tol)
    return stable_df[stable_df["name"].isin(uniq_stable_mofs)]


def get_stable_mofs_w_unique_mcenv_random(stable_df, solvent_data_file, tol=1e-7, random_state=None):
    """Among MOFs with identical mc-RACs, keep one representative chosen at random (no T available).
    Uses random_state for reproducibility when provided (e.g. integer seed)."""
    stable_mofs = stable_df["name"].values
    mc_racs_list = _get_mc_racs_list(stable_mofs, solvent_data_file)
    rng = np.random.default_rng(random_state)
    order = np.arange(len(stable_mofs))
    rng.shuffle(order)
    uniq_stable_mofs = _get_uniq_mcenv_representatives(stable_mofs, mc_racs_list, order, tol=tol)
    return stable_df[stable_df["name"].isin(uniq_stable_mofs)]

def main():
    p = get_paths()
    config = get_config()
    bypass_activation = config.get("bypass_activation_stability", False)
    bypass_thermal = config.get("bypass_thermal_stability", False)
    run_confidence_checks = config.get("run_confidence_checks", True)
    T_threshold = config.get("thermal_stability_threshold", 300)
    lse_cutoff = config.get("lse_cutoff", 0.19)
    lsd_cutoff = config.get("lsd_cutoff", 0.21)

    run_activation = not bypass_activation
    run_thermal = not bypass_thermal

    if run_activation and run_thermal:
        # Both: optionally LSE + LSD confidence, then flag + T for stable, then unique mcenv
        combined_df = get_combined_activation_thermal_data(
            p.activation_predictions_csv,
            p.thermal_predictions_csv,
        )
        confident_df = (
            get_confident_preds(combined_df, lse_cutoff=lse_cutoff, lsd_cutoff=lsd_cutoff)
            if run_confidence_checks
            else combined_df
        )
        stable_df = get_stable_mofs(confident_df, T_threshold=T_threshold)
        uniq_mcenv_df = get_stable_mofs_w_unique_mcenv(stable_df, p.activation_predictions_csv)
        uniq_mcenv_df.to_csv(p.stable_mofs_unique_mc_csv)
    elif run_activation:
        # Activation only: optionally LSE cutoff, then flag == 1; dedup by mc-RACs (one random per set, no T).
        activation_df = pd.read_csv(p.activation_predictions_csv)
        confident_df = (
            get_confident_preds_activation_only(activation_df, lse_cutoff=lse_cutoff)
            if run_confidence_checks
            else activation_df
        )
        stable_df = get_stable_mofs_activation_only(confident_df)
        mcenv_random_seed = config.get("mcenv_dedup_random_seed", 42)
        uniq_mcenv_df = get_stable_mofs_w_unique_mcenv_random(
            stable_df, p.activation_predictions_csv, random_state=mcenv_random_seed
        )
        uniq_mcenv_df.to_csv(p.stable_mofs_unique_mc_csv)
    elif run_thermal:
        # Thermal only: optionally LSD cutoff, then T > threshold; unique mcenv by highest T (thermal CSV has mc-RACs)
        thermal_df = pd.read_csv(p.thermal_predictions_csv)
        confident_df = (
            get_confident_preds_thermal_only(thermal_df, lsd_cutoff=lsd_cutoff)
            if run_confidence_checks
            else thermal_df
        )
        stable_df = get_stable_mofs_thermal_only(confident_df, T_threshold=T_threshold)
        stable_df = stable_df.copy()
        stable_df["thermal_T"] = stable_df["predicted"]  # for get_stable_mofs_w_unique_mcenv
        uniq_mcenv_df = get_stable_mofs_w_unique_mcenv(stable_df, p.thermal_predictions_csv)
        uniq_mcenv_df.to_csv(p.stable_mofs_unique_mc_csv)
    else:
        # Both stability steps bypassed: no confidence or stability filtering; only unique mc-RAC dedup (random pick).
        # Input is featurization output (metal-filtered, featurized MOFs with mc-RACs).
        featurization_df = pd.read_csv(p.merged_features_featurizable_csv)
        mcenv_random_seed = config.get("mcenv_dedup_random_seed", 42)
        uniq_mcenv_df = get_stable_mofs_w_unique_mcenv_random(
            featurization_df,
            p.merged_features_featurizable_csv,
            random_state=mcenv_random_seed,
        )
        uniq_mcenv_df.to_csv(p.stable_mofs_unique_mc_csv)

    return

if __name__ == "__main__":
    main()
