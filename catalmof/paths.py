"""
Central path and filename definitions for the CatalMOF pipeline.
Paths are built from a config file (when CATALMOF_CONFIG is set) or defaults,
so bypassed steps and custom layouts stay consistent.
"""

import os
from types import SimpleNamespace

try:
    import yaml
except ImportError:
    yaml = None

# Canonical short filenames (no scattered long names)
FILES = SimpleNamespace(
    cifs_with_metals="cifs_with_metals.csv",
    metal_filtered_cifs="metal_filtered_cifs.csv",
    merged_features="merged_features.csv",
    merged_features_featurizable="merged_features_featurizable_mofs.csv",
    activation_predictions="activation_predictions.csv",
    thermal_predictions="thermal_predictions.csv",
    stable_mofs_unique_mc="stable_mofs_unique_mc.csv",
    stable_uniq_no_catalysis="stable_uniq_no_catalysis.csv",
    manuscript_data="CoRE_manuscript_data_w_titles.csv",
    position_in_paper="position_in_paper_distribution.csv",
    text_mining_results="stable_uniq_mofs_text_mining_results.csv",
)

DEFAULT_BASE_DIR = "data"
DEFAULT_CORE_CIFS_DIR = "data/CoRE_ASR_2024"
DEFAULT_CORE_RFACTORS = "data/CoRE_ASR_2024_Rfactors.csv"
DEFAULT_STABILITY_MODELS = "stability_ml_models"
DEFAULT_ZEO_NETWORK = "submodules/zeo++-0.3/network"


def _load_config_paths():
    """Load paths section from config file pointed by CATALMOF_CONFIG."""
    cfg = get_config()
    return cfg.get("paths", {})


def get_config():
    """Load full config from file pointed by CATALMOF_CONFIG. Returns {} if not set or on error."""
    config_path = os.environ.get("CATALMOF_CONFIG")
    if not config_path or not os.path.isfile(config_path) or yaml is None:
        return {}
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception:
        return {}


def get_paths():
    """
    Return a namespace of all pipeline paths.
    Uses CATALMOF_CONFIG if set, otherwise defaults (backward compatible).
    """
    cfg = _load_config_paths()
    base = cfg.get("base_dir", DEFAULT_BASE_DIR)
    base = base.rstrip("/")

    # Directories under base
    featurization_dir = cfg.get("featurization_dir") or f"{base}/featurization"
    mof_stability_dir = cfg.get("mof_stability_dir") or f"{base}/mof_stability"
    text_mining_dir = cfg.get("text_mining_dir") or f"{base}/text_mining"
    sbu_clusters_dir = cfg.get("sbu_clusters_dir") or f"{base}/sbu_clusters"

    # External / cross-cutting (can live outside base)
    core_cifs_dir = cfg.get("core_cifs_dir", DEFAULT_CORE_CIFS_DIR)
    core_rfactors_csv = cfg.get("core_rfactors_csv", DEFAULT_CORE_RFACTORS)
    stability_models_dir = cfg.get("stability_models_dir", DEFAULT_STABILITY_MODELS)
    zeo_network = cfg.get("zeo_network", DEFAULT_ZEO_NETWORK)
    manuscript_data_csv = cfg.get("manuscript_data_csv") or f"{text_mining_dir}/{FILES.manuscript_data}"
    text_mining_html_dir = cfg.get("text_mining_html_dir")  # User's HTML/XML corpus (for paper_pickler)
    text_mining_pickle_dir = cfg.get("text_mining_pickle_dir")  # None if not provided (forces title-only)

    # Key file paths
    metal_filtered_cifs_csv = f"{base}/{FILES.metal_filtered_cifs}"
    cifs_with_metals_csv = f"{base}/{FILES.cifs_with_metals}"
    merged_features_featurizable_csv = f"{featurization_dir}/{FILES.merged_features_featurizable}"
    merged_features_in_stability_dir = f"{mof_stability_dir}/{FILES.merged_features_featurizable}"
    activation_predictions_csv = f"{mof_stability_dir}/{FILES.activation_predictions}"
    thermal_predictions_csv = f"{mof_stability_dir}/{FILES.thermal_predictions}"
    stable_mofs_unique_mc_csv = f"{mof_stability_dir}/{FILES.stable_mofs_unique_mc}"
    stable_uniq_no_catalysis_csv = f"{text_mining_dir}/{FILES.stable_uniq_no_catalysis}"
    position_in_paper_csv = f"{text_mining_dir}/{FILES.position_in_paper}"
    text_mining_results_csv = f"{text_mining_dir}/{FILES.text_mining_results}"

    return SimpleNamespace(
        base_dir=base,
        # Dirs
        featurization_dir=featurization_dir,
        mof_stability_dir=mof_stability_dir,
        text_mining_dir=text_mining_dir,
        sbu_clusters_dir=sbu_clusters_dir,
        core_cifs_dir=core_cifs_dir,
        stability_models_dir=stability_models_dir,
        zeo_network=zeo_network,
        # CSVs
        cifs_with_metals_csv=cifs_with_metals_csv,
        metal_filtered_cifs_csv=metal_filtered_cifs_csv,
        merged_features_featurizable_csv=merged_features_featurizable_csv,
        merged_features_in_stability_dir=merged_features_in_stability_dir,
        activation_predictions_csv=activation_predictions_csv,
        thermal_predictions_csv=thermal_predictions_csv,
        stable_mofs_unique_mc_csv=stable_mofs_unique_mc_csv,
        stable_uniq_no_catalysis_csv=stable_uniq_no_catalysis_csv,
        manuscript_data_csv=manuscript_data_csv,
        position_in_paper_csv=position_in_paper_csv,
        text_mining_results_csv=text_mining_results_csv,
        core_rfactors_csv=core_rfactors_csv,
        # For featurization subdirs
        featurization_zeo_data=f"{featurization_dir}/zeo_data",
        # Text mining: HTML corpus (for pickler) and pickle directory (None if not provided)
        text_mining_html_dir=text_mining_html_dir,
        text_mining_pickle_dir=text_mining_pickle_dir,
    )


def get_sbu_input_csv():
    """
    Path to the MOF list used as input for SBU analysis.
    When text mining is bypassed, use stable_mofs_unique_mc instead of stable_uniq_no_catalysis.
    """
    p = get_paths()
    if os.environ.get("CATALMOF_SBU_INPUT") == "stable_mofs_unique_mc":
        return p.stable_mofs_unique_mc_csv
    return p.stable_uniq_no_catalysis_csv
