"""CatalMOF CLI implementation. Use from repo root: python cli.py run -c config.yaml (or catalmof run -c config.yaml when installed)."""

import os
import sys
import subprocess
from pathlib import Path

import click
import yaml

from catalmof.paths import get_paths

# Pipeline steps in order (module path, description)
PIPELINE_STEPS = [
    ("catalmof.screen_by_metal", "Screen by metal"),
    ("catalmof.featurization", "Featurization"),
    ("catalmof.predict_activation_stability", "Predict activation stability"),
    ("catalmof.predict_thermal_stability", "Predict thermal stability"),
    ("catalmof.get_stable_mofs", "Get stable MOFs"),
    ("catalmof.text_mining", "Text mining"),
    ("catalmof.sbu_analysis", "SBU analysis"),
    ("catalmof.sbu_capping", "SBU capping"),
]

ACTIVATION_STEP_INDEX = 2   # predict_activation_stability
THERMAL_STEP_INDEX = 3      # predict_thermal_stability
GET_STABLE_STEP_INDEX = 4   # get_stable_mofs
TEXT_MINING_STEP_INDEX = 5


def welcome():
    """Print welcome banner."""
    print("\n")
    print("      ╔════════════════════════════════════════╗   ")
    print("      ║   ____            __  __  ___  _____   ║   ")
    print("      ║  / ___| ___ _ __ |  \\/  |/ _ \\|  ___|  ║   ")
    print("      ║ | |  _ / _ \\ '_ \\| |\\/| | | | | |_     ║   ")
    print("      ║ | |_| |  __/ | | | |  | | |_| |  _|    ║   ")
    print("      ║  \\____|\\___|_| |_|_|  |_|\\___/|_|      ║   ")
    print("      ║                                        ║   ")
    print("      ║                 CatalMOF                 ║   ")
    print("      ║            [catalmof.rtfd.io]            ║   ")
    print("      ╚══════════════════╗╔════════════════════╝   ")
    print("                 ╔═══════╝╚═══════╗                 ")
    print("                 ║ THE KULIK LAB  ║                 ")
    print("                 ╚═══════╗╔═══════╝                 ")
    print("  ╔══════════════════════╝╚══════════════════════╗  ")
    print("  ║   Code: github.com/husainadamji/catalmof       ║  ")
    print("  ║   Docs: catalmof.readthedocs.io                ║  ")
    print("  ║      - Workflow: catalmof run -c config.yaml   ║  ")
    print("  ╚══════════════════════════════════════════════╝  \n")


def read_config(config_path):
    """Load and validate config YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    data.setdefault("metals", ["Mn", "Fe", "Co", "Ni", "Cu", "Ru"])
    data.setdefault("bypass_activation_stability", False)
    data.setdefault("bypass_thermal_stability", False)
    data.setdefault("run_confidence_checks", True)
    data.setdefault("bypass_text_mining", False)
    if data.get("bypass_stability_predictions") is True:
        data["bypass_activation_stability"] = True
        data["bypass_thermal_stability"] = True
    return data


def run_step(step_module, step_name, cwd, env):
    """Run a single pipeline step as a subprocess."""
    click.echo(f"  → {step_name} ({step_module})")
    result = subprocess.run(
        [sys.executable, "-m", step_module],
        cwd=cwd,
        env=env,
    )
    if result.returncode != 0:
        click.echo(f"  ✗ {step_name} failed with exit code {result.returncode}", err=True)
        raise SystemExit(result.returncode)
    click.echo(f"  ✓ {step_name} completed")
    return result


@click.group()
def cli():
    """CatalMOF CLI: run the pipeline from a config file."""
    pass


@cli.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to the configuration YAML file.",
)
@click.option(
    "--workdir", "-w",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Working directory (default: current directory). Scripts expect paths like data/ relative to this.",
)
def run(config, workdir):
    """Run the CatalMOF pipeline sequentially from config.

    Steps: 1 screen_by_metal, 2 featurization, 3 predict_activation_stability,
    4 predict_thermal_stability, 5 get_stable_mofs, 6 text_mining, 7 sbu_analysis, 8 sbu_capping.
    Featurization always runs (needed for mc-RAC dedup). When both stability steps are bypassed,
    get_stable_mofs still runs and only performs unique mc-RAC deduplication (random pick per set).

    Config (see config.example.yaml): paths (including text_mining_pickle_dir for full paper analysis,
    core_rfactors_csv for R-factor check), metals, bypass_activation_stability, bypass_thermal_stability,
    run_confidence_checks, lse_cutoff, lsd_cutoff, thermal_stability_threshold, bypass_text_mining,
    text_mining_title_only, run_rfactor_check, etc.
    """
    welcome()

    workdir = Path(workdir or os.getcwd()).resolve()
    if not workdir.is_dir():
        raise click.ClickException(f"Work directory does not exist: {workdir}")

    try:
        config_data = read_config(config)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    metals = config_data["metals"]
    bypass_activation = config_data["bypass_activation_stability"]
    bypass_thermal = config_data["bypass_thermal_stability"]
    bypass_text_mining = config_data["bypass_text_mining"]

    config_abs = Path(config).resolve()
    env = os.environ.copy()
    env["CATALMOF_CONFIG"] = str(config_abs)
    env["CATALMOF_METALS"] = ",".join(str(m) for m in metals)
    if bypass_text_mining:
        env["CATALMOF_SBU_INPUT"] = "stable_mofs_unique_mc"

    run_confidence = config_data.get("run_confidence_checks", True)
    click.echo(
        f"Config: metals={metals}, bypass_activation={bypass_activation}, "
        f"bypass_thermal={bypass_thermal}, run_confidence_checks={run_confidence}, bypass_text_mining={bypass_text_mining}"
    )
    click.echo(f"Working directory: {workdir}\n")

    for i, (step_module, step_name) in enumerate(PIPELINE_STEPS):
        if i == ACTIVATION_STEP_INDEX and bypass_activation:
            click.echo(f"  ⊘ Skipping (bypass_activation_stability): {step_name}")
            continue
        if i == THERMAL_STEP_INDEX and bypass_thermal:
            click.echo(f"  ⊘ Skipping (bypass_thermal_stability): {step_name}")
            continue
        if i == GET_STABLE_STEP_INDEX:
            # Always run get_stable_mofs: with stability it filters + unique_mcenv; when both bypassed it only does unique_mcenv (from featurization).
            run_step(step_module, step_name, cwd=str(workdir), env=env)
            continue
        if i == TEXT_MINING_STEP_INDEX:
            if bypass_text_mining:
                click.echo(f"  ⊘ Skipping (bypass_text_mining): {step_name}")
                continue
            # If full paper analysis requested and HTML dir is set, run paper pickler first so pickles exist
            title_only = config_data.get("text_mining_title_only", True)
            html_dir = (config_data.get("paths") or {}).get("text_mining_html_dir")
            if not title_only and html_dir:
                click.echo(f"  → Paper pickler (building pickles from HTMLs for full paper analysis)")
                result = subprocess.run(
                    [sys.executable, "-m", "catalmof.text_mining_tools.paper_pickler"],
                    cwd=str(workdir),
                    env=env,
                )
                if result.returncode != 0:
                    click.echo("  ✗ Paper pickler failed.", err=True)
                    raise SystemExit(result.returncode)
                click.echo(f"  ✓ Paper pickler completed")
        run_step(step_module, step_name, cwd=str(workdir), env=env)

    click.echo("\nPipeline finished successfully.")


if __name__ == "__main__":
    cli()
