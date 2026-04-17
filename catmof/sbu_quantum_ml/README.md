# SBU quantum ML (equivariant Behler–Parinello network)

Train a regression model on **atom-centered density-fitting coefficients** (and bundled xTB scalars) per SBU, with optional hyperparameter search (TPE via hyperopt).

`catmof.sbu_quantum_ml` is part of the same **`catmof`** package as the rest of the library, so a normal CatMOF install includes this code. The training CLI still expects **PyTorch, e3nn, hyperopt, and matplotlib** to be available in the environment (they are optional extras in project metadata, not in CatMOF’s minimal dependency list).

## What you must provide

All paths below are relative to **`--workdir`** unless you pass an absolute `--features-pickle`.

### 1. Feature bundle (pickle)

Default filename: **`density_fitting_features.pkl`** (override with `--features-pickle`).

The loader expects a `pickle` dict with at least:

| Key | Description |
|-----|-------------|
| `sbu` | List/array of SBU id strings (must match the targets CSV). |
| `class_based_atomic_coeffs` | Per-SBU list of **10** tiles; each tile is a NumPy array `(n_atoms_in_class, n_features)` or empty `shape (0, *)`. |
| `class_based_atomic_irreps` | Per-SBU list of irreps metadata per class (used to build e3nn `Irreps` per atom class). |
| `Final energy (Eh)` | Length-`N` array (one value per SBU). |
| `HOMO-LUMO gap (eV)` | Same. |
| `Dipole moment (Debye)` | Same. |
| `Vertical IP (eV)` | Same. |
| `Vertical EA (eV)` | Same. |
| `Global Electrophilicity Index (eV)` | Same. |

The six xTB columns are **read and stacked** when building splits; the current training entrypoint does **not** feed them into the model (only atom coefficients are used in `prepare_data` for training). They remain part of the pickle contract for `load_or_create_splits`.

### 2. Targets CSV

Filename and label column depend on **task** (see `tasks.py`). Defaults:

| `--task` | Targets file (under `--workdir`) | SBU column | Target column |
|----------|-----------------------------------|------------|----------------|
| `oxo` | `targets_outliers_removed.csv` | `sbu_name` | `DEoxo (kcal/mol)` |
| `hat` | `targets_outliers_removed.csv` | `sbu_name` | `DEHAT (kcal/mol)` |

Rows with missing / non-numeric targets are dropped. SBUs must appear in both the pickle and the CSV (inner join on `sbu_name`).

To use different filenames or columns, edit **`catmof/sbu_quantum_ml/tasks.py`** or extend the task registry.

## Usage

```bash
# Oxo task (default)
python -m catmof.sbu_quantum_ml.train --workdir /path/to/run --task oxo

# HAT task
python -m catmof.sbu_quantum_ml.train --workdir /path/to/run --task hat

# Custom feature pickle name (relative to workdir)
python -m catmof.sbu_quantum_ml.train --workdir /path/to/run --features-pickle my_features.pkl

# Rebuild train/val/test split from scratch (ignore cached splits)
python -m catmof.sbu_quantum_ml.train --workdir /path/to/run --refresh-splits

# Fewer hyperopt trials
python -m catmof.sbu_quantum_ml.train --workdir /path/to/run --max-evals 25
```

### CLI summary

| Option | Default | Meaning |
|--------|---------|---------|
| `--workdir` | `.` | Directory containing inputs and outputs. |
| `--task` | `oxo` | `oxo` or `hat` (presets in `tasks.py`). |
| `--features-pickle` | `density_fitting_features.pkl` | Feature bundle; relative paths join `--workdir`. |
| `--max-evals` | `100` | Hyperopt TPE evaluations. |
| `--refresh-splits` | off | Delete/regenerate `data_splits_and_irreps_<task>.pkl` logic by rebuilding from pickle + CSV. |

## Outputs (written under `--workdir`)

- `data_splits_and_irreps_<task>.pkl` — Cached split + irreps (skip with existing cache unless `--refresh-splits`).
- `hyperopt_trials.pkl`, `best_hyperparameters.pkl` — Search state and best decoded hyperparameters.
- `best_model.pth` — Weights after final training with best hyperparameters.
- `loss_curves.png` — Train / validation loss.
- `<task>_parity_bpnn.png` — Parity plot.
- `<task>_model_performance_bpnn.txt` — MAE, R², MAPE, RMSE on train/val/test.
- `train_true_and_pred.csv`, `val_true_and_pred.csv`, `test_true_and_pred.csv`.

## Layout in the package

- `models/bp_net.py` — e3nn equivariant BP-style model.
- `training/data.py` — Pickle IO, alignment to targets, scaling, batching.
- `training/hyperopt.py` — TPE search and decoding of `hp.choice` indices.
- `training/loop.py` — Training loop with early stopping.
- `train.py` — CLI entry point.
- `density_fitting/` — Molden parsing, Psi4 def2-universal-jfkit auxiliary density fitting, xTB log parsing, and assembly of `density_fitting_features.pkl` (see below).

## Building `density_fitting_features.pkl` (xTB → Psi4 → DF)

External programs: **xTB**, **Psi4** (conda install recommended), and **PyTorch + e3nn** for DF tensor code. Molden `[Atoms]` uses **`numbers`** from **`molSimplify.Classes.globalvars.amassdict`** (element symbol, first column) and **`pseudo_numbers`** from the third column

**Python API (import from `catmof.sbu_quantum_ml.density_fitting.workflows`):**

- `symbols_from_xyz` — element symbols in XYZ order.
- `psi4_rohf_sto3g_wfn` — minimal ROHF/STO-3G reference wfn (`.npy`) from an xTB optimized geometry.
- `psi4_from_xtb_molden_mos` — replace alpha MO coefficients using an xTB (or compatible) Molden file that corresponds to a closed shell system.
- `run_df_e3nn` — one SBU → pickle with `sbu`, `atom_list`, `atom_irreps`, `coefficients`. The `orbital_basis` argument is the Psi4 **orbital** basis keyword used to build matching **JKFIT** auxiliaries (default **`def2-universal-jfkit`**).
- `charge_spin_lookup` — `{sbu: (charge, spin_mult)}` from a CSV.

Cluster job scripts and directory layout are **not** generated by the library.

**Other modules (same package):**

- `assemble_features.merge_per_sbu_pickles` → bundled pickle with parallel lists.
- `assemble_features.assemble_class_tiled_features` — add xTB scalar columns and emit the **training** bundle (`sbu`, `class_based_atomic_*`, `max_atoms_all_sbus`, six xTB keys). `load_or_create_splits` / `prepare_data` use `max_atoms_all_sbus` for padding (fallback 180 if missing).
- `xtb_parse.parse_xtb_root_to_dataframe` — table of energies/properties from finished xTB logs (one row per child directory of the run root).

**CLI:**

```bash
python -m catmof.sbu_quantum_ml.density_fitting xtb-parse-root --root ./xtb --out-csv xtb_output_data.csv
python -m catmof.sbu_quantum_ml.density_fitting merge-df-pickles --inputs sbu1.pkl sbu2.pkl --out all_coeffs.pkl
python -m catmof.sbu_quantum_ml.density_fitting assemble-features --coeff-pkl all_coeffs.pkl --xtb-csv xtb_output_data.csv --out density_fitting_features.pkl
```

Charge/spin CSV for Psi4/DF jobs is expected to have columns **`sbu`**, **`charge`**, **`spin multiplicity`** (see `workflows.charge_spin_lookup`).

## Data hosting

The CatMOF repository does not ship `density_fitting_features.pkl` or targets CSVs. Check Zenodo repository from corresponding manuscript.
