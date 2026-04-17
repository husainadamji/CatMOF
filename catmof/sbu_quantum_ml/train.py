"""
Train equivariant BPNN with hyperopt (TPE).

Example (from a directory that contains ``density_fitting_features.pkl`` and the task targets CSV)::

    python -m catmof.sbu_quantum_ml.train --task oxo --workdir .

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from catmof.sbu_quantum_ml.tasks import get_task
from catmof.sbu_quantum_ml.training.data import load_or_create_splits, normalize_features_labels, prepare_data
from catmof.sbu_quantum_ml.training.hyperopt import optimize_model
from catmof.sbu_quantum_ml.training.metrics import evaluate_regression


def _predict_all(model: torch.nn.Module, inputs: dict, device: torch.device) -> np.ndarray:
    model.eval()
    batch_inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        out = model(batch_inputs).cpu().numpy().reshape(-1, 1)
    return out


def run_training(
    workdir: Path,
    task_name: str,
    max_evals: int,
    force_refresh: bool,
    features_pickle: Path,
) -> None:
    task = get_task(task_name)
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    features_pkl = features_pickle if features_pickle.is_absolute() else (workdir / features_pickle)
    targets_csv = workdir / task.targets_csv
    splits_cache = workdir / f"data_splits_and_irreps_{task.name}.pkl"

    bundle = load_or_create_splits(
        task=task,
        features_pickle=features_pkl,
        targets_csv=targets_csv,
        splits_cache=splits_cache,
        force_refresh=force_refresh,
    )
    (
        sbus_train,
        sbus_val,
        sbus_test,
        atomic_train,
        atomic_val,
        atomic_test,
        _full_tr,
        _full_va,
        _full_te,
        y_train,
        y_val,
        y_test,
        input_irreps,
        max_atoms,
    ) = bundle

    (
        norm_atomic_tr,
        norm_atomic_va,
        norm_atomic_te,
        _norm_full_tr,
        _norm_full_va,
        _norm_full_te,
        y_tr_s,
        y_va_s,
        y_te_s,
        y_scaler,
    ) = normalize_features_labels(
        sbus_train,
        atomic_train,
        atomic_val,
        atomic_test,
        y_train,
        y_val,
        y_test,
        full_sbu_features_train=None,
        full_sbu_features_val=None,
        full_sbu_features_test=None,
    )

    train_inputs, train_y = prepare_data(norm_atomic_tr, y_tr_s, max_atoms=max_atoms)
    val_inputs, val_y = prepare_data(norm_atomic_va, y_va_s, max_atoms=max_atoms)
    test_inputs, _test_y_scaled = prepare_data(norm_atomic_te, y_te_s, max_atoms=max_atoms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    best_model, _best_hp = optimize_model(
        train_inputs,
        train_y,
        val_inputs,
        val_y,
        input_irreps,
        device,
        output_dir=workdir,
        max_evals=max_evals,
    )

    print("Making predictions...")
    pred_train = y_scaler.inverse_transform(_predict_all(best_model, train_inputs, device))
    pred_val = y_scaler.inverse_transform(_predict_all(best_model, val_inputs, device))
    pred_test = y_scaler.inverse_transform(_predict_all(best_model, test_inputs, device))

    y_train_2 = y_train.reshape(-1, 1)
    y_val_2 = y_val.reshape(-1, 1)
    y_test_2 = y_test.reshape(-1, 1)

    train_m = evaluate_regression(y_train_2, pred_train)
    val_m = evaluate_regression(y_val_2, pred_val)
    test_m = evaluate_regression(y_test_2, pred_test)

    perf_path = workdir / f"{task.name}_model_performance_bpnn.txt"
    with open(perf_path, "w") as f:
        f.write(f"Training data MAE = {train_m[0]} kcal/mol\n")
        f.write(f"Training data R^2 = {train_m[1]}\n")
        f.write(f"Training data MAPE = {train_m[2]} %\n")
        f.write(f"Training data RMSE = {train_m[3]} kcal/mol\n")
        f.write(f"Validation data MAE = {val_m[0]} kcal/mol\n")
        f.write(f"Validation data R^2 = {val_m[1]}\n")
        f.write(f"Validation data MAPE = {val_m[2]} %\n")
        f.write(f"Validation data RMSE = {val_m[3]} kcal/mol\n")
        f.write(f"Test data MAE = {test_m[0]} kcal/mol\n")
        f.write(f"Test data R^2 = {test_m[1]}\n")
        f.write(f"Test data MAPE = {test_m[2]} %\n")
        f.write(f"Test data RMSE = {test_m[3]} kcal/mol\n")

    fig, ax = plt.subplots()
    ax.scatter(y_train_2.flatten(), pred_train.flatten(), linestyle="None", marker="o", alpha=0.5, label="train", color="red")
    ax.scatter(y_val_2.flatten(), pred_val.flatten(), linestyle="None", marker="o", alpha=0.5, label="validation", color="blue")
    ax.scatter(y_test_2.flatten(), pred_test.flatten(), linestyle="None", marker="o", alpha=0.5, label="test", color="green")
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "0.5")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend()
    ax.set_xlabel("DFT (kcal/mol)")
    ax.set_ylabel("ML prediction (kcal/mol)")
    ax.set_title(f"{task.name} (kcal/mol)")
    fig.savefig(workdir / f"{task.name}_parity_bpnn.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"y_train": y_train_2.flatten(), "pred_train": pred_train.flatten()}).to_csv(
        workdir / "train_true_and_pred.csv", index=False
    )
    pd.DataFrame({"y_val": y_val_2.flatten(), "pred_val": pred_val.flatten()}).to_csv(
        workdir / "val_true_and_pred.csv", index=False
    )
    pd.DataFrame({"y_test": y_test_2.flatten(), "pred_test": pred_test.flatten()}).to_csv(
        workdir / "test_true_and_pred.csv", index=False
    )

    print(f"Wrote metrics to {perf_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Hyperopt training for SBU equivariant BPNN.")
    p.add_argument("--task", choices=("oxo", "hat"), default="oxo", help="Target column preset (see tasks.py).")
    p.add_argument(
        "--workdir",
        type=Path,
        default=Path("."),
        help="Directory with features pickle, targets CSV, and output artifacts.",
    )
    p.add_argument(
        "--features-pickle",
        type=Path,
        default=Path("density_fitting_features.pkl"),
        help="Feature bundle pickle (SBU list, class_based_atomic_coeffs, irreps, xTB block). "
        "Relative paths are resolved under --workdir.",
    )
    p.add_argument("--max-evals", type=int, default=100, help="Hyperopt TPE evaluations.")
    p.add_argument(
        "--refresh-splits",
        action="store_true",
        help="Ignore cached data_splits_and_irreps_<task>.pkl and rebuild from raw pickles/CSV.",
    )
    args = p.parse_args()
    run_training(
        args.workdir,
        args.task,
        args.max_evals,
        force_refresh=args.refresh_splits,
        features_pickle=args.features_pickle,
    )


if __name__ == "__main__":
    main()
