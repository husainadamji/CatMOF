"""Hyperparameter search with hyperopt TPE + final training pass."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from hyperopt import Trials, fmin, hp, tpe
from torch.utils.data import DataLoader

from catmof.sbu_quantum_ml.models.bp_net import FullEquivariantModel
from catmof.sbu_quantum_ml.training.data import DictDataset
from catmof.sbu_quantum_ml.training.loop import train_and_validate


def build_model(
    hyperparams: Dict[str, Any],
    input_irreps: List[Any],
    device: torch.device,
    n_atom_groups: int = 10,
) -> FullEquivariantModel:
    return FullEquivariantModel(
        input_irreps=input_irreps,
        n_atom_groups=n_atom_groups,
        mlp_output_dim=int(hyperparams["mlp_output_dim"]),
        final_hidden_dim=int(hyperparams["final_hidden_dim"]),
        final_n_layers=int(hyperparams["final_n_layers"]),
        hidden_lmax=int(hyperparams["hidden_lmax"]),
        hidden_dim=int(hyperparams["hidden_dim"]),
        n_layers=int(hyperparams["n_layers"]),
        activation=hyperparams["activation"],
        dropout_rate=float(hyperparams["dropout_rate"]),
    ).to(device)


def _objective_step(
    hyperparams: Dict[str, Any],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    input_irreps: List[Any],
    device: torch.device,
    checkpoint_path: Path,
) -> float:
    model = build_model(hyperparams, input_irreps, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hyperparams["learning_rate"]))
    _, val_losses = train_and_validate(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        checkpoint_path=checkpoint_path,
    )
    return float(np.min(val_losses))


def default_hyperopt_space() -> dict:
    mlp_output_dim_choices = [1, 3, 5]
    hidden_dim_choices = [32, 64, 96, 128]
    hidden_lmax_choices = [1, 2, 3]
    n_layer_choices = [2, 3]
    activation_choices = [nn.Sigmoid(), nn.Tanh(), nn.SiLU()]
    batch_size_choices = [16, 32, 64, 256]
    return {
        "mlp_output_dim": hp.choice("mlp_output_dim", mlp_output_dim_choices),
        "final_hidden_dim": hp.choice("final_hidden_dim", hidden_dim_choices),
        "final_n_layers": hp.choice("final_n_layers", n_layer_choices),
        "hidden_lmax": hp.choice("hidden_lmax", hidden_lmax_choices),
        "hidden_dim": hp.choice("hidden_dim", hidden_dim_choices),
        "n_layers": hp.choice("n_layers", n_layer_choices),
        "activation": hp.choice("activation", activation_choices),
        "dropout_rate": hp.uniform("dropout_rate", 0, 0.5),
        "learning_rate": hp.loguniform("learning_rate", -11.5, -6.9),
        "batch_size": hp.choice("batch_size", batch_size_choices),
        "_choices": {
            "mlp_output_dim": mlp_output_dim_choices,
            "final_hidden_dim": hidden_dim_choices,
            "final_n_layers": n_layer_choices,
            "hidden_lmax": hidden_lmax_choices,
            "hidden_dim": hidden_dim_choices,
            "n_layers": n_layer_choices,
            "activation": activation_choices,
            "batch_size": batch_size_choices,
        },
    }


def decode_hyperopt_sample(raw: Dict[str, Any], space: dict) -> Dict[str, Any]:
    """Map hyperopt `choice` indices to concrete values (same encoding during search and after `fmin`)."""
    ch = space["_choices"]
    out = dict(raw)
    out["mlp_output_dim"] = ch["mlp_output_dim"][int(raw["mlp_output_dim"])]
    out["final_hidden_dim"] = ch["final_hidden_dim"][int(raw["final_hidden_dim"])]
    out["final_n_layers"] = ch["final_n_layers"][int(raw["final_n_layers"])]
    out["hidden_lmax"] = ch["hidden_lmax"][int(raw["hidden_lmax"])]
    out["hidden_dim"] = ch["hidden_dim"][int(raw["hidden_dim"])]
    out["n_layers"] = ch["n_layers"][int(raw["n_layers"])]
    out["activation"] = ch["activation"][int(raw["activation"])]
    out["batch_size"] = ch["batch_size"][int(raw["batch_size"])]
    out["learning_rate"] = float(raw["learning_rate"])
    out["dropout_rate"] = float(raw["dropout_rate"])
    return out


def optimize_model(
    train_inputs: dict,
    train_y: torch.Tensor,
    val_inputs: dict,
    val_y: torch.Tensor,
    input_irreps: List[Any],
    device: torch.device,
    output_dir: Path,
    max_evals: int = 100,
    random_seed: int = 0,
) -> Tuple[FullEquivariantModel, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "hyperopt_trials.pkl"
    best_hp_path = output_dir / "best_hyperparameters.pkl"
    trial_ckpt = output_dir / "trial_best_model.pth"
    final_ckpt = output_dir / "best_model.pth"

    def create_dataloader(inputs: dict, targets: torch.Tensor, batch_size: int) -> DataLoader:
        dataset = DictDataset(inputs, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    space = default_hyperopt_space()

    if trials_path.is_file():
        with open(trials_path, "rb") as f:
            trials = pickle.load(f)
    else:
        trials = Trials()

    def wrapped_objective(hp_sample: dict) -> float:
        decoded = decode_hyperopt_sample(hp_sample, space)
        batch_size = int(decoded["batch_size"])
        train_dl = create_dataloader(train_inputs, train_y, batch_size)
        val_dl = create_dataloader(val_inputs, val_y, batch_size)
        loss = _objective_step(
            decoded, train_dl, val_dl, input_irreps, device, checkpoint_path=trial_ckpt
        )
        with open(trials_path, "wb") as f:
            pickle.dump(trials, f)
        return loss

    hp_for_fmin = {k: v for k, v in space.items() if k != "_choices"}

    print("Performing hyperparameter search...")
    best = fmin(
        fn=wrapped_objective,
        space=hp_for_fmin,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.RandomState(random_seed),
    )

    best_hyperparams = decode_hyperopt_sample(dict(best), space)
    with open(best_hp_path, "wb") as f:
        pickle.dump(best_hyperparams, f)

    print("Training final model with best hyperparameters...")
    best_bs = int(best_hyperparams["batch_size"])
    train_dl = create_dataloader(train_inputs, train_y, best_bs)
    val_dl = create_dataloader(val_inputs, val_y, best_bs)

    best_model = build_model(best_hyperparams, input_irreps, device)
    optimizer = torch.optim.Adam(
        best_model.parameters(), lr=float(best_hyperparams["learning_rate"])
    )
    train_losses, val_losses = train_and_validate(
        best_model,
        train_dl,
        val_dl,
        optimizer,
        device,
        checkpoint_path=final_ckpt,
    )

    fig, ax = plt.subplots()
    ax.plot(train_losses, color="red", linestyle="-", label="training loss")
    ax.plot(val_losses, color="blue", linestyle="-", label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (scaled target)")
    ax.set_title("Train and validation loss")
    ax.legend()
    fig.savefig(output_dir / "loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(trials_path, "wb") as f:
        pickle.dump(trials, f)

    return best_model, best_hyperparams
