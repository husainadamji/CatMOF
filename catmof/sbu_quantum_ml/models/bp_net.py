"""Equivariant Behler–Parinello-style network built with e3nn."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from e3nn.nn import Activation, Dropout, Gate
from e3nn.o3 import Irreps, Linear


class EquivariantMLP(nn.Module):
    """Per-class equivariant MLP: linear → gate → dropout per layer, then scalar readout and scatter-sum."""

    def __init__(
        self,
        input_irreps: Irreps,
        output_dim: int,
        hidden_lmax: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 2,
        activation: nn.Module | None = None,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.SiLU()

        self.n_layers = n_layers
        hidden_reps_scalar_mult = hidden_dim * (hidden_lmax + 1)
        linear_irreps = Irreps(
            "+".join(
                [
                    f"{hidden_dim if l > 0 else hidden_reps_scalar_mult}x{l}{'e' if l % 2 == 0 else 'o'}"
                    for l in range(hidden_lmax + 1)
                ]
            )
        )
        gate_irreps = Irreps(
            "+".join(
                [f"{hidden_dim}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(hidden_lmax + 1)]
            )
        )

        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        inp = input_irreps
        for _ in range(n_layers):
            self.hidden_layers.append(Linear(inp, linear_irreps))
            self.activations.append(
                Gate(
                    irreps_scalars=f"{hidden_dim}x0e",
                    act_scalars=[activation],
                    irreps_gates=f"{hidden_dim * hidden_lmax}x0e",
                    act_gates=[activation],
                    irreps_gated="+".join(
                        [
                            f"{hidden_dim}x{l}{'e' if l % 2 == 0 else 'o'}"
                            for l in range(1, hidden_lmax + 1)
                        ]
                    ),
                )
            )
            inp = gate_irreps

        self.dropout = Dropout(gate_irreps, dropout_rate)
        self.output_dim = output_dim
        self.output_layer = Linear(gate_irreps, Irreps(f"{output_dim}x0e"))

    def forward(self, x: torch.Tensor, batch_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (total_atoms, D) features for this class across the minibatch.
        batch_indices
            (total_atoms,) batch index per row (``long``).
        batch_size
            Minibatch size B; output rows are summed per batch index.
        """
        if x.shape[0] == 0:
            return torch.zeros((batch_size, self.output_dim), device=x.device, dtype=x.dtype)

        for i in range(self.n_layers):
            x = self.hidden_layers[i](x)
            x = self.activations[i](x)
            x = self.dropout(x)

        x = self.output_layer(x)
        result = torch.zeros((batch_size, x.shape[1]), device=x.device, dtype=x.dtype)
        result.scatter_add_(0, batch_indices.unsqueeze(1).expand_as(x), x)
        return result


class TiledEquivariantMLP(nn.Module):
    """One :class:`EquivariantMLP` per atom class; batched across the minibatch, then concatenated."""

    def __init__(
        self,
        input_irreps_list: List[Irreps],
        output_dim: int,
        n_tiles: int,
        feature_vec_lengths: Dict[int, int],
        hidden_lmax: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 2,
        activation: nn.Module | None = None,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if len(input_irreps_list) != n_tiles:
            raise ValueError("n_tiles must match len(input_irreps_list)")
        if activation is None:
            activation = nn.SiLU()

        self.feature_vec_lengths = dict(feature_vec_lengths)
        self.mlps = nn.ModuleList(
            [
                EquivariantMLP(
                    input_irreps=input_irreps_list[i],
                    output_dim=output_dim,
                    hidden_lmax=hidden_lmax,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    activation=activation,
                    dropout_rate=dropout_rate,
                )
                for i in range(n_tiles)
            ]
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : dict
            ``atom_features`` — (B, n_atoms, D_pad)
            ``mlp_mapping`` — (B, n_atoms, 1) class id per atom (10 = ghost / padding)
        """
        atom_features = inputs["atom_features"]
        mlp_mapping = torch.squeeze(inputs["mlp_mapping"], dim=-1)
        batch_size = atom_features.shape[0]

        outputs: List[torch.Tensor] = []
        for i, mlp in enumerate(self.mlps):
            mask = mlp_mapping == i
            if not mask.any():
                outputs.append(
                    torch.zeros((batch_size, mlp.output_dim), device=atom_features.device, dtype=atom_features.dtype)
                )
                continue

            batch_idx = torch.arange(batch_size, device=atom_features.device, dtype=torch.long)
            batch_idx = batch_idx.unsqueeze(1).expand_as(mask)
            batch_indices = batch_idx[mask]

            feat_len = int(self.feature_vec_lengths[i])
            class_features = atom_features[mask][:, :feat_len]
            group_output = mlp(class_features, batch_indices, batch_size)
            outputs.append(group_output)

        return torch.cat(outputs, dim=1)


class FinalEquivariantMLP(nn.Module):
    """Scalar MLP on concatenated per-class summaries → single scalar prediction."""

    def __init__(
        self,
        input_irreps: Irreps,
        hidden_dim: int = 64,
        n_layers: int = 2,
        activation: nn.Module | None = None,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.SiLU()

        self.n_layers = n_layers
        hidden_irreps = Irreps(f"{hidden_dim}x0e")

        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        inp = input_irreps
        for _ in range(n_layers):
            self.hidden_layers.append(Linear(inp, hidden_irreps))
            self.activations.append(Activation(hidden_irreps, [activation]))
            inp = hidden_irreps

        self.dropout = Dropout(hidden_irreps, dropout_rate)
        self.output_layer = Linear(hidden_irreps, Irreps("1x0e"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_layers):
            x = self.hidden_layers[i](x)
            x = self.activations[i](x)
            x = self.dropout(x)
        return self.output_layer(x)


class FullEquivariantModel(nn.Module):
    """Behler–Parinello-style model: tiled equivariant atom-class MLPs + final scalar head."""

    def __init__(
        self,
        input_irreps: List[Irreps],
        n_atom_groups: int,
        mlp_output_dim: int,
        feature_vec_lengths: Dict[int, int],
        final_hidden_dim: int = 64,
        final_n_layers: int = 2,
        hidden_lmax: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 2,
        activation: nn.Module | None = None,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.SiLU()

        self.tiled_mlp = TiledEquivariantMLP(
            input_irreps_list=input_irreps,
            output_dim=mlp_output_dim,
            n_tiles=n_atom_groups,
            feature_vec_lengths=feature_vec_lengths,
            hidden_lmax=hidden_lmax,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        final_input_irreps = Irreps(f"{mlp_output_dim * n_atom_groups}x0e")
        self.final_mlp = FinalEquivariantMLP(
            input_irreps=final_input_irreps,
            hidden_dim=final_hidden_dim,
            n_layers=final_n_layers,
            activation=activation,
            dropout_rate=dropout_rate,
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        return self.final_mlp(self.tiled_mlp(inputs))
