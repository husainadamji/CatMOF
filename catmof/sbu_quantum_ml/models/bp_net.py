"""Equivariant Behler–Parinello-style network built with e3nn."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from e3nn.nn import Activation, Dropout, Gate
from e3nn.o3 import Irreps, Linear


class EquivariantMLP(nn.Module):
    """Per-class MLP: irreps-preserving stack with gating, then scalar readout."""

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
                        [f"{hidden_dim}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(1, hidden_lmax + 1)]
                    ),
                )
            )
            inp = gate_irreps

        self.dropout = Dropout(gate_irreps, dropout_rate)
        self.output_layer = Linear(gate_irreps, Irreps(f"{output_dim}x0e"))

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Each tensor is (n_atoms, D) for one batch element; empty list elements sum to zero."""
        outputs: List[torch.Tensor] = []
        for batch_inputs in inputs:
            if batch_inputs.numel() == 0:
                outputs.append(
                    torch.zeros((1, self.output_layer.irreps_out.dim), device=batch_inputs.device)
                )
                continue

            x = batch_inputs
            self.dropout.to(batch_inputs.device)
            for i in range(self.n_layers):
                self.hidden_layers[i].to(batch_inputs.device)
                self.activations[i].to(batch_inputs.device)
                x = self.hidden_layers[i](x)
                x = self.activations[i](x)
                x = self.dropout(x)
            self.output_layer.to(batch_inputs.device)
            outputs.append(self.output_layer(x))

        return outputs


class TiledEquivariantMLP(nn.Module):
    """One :class:`EquivariantMLP` per atom class; outputs are concatenated after per-class pooling."""

    def __init__(
        self,
        input_irreps_list: List[Irreps],
        output_dim: int,
        n_tiles: int,
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
            ``mlp_mapping`` — (B, n_atoms, 1) class id per atom (10 = ghost pad)
            ``feature_vec_length`` — (B, n_atoms, 1) true D per atom before padding
        """
        atom_features = inputs["atom_features"]
        mlp_mapping = torch.squeeze(inputs["mlp_mapping"], dim=-1)
        feature_vec_length = torch.squeeze(inputs["feature_vec_length"], dim=-1)

        outputs: List[torch.Tensor] = []
        for i, mlp in enumerate(self.mlps):
            class_features: List[torch.Tensor] = []
            for batch_idx in range(atom_features.shape[0]):
                batch_mlp_mapping = mlp_mapping[batch_idx]
                condition = batch_mlp_mapping == i
                if not torch.any(condition):
                    class_features.append(torch.empty(0, device=atom_features.device))
                    continue
                batch_feature_vec_length = feature_vec_length[batch_idx]
                batch_atom_features = atom_features[batch_idx]
                batch_feature_length = batch_feature_vec_length[condition][0]
                batch_class_features = batch_atom_features[condition][:, : int(batch_feature_length.item())]
                class_features.append(batch_class_features)

            group_output = mlp(class_features) # List of tensors of shape: [n_atoms, output_dim] for each batch item
            group_sum = [torch.sum(batch_tensor, dim=0) for batch_tensor in group_output] # List of tensors of shape: [output_dim] for each batch item
            group_sum_tensor = torch.stack(group_sum, dim=0) # Tensor of shape: [n_batches, output_dim]
            outputs.append(group_sum_tensor)

        return torch.cat(outputs, dim=1) # Shape: [n_batches, n_tiles * output_dim]


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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        self.dropout.to(inputs.device)
        for i in range(self.n_layers):
            self.hidden_layers[i].to(inputs.device)
            self.activations[i].to(inputs.device)
            x = self.hidden_layers[i](x)
            x = self.activations[i](x)
            x = self.dropout(x)
        self.output_layer.to(inputs.device)
        return self.output_layer(x)


class FullEquivariantModel(nn.Module):
    """Behler–Parinello-style model: tiled equivariant atom-class MLPs + final scalar head."""

    def __init__(
        self,
        input_irreps: List[Irreps],
        n_atom_groups: int,
        mlp_output_dim: int,
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
        concat_outputs = self.tiled_mlp(inputs) # Shape: [n_batches, n_tiles * output_dim]
        return self.final_mlp(concat_outputs)
