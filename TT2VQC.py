#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 13:00:09 2025

@author: junqi
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchquantum as tq
import torchquantum.functional as tqf

seed = 1234
torch.manual_seed(seed)

# Scaling factor for TT core initialization
TT_INIT_SCALE = 0.01

class TensorTrainLayer(nn.Module):
    """
    Factorizes a weight tensor into multiple TT-cores.
    """
    def __init__(self, input_dims, output_dims, tt_ranks):
        """
        Args:
            input_dims (list[int]): Factorization of input dimension.
            output_dims (list[int]): Factorization of output dimension.
            tt_ranks (list[int]): TT-ranks (length = len(input_dims) + 1),
                                  with tt_ranks[0] = tt_ranks[-1] = 1.
        """
        super().__init__()
        assert len(input_dims) == len(output_dims), (
            "input_dims and output_dims must have the same number of factors."
        )
        self.d = len(input_dims)
        assert len(tt_ranks) == self.d + 1, (
            "tt_ranks must be of length d+1 (where d is # factors)."
        )
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.tt_ranks = tt_ranks

        # Build TT-cores with smaller initial weights.
        self.tt_cores = nn.ParameterList()
        for k in range(self.d):
            r_k, r_k1 = tt_ranks[k], tt_ranks[k+1]
            n_k, m_k = input_dims[k], output_dims[k]
            core = nn.Parameter(torch.randn(r_k, n_k, m_k, r_k1) * TT_INIT_SCALE)
            nn.init.xavier_uniform_(core)
            self.tt_cores.append(core)

        self.out_features = math.prod(output_dims)
        self.bias = nn.Parameter(torch.zeros(self.out_features))

    def forward(self, x):
        bsz = x.size(0)
        x_reshaped = x.view(bsz, *self.input_dims)
        # Build einsum subscript.
        batch_letter = 'b'
        letters = [chr(c) for c in range(ord('a'), ord('z')+1) if chr(c) != batch_letter]
        i_letters = letters[:self.d]
        o_letters = letters[self.d:2*self.d]
        r_letters = letters[2*self.d:2*self.d+self.d+1]

        input_subscript = batch_letter + "".join(i_letters)
        core_subscripts = [
            f"{r_letters[k]}{i_letters[k]}{o_letters[k]}{r_letters[k+1]}"
            for k in range(self.d)
        ]
        output_subscript = batch_letter + "".join(o_letters)
        einsum_str = input_subscript + "," + ",".join(core_subscripts) + "->" + output_subscript

        out = torch.einsum(einsum_str, x_reshaped, *self.tt_cores)
        out = out.reshape(bsz, -1) + self.bias
        return out

class MPS_VQC(tq.QuantumModule):
    """
    A VQC variant with an MPS entanglement pattern.
    Gate angles are provided externally.
    """
    def __init__(self, n_wires=8, n_qlayers=1, tensor_product_enc=True,
                 add_fc=False, out_features=2, noise_prob=0.0):
        super().__init__()
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.add_fc = add_fc
        self.noise_prob = noise_prob

        # Encoder: RY rotation for each qubit.
        self.encoder = (
            tq.GeneralEncoder([{'input_idx': [i], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
            if tensor_product_enc else tq.AmplitudeEncoder()
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        if add_fc:
            self.fc_layer = nn.Linear(n_wires, out_features)

    def reset_quantum_device(self, bsz: int):
        self.q_device.reset_states(bsz)

    def apply_depolarizing_noise(self):
        for i in range(self.n_wires):
            if torch.rand(1).item() < self.noise_prob:
                error_type = torch.randint(0, 3, (1,)).item()
                if error_type == 0:
                    tqf.x(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)
                elif error_type == 1:
                    tqf.y(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)
                else:
                    tqf.z(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)

    @tq.static_support
    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice, angles: torch.Tensor) -> torch.Tensor:
        self.q_device = q_device
        bsz = x.shape[0]
        self.reset_quantum_device(bsz)
        self.encoder(self.q_device, x)

        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                rx_angle = angles[:, k, i, 0]
                ry_angle = angles[:, k, i, 1]
                rz_angle = angles[:, k, i, 2]
                tqf.rx(self.q_device, wires=i, params=rx_angle,
                       static=self.static_mode, parent_graph=self.graph)
                tqf.ry(self.q_device, wires=i, params=ry_angle,
                       static=self.static_mode, parent_graph=self.graph)
                tqf.rz(self.q_device, wires=i, params=rz_angle,
                       static=self.static_mode, parent_graph=self.graph)

                if i < self.n_wires - 1:
                    tqf.cnot(self.q_device, wires=[i, i+1],
                             static=self.static_mode, parent_graph=self.graph)

            self.apply_depolarizing_noise()

        qc_out = self.measure(self.q_device)
        if self.add_fc:
            qc_out = self.fc_layer(qc_out)
        return qc_out

class MPSParamVQC(nn.Module):
    """
    High-level model that uses a Tensor-Train layer to generate VQC parameters.
    """
    def __init__(self, input_dim, tt_input_dims, tt_output_dims, tt_ranks,
                 n_wires=8, n_qlayers=1, tensor_product_enc=True, add_fc=False, out_features=2, noise_prob=0.0):
        super().__init__()
        self.total_angles = n_qlayers * n_wires * 3
        self.tt_layer = TensorTrainLayer(tt_input_dims, tt_output_dims, tt_ranks)
        self.mps_vqc = MPS_VQC(n_wires=n_wires,
                               n_qlayers=n_qlayers,
                               tensor_product_enc=tensor_product_enc,
                               add_fc=add_fc,
                               out_features=out_features,
                               noise_prob=noise_prob)
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        bsz = x.shape[0]
        assert x.shape[1] == self.input_dim, "Input dimension mismatch."
        angles_tt = self.tt_layer(x)
        angles = angles_tt.view(bsz, self.n_qlayers, self.n_wires, 3)
        out = self.mps_vqc(x, q_device=q_device, angles=angles)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    batch_size = 64
    input_dim = 2500  # classical data dimension

    # Factorize input for TT layer: here [50, 50] so that 50*50 = 2500.
    tt_input_dims = [50, 50]
    # For VQC parameters, assume n_qlayers=2, n_wires=12, 3 rotations => total 72.
    # Factorize 72 = 8 * 9 (for example).
    tt_output_dims = [8, 9]  
    tt_ranks = [1, 4, 1]

    # Create the model.
    model = MPSParamVQC(
        input_dim=input_dim,
        tt_input_dims=tt_input_dims,
        tt_output_dims=tt_output_dims,
        tt_ranks=tt_ranks,
        n_wires=12,
        n_qlayers=2,
        tensor_product_enc=False,
        add_fc=True,
        out_features=2,
        noise_prob=0.01
    ).to(device)

    # Create a TorchQuantum device.
    q_device = tq.QuantumDevice(n_wires=12, bsz=batch_size).to(device)

    X = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, 2, (batch_size,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epochs = 30
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(X, q_device)
        loss = criterion(out, y)
        loss.backward()
        # Gradient clipping to avoid exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    with torch.no_grad():
        out = model(X, q_device)
        preds = torch.argmax(out, dim=1)
        acc = (preds == y).float().mean().item()
    print("Training accuracy:", acc)

