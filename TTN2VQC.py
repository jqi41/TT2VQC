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

#############################################
# 1. Tensor Tree Encoder (TTN)
#############################################

seed = 1234
torch.manual_seed(seed)

class TensorTreeEncoder(nn.Module):
    def __init__(self, n_leaves, leaf_dim, tree_ranks):
        """
        Args:
            n_leaves (int): Must be a power of 2.
            leaf_dim (int): Dimension per leaf.
            tree_ranks (list of ints): Length = log2(n_leaves).
                - Level 0 cores: (leaf_dim, leaf_dim, tree_ranks[0])
                - Level l>=1 cores: (tree_ranks[l-1], tree_ranks[l-1], tree_ranks[l])
        """
        super().__init__()
        assert (n_leaves & (n_leaves - 1)) == 0, "n_leaves must be a power of 2."
        self.n_leaves = n_leaves
        self.leaf_dim = leaf_dim
        self.L = int(math.log2(n_leaves))
        assert len(tree_ranks) == self.L, f"tree_ranks must have length = log2(n_leaves)={self.L}"
        self.tree_ranks = tree_ranks

        # Build cores for each level
        self.cores = nn.ModuleList()

        # Level 0: cores of shape (leaf_dim, leaf_dim, tree_ranks[0])
        level0 = nn.ParameterList()
        for _ in range(n_leaves // 2):
            core = nn.Parameter(torch.randn(leaf_dim, leaf_dim, tree_ranks[0]))
            nn.init.xavier_uniform_(core)
            level0.append(core)
        self.cores.append(level0)

        # Higher levels: cores of shape (tree_ranks[l-1], tree_ranks[l-1], tree_ranks[l])
        for l in range(1, self.L):
            num_cores = n_leaves // (2 ** (l + 1))
            level_cores = nn.ParameterList()
            for _ in range(num_cores):
                core = nn.Parameter(torch.randn(tree_ranks[l-1], tree_ranks[l-1], tree_ranks[l]))
                nn.init.xavier_uniform_(core)
                level_cores.append(core)
            self.cores.append(level_cores)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, n_leaves * leaf_dim)
        Returns:
            Tensor of shape (batch_size, tree_ranks[-1])
        """
        bsz = x.size(0)
        current = x.view(bsz, self.n_leaves, self.leaf_dim)
        for level_cores in self.cores:
            new_nodes = []
            for i in range(0, current.size(1), 2):
                left = current[:, i, :]      # (bsz, leaf_dim or previous rank)
                right = current[:, i+1, :]   # (bsz, leaf_dim or previous rank)
                core = level_cores[i // 2]   # shape depends on level
                # Contract: (bsz,d1) x (bsz,d2) x (d1,d2,r) -> (bsz,r)
                merged = torch.einsum('bi,bj,ijk->bk', left, right, core)
                new_nodes.append(merged)
            current = torch.stack(new_nodes, dim=1)
        return current.squeeze(1)

#############################################
# 2. Quantum Circuit: TTN_VQC
#############################################

class TTN_VQC(tq.QuantumModule):
    """
    A variational quantum circuit with a tree entanglement pattern.
    External rotation angles are provided, and the entanglement is applied in a binary tree fashion.
    """
    def __init__(self, n_wires=4, n_qlayers=1, add_fc=False, out_features=2, noise_prob=0.0):
        super().__init__()
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.add_fc = add_fc
        self.noise_prob = noise_prob

        # Encoder: use a general encoder (RY rotation per qubit).
        self.encoder = tq.GeneralEncoder(
            [{'input_idx': [i], 'func': 'ry', 'wires': [i]} for i in range(n_wires)]
        )
        # Measurement: measure all qubits in the Pauli-Z basis.
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Optional final fully connected layer to produce output logits.
        if add_fc:
            self.fc_layer = nn.Linear(n_wires, out_features)

    def reset_quantum_device(self, bsz: int):
        self.q_device.reset_states(bsz)

    def apply_ttn_entanglement(self):
        """
        Dynamically apply entanglement in a binary tree pattern.
        Assumes that n_wires is a power of 2.
        """
        current_wires = list(range(self.n_wires))
        while len(current_wires) > 1:
            new_wires = []
            for i in range(0, len(current_wires), 2):
                q1 = current_wires[i]
                q2 = current_wires[i+1]
                tqf.cnot(self.q_device, wires=[q1, q2],
                         static=self.static_mode, parent_graph=self.graph)
                new_wires.append(q1)  # Choose q1 as the representative for the merged pair.
            current_wires = new_wires

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
    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice,
                angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_wires) classical data for quantum encoding.
            angles: (batch_size, n_qlayers, n_wires, 3) rotation angles.
        """
        self.q_device = q_device
        bsz = x.shape[0]
        self.reset_quantum_device(bsz)

        # 1. Encode data into the quantum device.
        self.encoder(self.q_device, x)

        # 2. For each variational layer, apply rotations then entangle in a tree pattern.
        for layer_idx in range(self.n_qlayers):
            for i in range(self.n_wires):
                rx_angle = angles[:, layer_idx, i, 0]
                ry_angle = angles[:, layer_idx, i, 1]
                rz_angle = angles[:, layer_idx, i, 2]
                tqf.rx(self.q_device, wires=i, params=rx_angle,
                       static=self.static_mode, parent_graph=self.graph)
                tqf.ry(self.q_device, wires=i, params=ry_angle,
                       static=self.static_mode, parent_graph=self.graph)
                tqf.rz(self.q_device, wires=i, params=rz_angle,
                       static=self.static_mode, parent_graph=self.graph)

            self.apply_ttn_entanglement()
            self.apply_depolarizing_noise()

        qc_out = self.measure(self.q_device)  # (batch_size, n_wires)
        if self.add_fc:
            qc_out = self.fc_layer(qc_out)
        return qc_out

#############################################
# 3. Full Model: TTNParamVQC with Preprocessor
#############################################

class TTNParamVQC(nn.Module):
    """
    Overall pipeline:
      1. Preprocessor: Maps raw input (2500-d) to a lower dimension (2048).
      2. TensorTreeEncoder: Compresses the 2048-d vector into a latent vector.
      3. Angle Mapper: Maps the latent vector to rotation angles for the VQC.
      4. Quantum Encoder: Maps the 2048-d preprocessed vector to n_wires features.
      5. TTN_VQC: Receives the quantum encoding and rotation angles.
    """
    def __init__(self,
                 input_dim=2500,          # Raw input dimension.
                 preproc_out_dim=2048,    # Target dimension after preprocessing.
                 n_leaves=32,             # Must satisfy: n_leaves * leaf_dim = preproc_out_dim.
                 leaf_dim=64,             # 32 * 64 = 2048.
                 tree_ranks=[2,2,2,2,2],   # Reduced TT ranks to help avoid overfitting.
                 n_wires=8,               # Quantum circuit qubit count (power of 2 for tree entanglement).
                 n_qlayers=2,
                 out_features=2,
                 noise_prob=0.01):
        super().__init__()

        # 1. Preprocessor: map raw 2500-d input to 2048-d.
        self.preprocessor = nn.Linear(input_dim, preproc_out_dim)

        # 2. TT Encoder: expects input of dimension preproc_out_dim = n_leaves * leaf_dim.
        self.tt_encoder = TensorTreeEncoder(n_leaves=n_leaves, leaf_dim=leaf_dim, tree_ranks=tree_ranks)
        # The latent output will have dimension tree_ranks[-1] (here, 2).

        # 3. Angle Mapper: for VQC, we need n_qlayers * n_wires * 3 rotation angles.
        self.total_angles = n_qlayers * n_wires * 3
        self.angle_mapper = nn.Linear(tree_ranks[-1], self.total_angles)

        # 4. Quantum Encoder: map the preprocessed vector to n_wires features for the quantum circuit.
        self.encoder_mapper = nn.Linear(preproc_out_dim, n_wires)

        # 5. The TTN_VQC circuit.
        self.ttn_vqc = TTN_VQC(
            n_wires=n_wires,
            n_qlayers=n_qlayers,
            add_fc=True,
            out_features=out_features,
            noise_prob=noise_prob
        )

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice):
        """
        Args:
            x: (batch_size, 2500) raw input.
            q_device: TorchQuantum device.
        """
        bsz = x.shape[0]

        # 1. Preprocess: 2500-d -> 2048-d.
        x_pre = self.preprocessor(x)  # (bsz, 2048)
        x_pre = torch.tanh(x_pre)      # Optional nonlinearity.

        # 2. TT Encoding: compress 2048-d vector into a latent vector.
        latent = self.tt_encoder(x_pre)  # (bsz, tree_ranks[-1]) e.g. (bsz, 2)

        # 3. Map latent to rotation angles.
        angles_flat = self.angle_mapper(latent)  # (bsz, total_angles)
        angles = angles_flat.view(bsz, self.ttn_vqc.n_qlayers, self.ttn_vqc.n_wires, 3)

        # 4. Quantum encoding: map preprocessed vector to n_wires features.
        q_enc = self.encoder_mapper(x_pre)  # (bsz, n_wires)
        q_enc = torch.tanh(q_enc)           # Map values to (-1,1)

        # 5. Pass encoding and angles to the quantum circuit.
        out = self.ttn_vqc(q_enc, q_device=q_device, angles=angles)
        return out

#############################################
# 4. Example Usage / Toy Training Loop
#############################################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create a toy dataset:
    n_samples = 200
    raw_input_dim = 2500
    X = torch.randn(n_samples, raw_input_dim)
    y = torch.randint(0, 2, (n_samples,))  # Binary classification.

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

    # Instantiate the TTNParamVQC model.
    model = TTNParamVQC(
        input_dim=raw_input_dim,
        preproc_out_dim=2048,
        n_leaves=32,
        leaf_dim=64,
        tree_ranks=[2,2,2,2,2],
        n_wires=8,
        n_qlayers=2,
        out_features=2,
        noise_prob=0.01
    ).to(device)

    # Create a TorchQuantum device with n_wires=8.
    q_device = tq.QuantumDevice(n_wires=8, bsz=50).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00029)

    # Training loop.
    for epoch in range(30):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x, q_device)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss = {running_loss/len(loader):.4f}")

    # Evaluate training accuracy.
    with torch.no_grad():
        out = model(X.to(device), q_device)
        preds = torch.argmax(out, dim=1)
        accuracy = (preds == y.to(device)).float().mean().item()
    print(f"Training accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
