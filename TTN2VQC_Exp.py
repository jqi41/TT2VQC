#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 13:08:35 2025

@author: junqi
"""

# Helper Libraries 
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Machine learning related libraries:
import torch
import torch.nn as nn
import torch.optim as optim

import torchquantum as tq

# DataLoader utilities
from torch.utils.data import DataLoader  
from torch.utils.data import Dataset   
  

from TTN2VQC import TTNParamVQC

seed = 1234
torch.manual_seed(seed)

parser = argparse.ArgumentParser(
    description='Training a hybrid quantum-classical model (MLP_VQC variants) for charge stability diagram classification of quantum dots with depolarizing noise simulation.'
)
parser.add_argument('--save_path', metavar='DIR', default='models', help='Path to save the trained model')
parser.add_argument('--num_qubits', default=8, help='Number of qubits in the quantum circuit', type=int)
parser.add_argument('--batch_size', default=8, help='Batch size for training', type=int)
parser.add_argument('--num_epochs', default=30, help='Number of training epochs', type=int)
parser.add_argument('--depth_vqc', default=1, help='Depth (number of variational layers) of the VQC', type=int)
parser.add_argument('--lr', default=0.00029, help='Learning rate', type=float)
parser.add_argument('--test_kind', metavar='DIR', default='gen', help='Test type: "rep" for representation, "gen" for generalization')
parser.add_argument('--model_kind', metavar='DIR', default='mps_vqc', help='Model type: vqc, mps_vqc, tree_vqc')
parser.add_argument('--amplitude_damping_rate', default=0.05, help='amplitude_damping_rate', type=float)
parser.add_argument('--phase_damping_rate', default=0.05, help='phase_damping_rate', type=float)

args = parser.parse_args()

####### Detect if running on a GPU/CPU #######
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("is CUDA available?", torch.cuda.is_available())
else:
    device = torch.device("cpu")
    print("Running on the CPU")
    
# Load datasets
with open('mlqe_2023_edx/week1/dataset/csds.npy', 'rb') as f:
    data_noisy = np.load(f)

with open('mlqe_2023_edx/week1/dataset/csds_noiseless.npy', 'rb') as f:
    data_clean = np.load(f)
    
with open('mlqe_2023_edx/week1/dataset/labels.npy', 'rb') as f:
    labels = np.load(f)
    
# Visualize 10 random noiseless charge stability diagrams with their labels
fig, ax = plt.subplots(1, 10, figsize=(20, 10))
for index, d in enumerate(data_clean[np.random.choice(len(data_clean), size=10)]):
    ax[index].imshow(d)
    ax[index].axis('off')
    ax[index].set_title(f'Label: {labels[index]}')
plt.show()
plt.close()

""" Data preparation """
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_idx = self.data[idx]
        label = self.labels[idx].type(torch.LongTensor)
        return data_idx, label
    
    
def train(model, device, q_device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Flatten MNIST images: (N,1,28,28) -> (N,784)
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        output = model(data, q_device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
    
    return avg_loss, acc


def test(model, device, q_device, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data, q_device)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {acc:.2f}%\n")
    
    return avg_loss, acc
    
    
if __name__ == "__main__":
    # Use the noisy dataset for generalization test, otherwise use the clean dataset.
    if args.test_kind == 'gen':
        dataset = CustomDataset(data_noisy, labels)
    else:
        dataset = CustomDataset(data_clean, labels)
    trainset, testset = torch.utils.data.random_split(
        dataset, (int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8))
    )

    epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    train_loader = DataLoader(trainset, batch_size=batch_size)
    test_loader = DataLoader(testset, batch_size=batch_size)
    
    # Create a TorchQuantum device
    q_device = tq.QuantumDevice(n_wires=args.num_qubits, bsz=batch_size).to(device)

    # Check a sample batch shape
    for X, y in train_loader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        print(y)
        break

    # Flatten each sample and obtain input dimension
    bsz, input_dim = (X.reshape(X.shape[0], -1)).shape
    print(f"Using {device} device")

    # Instantiate the model
    model = TTNParamVQC(
        input_dim=input_dim,
        preproc_out_dim=2048,
        n_leaves=32,
        leaf_dim=64,
        tree_ranks=[2, 2, 2, 2, 2],
        n_wires=args.num_qubits,
        n_qlayers=args.depth_vqc,
        out_features=2,
        noise_prob=0.1
    ).to(device)
    
    # print the status of the model
    # the print command is inherited from nn.Module in the definition of the network
    print(model)

    # We can see the number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {pytorch_total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # For each 10 iterations, the lr is divided by 2
    
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    # Training loop.
    best_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, q_device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, q_device, test_loader, criterion)
        
        # Optional: Save the best model.
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_model.pt")
         
        train_loss_list.append(train_loss)
        train_acc_list.append((train_acc))
        test_loss_list.append(test_loss)
        test_acc_list.append((test_acc))
    print("Training complete")
