# TT2VQC
Leveraging Tensor-Train Network to Generate VQC's Parameters

### Installation

The main dependencies include *pytorch* and *torchquantum*

### Torch Quantum
```
pip3 install torchquantum
```

### 0. Downloading the dataset
```
git clone https://gitlab.com/QMAI/mlqe_2023_edx.git
```

##### Training the Tensor-Train Network to generate VQC's parameters
```
python TT2VQC_Exp.py --num_qubits=12 --depth_vqc=6 --lr=0.002
```

##### Training the Tensor-Tree Network to generate VQC's parameters
```
python TTN2VQC_Exp.py --num_qubits=12 --depth_vqc=6 --lr=0.001
```
