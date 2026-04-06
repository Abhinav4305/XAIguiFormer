import torch
import numpy as np
from data.EEGBenchmarkDataset import FreqbandData
import os

print("Checking first item shapes...")
path = "/Users/abhinavbhargava/Downloads/XAIguiFormer/EEGBenchmarkDataset/BEED/raw/train/row_01676/"

cohs = np.load(path + 'row_01676_EC_coherence.npy', allow_pickle=True)
wplis = np.load(path + 'row_01676_EC_wpli.npy', allow_pickle=True)
label = np.load(path + 'row_01676_EC_label.npy', allow_pickle=True)
demographic_info = np.load(path + 'row_01676_EC_demographics.npy', allow_pickle=True)

print("Coherence shape:", cohs.shape)
print("wPLI shape:", wplis.shape)
print("Label shape:", label.shape)
print("Demographics shape:", demographic_info.shape)

for coh, wpli in zip(cohs, wplis):
    coh = np.transpose(coh, (2, 0, 1))
    node_feat = coh.reshape((-1, coh.shape[2]))
    print("PyG node_feat shape:", node_feat.shape)
    
    edge_attr = wpli[np.arange(16), np.arange(16), :].reshape((-1, 1), order='F') # dummy
    print("PyG edge_attr shape:", edge_attr.shape)

print("Label final tensor shape:", torch.from_numpy(label).to(torch.float32).shape)
print("Demo final tensor shape:", torch.from_numpy(demographic_info).t().to(torch.float32).shape)
