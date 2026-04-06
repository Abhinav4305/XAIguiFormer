import numpy as np
path = "/Users/abhinavbhargava/Downloads/XAIguiFormer/EEGBenchmarkDataset/BEED/raw/train/row_02377/"
print("coh:", np.load(path + "row_02377_EC_coherence.npy").shape)
print("wpli:", np.load(path + "row_02377_EC_wpli.npy").shape)
print("demo:", np.load(path + "row_02377_EC_demographics.npy").shape)
print("label:", np.load(path + "row_02377_EC_label.npy").shape)
