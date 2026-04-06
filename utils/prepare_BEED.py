import pandas as pd
import numpy as np
import os
import torch

'''
Convert 1D tabular dataset BEED_Data.csv into pseudo-connectomes.
Each row represents 1 pseudo-subject. 
Features X1-X16 become 16 brain nodes. Graph edges are absolute differences between nodes.
'''

def create_pseudo_connectomes(csv_path, save_dir):
    print("Loading BEED_Data.csv...")
    df = pd.read_csv(csv_path)

    # Prevent time-series leakage by splitting block-wise per class instead of random globals
    train_dfs, val_dfs, test_dfs = [], [], []
    for c in sorted(df['y'].unique()):
        class_df = df[df['y'] == c]
        num_samples = len(class_df)
        train_idx = int(0.6 * num_samples)
        val_idx = int(0.8 * num_samples)
        
        train_dfs.append(class_df.iloc[:train_idx])
        val_dfs.append(class_df.iloc[train_idx:val_idx])
        test_dfs.append(class_df.iloc[val_idx:])
        
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)
    
    # Calculate Z-score normalization factors (fit only on train data to prevent leakage)
    feature_cols = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16']
    train_mean = train_df[feature_cols].values.mean(axis=0)
    train_std = train_df[feature_cols].values.std(axis=0)
    train_std[train_std == 0] = 1e-6 # prevent division by zero
    
    def process_split(split_df, split_name):
        print(f"Processing {split_name} split ({len(split_df)} samples)...")
        for i, (idx, row) in enumerate(split_df.iterrows()):
            features = row[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16']].values.astype(np.float32)
            
            # Apply Z-score normalization globally using train set distribution parameters
            features = ((features - train_mean) / train_std).astype(np.float32)
            
            label = int(row['y'])
            
            # Sub-ID based on actual logical row (so it's unique)
            sub_id = f"row_{idx:05d}"
            subject_dir = os.path.join(save_dir, split_name, sub_id)
            os.makedirs(subject_dir, exist_ok=True)
            
            # --- 1. CONNECTOME NODES (Coherence substitute) ---
            # Shape needed: (n_epochs=1, n_nodes=16, n_nodes=16, n_freqbands=1)
            # In EEGBenchmarkDataset, the diagonal or average is often taken to get node features.
            # We'll put the literal feature value on the diagonal, and 0s elsewhere representing no "coherence" interaction.
            coh = np.zeros((1, 16, 16, 1), dtype=np.float32)
            for n in range(16):
                coh[0, n, n, 0] = features[n]
                
            # --- 2. CONNECTOME EDGES (wPLI substitute) ---
            # Shape needed: (n_epochs=1, n_nodes=16, n_nodes=16, n_freqbands=1)
            # Make a fully connected graph utilizing distance between features as the "connection weight"
            wpli = np.zeros((1, 16, 16, 1), dtype=np.float32)
            for r in range(16):
                for c in range(16):
                    if r != c: # Don't connect a node to itself for wPLI
                        # Calculate Edge Connection via Gaussian RBF Kernel on normalized features
                        gamma = 1.0
                        sq_dist = (features[r] - features[c]) ** 2
                        edge_weight = float(np.exp(-gamma * sq_dist))
                        wpli[0, r, c, 0] = edge_weight
                        
            
            # --- 3. DEMOGRAPHICS ---
            # Dummy [age, gender] shape (2,1)
            demo = np.array([[50.0], [0.0]], dtype=np.float32)
            
            # --- 4. LABELS ---
            # Convert 4-class label (0 to 3) to one-hot array [1,0,0,0], [0,1,0,0], etc.
            label_oh = np.zeros(4, dtype=np.float32)
            label_oh[label] = 1.0
            label_oh = label_oh.reshape(1, 4)
            
            # Save files
            np.save(os.path.join(subject_dir, f"{sub_id}_EC_coherence.npy"), coh)
            np.save(os.path.join(subject_dir, f"{sub_id}_EC_wpli.npy"), wpli)
            np.save(os.path.join(subject_dir, f"{sub_id}_EC_demographics.npy"), demo)
            np.save(os.path.join(subject_dir, f"{sub_id}_EC_label.npy"), label_oh)

    
    process_split(train_df, 'train')
    process_split(val_df, 'val')
    process_split(test_df, 'test')
    
    print("\n============================================================")
    print("Data preparation complete!")
    print("============================================================")
    print(f"Output directory: {save_dir}")

if __name__ == '__main__':
    csv_file = "/Users/abhinavbhargava/Downloads/Capstone/BEED_Data.csv"
    output_directory = "/Users/abhinavbhargava/Downloads/XAIguiFormer/EEGBenchmarkDataset/BEED/raw"
    create_pseudo_connectomes(csv_file, output_directory)
