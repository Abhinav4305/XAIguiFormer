"""
XAIguiFormer Visualization Script
==================================
Loads a trained BEED model and generates:
  1. Attention map heatmaps (per transformer layer & head)
  2. XAI frequency band contribution bar chart
  3. Per-class prediction confidence distribution
  4. Training curve summary (from log file)

Usage:
    conda activate XAIguiFormer
    python visualize_model.py
"""

from utils.visualizer import get_local
get_local.activate()

import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from timm.layers import RmsNorm
from config import get_cfg_defaults
from modules.activation import GeGLU
from models.XAIguiFormer import XAIguiFormer
from data.EEGBenchmarkDataset import EEGBenchmarkDataset
from torch_geometric.loader import DataLoader

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "output/results/BEED/XAIguiFormer_lr0.0001_weightdecay0.01_20260318_215111/XAIguiFormer_lr0.0001_weightdecay0.01_20260318_215111.pt"
LOG_PATH = "output/logs/BEED/XAIguiFormer_lr0.0001_weightdecay0.01_20260318_215111.log"
SAVE_DIR = "output/visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3"]


def load_model_and_data():
    """Load the trained model and test dataset."""
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/BEED_model.yaml')
    cfg.freeze()

    # Frequency band
    freqband = dict(cfg.connectome.frequency_band)
    freqband['beta'] = [freqband['theta'][0] / freqband['beta'][0],
                        freqband['theta'][1] / freqband['beta'][1]]

    # Build model
    model = XAIguiFormer(
        cfg.model.num_node_feat,
        cfg.model.num_edge_feat,
        cfg.model.dim_node_feat,
        cfg.model.dim_edge_feat,
        cfg.model.num_classes,
        cfg.model.num_gnn_layer,
        cfg.model.num_head,
        cfg.model.num_transformer_layer,
        torch.tensor(list(freqband.values())),
        cfg.model.gnn_type,
        act_func=GeGLU,
        norm=RmsNorm,
        dropout=cfg.model.dropout,
        explainer_type=cfg.model.explainer_type,
        mlp_ratio=cfg.model.mlp_ratio,
        init_values=cfg.model.init_values,
        attn_drop=cfg.model.attn_drop,
        droppath=cfg.model.droppath,
    )

    # Load weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load test data
    test_dataset = EEGBenchmarkDataset(cfg.root, cfg.dataset, 'test')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    return model, test_loader, device, cfg


def run_inference(model, test_loader, device):
    """Run inference on the test set, collecting attention maps and contributions."""
    all_attn_maps = []
    all_contributions = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            get_local.clear()
            data = data.to(device)

            out = model(data)
            pred_probs = torch.softmax(out[-1], dim=-1)
            preds = pred_probs.argmax(dim=-1)
            labels = data.y.argmax(dim=-1)

            all_preds.append(pred_probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Collect attention maps from the cache (from the XAI-guided 2nd pass)
            attn_key = 'XAIguiAttention.forward'
            if attn_key in get_local.cache and len(get_local.cache[attn_key]) > 0:
                # Each entry: (batch, num_heads, seq_len, seq_len)
                # We get attention maps from all 4 layers × 2 passes = 8 entries
                # Take only the last 4 (the XAI-guided pass)
                num_layers = 4
                attn_entries = get_local.cache[attn_key]
                if len(attn_entries) >= num_layers:
                    guided_attns = attn_entries[-num_layers:]
                    all_attn_maps.append(guided_attns)

            # Collect XAI contributions
            contrib_key = 'XAIguiFormer.forward'
            if contrib_key in get_local.cache and len(get_local.cache[contrib_key]) > 0:
                all_contributions.extend(get_local.cache[contrib_key])

            # Only process a few batches for visualization
            if batch_idx >= 4:
                break

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_attn_maps, all_contributions, all_preds, all_labels


def plot_attention_maps(all_attn_maps):
    """Plot attention heatmaps for each transformer layer and head."""
    if not all_attn_maps:
        print("⚠ No attention maps captured. Skipping attention visualization.")
        return

    # Use the first batch's attention maps
    attn_maps = all_attn_maps[0]  # list of 4 arrays, one per layer
    num_layers = len(attn_maps)

    for layer_idx in range(num_layers):
        attn = attn_maps[layer_idx]  # (batch, num_heads, seq_len, seq_len)
        # Average over the batch dimension
        avg_attn = np.mean(attn, axis=0)  # (num_heads, seq_len, seq_len)
        num_heads = avg_attn.shape[0]

        fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
        if num_heads == 1:
            axes = [axes]

        fig.suptitle(f'Attention Maps — Transformer Layer {layer_idx + 1} (XAI-Guided Pass)',
                     fontsize=14, fontweight='bold', y=1.02)

        for head_idx in range(num_heads):
            ax = axes[head_idx]
            im = ax.imshow(avg_attn[head_idx], cmap='viridis', aspect='auto',
                          vmin=0, vmax=avg_attn[head_idx].max())
            ax.set_title(f'Head {head_idx + 1}', fontsize=12)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, f'attention_layer_{layer_idx + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")


def plot_contributions(all_contributions):
    """Plot XAI frequency band contribution bar chart."""
    if not all_contributions:
        print("⚠ No XAI contributions captured. Skipping contribution visualization.")
        return

    contrib_array = np.array(all_contributions)
    print(f"   Contribution tensor shape: {contrib_array.shape}")

    # For BEED (single band), shape is (num_samples, embedding_dim)
    # For multi-band datasets, shape is (num_samples, num_bands, embedding_dim)
    if contrib_array.ndim == 3 and contrib_array.shape[1] > 1:
        # Multi-band: average across samples and embedding dims
        mean_contrib = np.abs(contrib_array).mean(axis=(0, 2))
        bands = ['delta', 'theta', 'low_α', 'high_α', 'low_β', 'mid_β', 'high_β', 'gamma', 'θ/β ratio']
        bands = bands[:len(mean_contrib)]
    else:
        # Single band (BEED): show the top-16 feature attributions instead
        if contrib_array.ndim == 3:
            flat = np.abs(contrib_array).mean(axis=0).squeeze()  # (embedding_dim,)
        elif contrib_array.ndim == 2:
            flat = np.abs(contrib_array).mean(axis=0)
        else:
            flat = np.abs(contrib_array).flatten()

        # Show top-16 feature dimensions by attribution importance
        top_k = min(16, len(flat))
        top_indices = np.argsort(flat)[-top_k:][::-1]
        mean_contrib = flat[top_indices]
        bands = [f'Feat {i}' for i in top_indices]

    fig, ax = plt.subplots(figsize=(max(10, len(bands) * 0.8), 5))
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(bands)))
    bars = ax.bar(bands, mean_contrib, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, mean_contrib):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mean_contrib) * 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title('XAI Feature Attribution Importance (DeepLift)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Dimension (Top-16 by importance)', fontsize=12)
    ax.set_ylabel('Mean Absolute Attribution', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'frequency_band_contributions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_prediction_confidence(all_preds, all_labels):
    """Plot per-class prediction confidence distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Prediction Confidence Distribution by True Class',
                 fontsize=16, fontweight='bold')

    for cls_idx in range(min(4, len(CLASS_NAMES))):
        ax = axes[cls_idx // 2][cls_idx % 2]
        mask = all_labels == cls_idx

        if mask.sum() == 0:
            ax.set_title(f'{CLASS_NAMES[cls_idx]} (no samples)')
            continue

        class_preds = all_preds[mask]

        # Plot confidence for each predicted class
        x = np.arange(len(CLASS_NAMES))
        mean_conf = class_preds.mean(axis=0)
        std_conf = class_preds.std(axis=0)

        colors = ['#e74c3c' if i != cls_idx else '#2ecc71' for i in range(len(CLASS_NAMES))]
        bars = ax.bar(x, mean_conf, yerr=std_conf, color=colors,
                     edgecolor='black', linewidth=0.5, capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.set_ylabel('Mean Confidence', fontsize=10)
        ax.set_title(f'True: {CLASS_NAMES[cls_idx]} (n={mask.sum()})',
                     fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, mean_conf):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'prediction_confidence.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_training_curves():
    """Parse the training log and plot loss and accuracy curves."""
    if not os.path.exists(LOG_PATH):
        print(f"⚠ Log file not found: {LOG_PATH}. Skipping training curves.")
        return

    epochs, train_bacs, val_bacs, test_bacs = [], [], [], []
    train_losses, val_losses = [], []

    with open(LOG_PATH, 'r') as f:
        content = f.read()

    # Parse epoch summaries
    pattern = (r'Epoch: (\d+)\n'
               r'train balanced accuracy: ([\d.]+)%.+loss: ([\d.]+)\n'
               r'val balanced accuracy: ([\d.]+)%.+loss: ([\d.]+)\n'
               r'test balanced accuracy: ([\d.]+)%.+loss: ([\d.]+)')

    for match in re.finditer(pattern, content):
        epoch = int(match.group(1))
        epochs.append(epoch)
        train_bacs.append(float(match.group(2)))
        train_losses.append(float(match.group(3)))
        val_bacs.append(float(match.group(4)))
        val_losses.append(float(match.group(5)))
        test_bacs.append(float(match.group(6)))

    if not epochs:
        print("⚠ Could not parse training log. Skipping training curves.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Curves — BEED Dataset', fontsize=16, fontweight='bold')

    # ── Accuracy Curves ──
    ax1.plot(epochs, train_bacs, 'o-', color='#3498db', linewidth=2, markersize=3,
             label='Train BAC', alpha=0.9)
    ax1.plot(epochs, val_bacs, 's-', color='#e67e22', linewidth=2, markersize=3,
             label='Val BAC', alpha=0.9)
    # Only plot test where it's non-zero
    test_active = [(e, t) for e, t in zip(epochs, test_bacs) if t > 0]
    if test_active:
        te, tb = zip(*test_active)
        ax1.plot(te, tb, '^-', color='#2ecc71', linewidth=2, markersize=4,
                 label='Test BAC', alpha=0.9)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Balanced Accuracy (%)', fontsize=12)
    ax1.set_title('Balanced Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, max(epochs) + 1)

    # ── Loss Curves ──
    ax2.plot(epochs, train_losses, 'o-', color='#3498db', linewidth=2, markersize=3,
             label='Train Loss', alpha=0.9)
    ax2.plot(epochs, val_losses, 's-', color='#e67e22', linewidth=2, markersize=3,
             label='Val Loss', alpha=0.9)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, max(epochs) + 1)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def main():
    print("=" * 60)
    print("  XAIguiFormer — Visualization Script")
    print("=" * 60)

    # 1. Plot training curves (doesn't need model loading)
    print("\n📊 [1/4] Generating training curves...")
    plot_training_curves()

    # 2. Load model and run inference
    print("\n🔧 [2/4] Loading model and test data...")
    model, test_loader, device, cfg = load_model_and_data()

    print("\n🧠 [3/4] Running inference on test set...")
    all_attn_maps, all_contributions, all_preds, all_labels = run_inference(model, test_loader, device)

    # 3. Generate attention map visualizations
    print("\n🗺️  Generating attention map heatmaps...")
    plot_attention_maps(all_attn_maps)

    # 4. Generate XAI contribution chart
    print("\n📈 Generating frequency band contribution chart...")
    plot_contributions(all_contributions)

    # 5. Generate prediction confidence plots
    print("\n🎯 [4/4] Generating prediction confidence distributions...")
    plot_prediction_confidence(all_preds, all_labels)

    print("\n" + "=" * 60)
    print(f"  All visualizations saved to: {SAVE_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
