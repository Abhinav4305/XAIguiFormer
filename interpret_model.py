"""
XAIguiFormer — Human-Interpretable XAI Analysis
=================================================
Traces attributions from the model's output back to the original 16 EEG
input features (X1–X16), producing:
  1. Per-class feature importance bar charts
  2. Feature importance heatmap (features × classes)
  3. Brain topography maps (colored head diagrams)
  4. Pass 1 vs Pass 2 comparison (how XAI guidance shifts focus)
  5. Cross-method agreement chart

Usage:
    conda activate XAIguiFormer
    python interpret_model.py
"""

from utils.visualizer import get_local
get_local.activate()

import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from timm.layers import RmsNorm
from captum.attr import IntegratedGradients, Saliency, FeatureAblation
from config import get_cfg_defaults
from modules.activation import GeGLU
from models.XAIguiFormer import XAIguiFormer
from data.EEGBenchmarkDataset import EEGBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from utils.electrode_map import (
    ELECTRODE_NAMES, ELECTRODE_POSITIONS_2D, BRAIN_REGIONS,
    REGION_COLORS, CLASS_NAMES, get_all_positions
)

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "output/results/BEED/XAIguiFormer_lr0.0001_weightdecay0.01_20260318_215111/XAIguiFormer_lr0.0001_weightdecay0.01_20260318_215111.pt"
SAVE_DIR = "output/interpretability"
NUM_CLASSES = 4
os.makedirs(SAVE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Model Wrapper for Input-Level Attribution
# ═══════════════════════════════════════════════════════════════════════════════

class InterpretableWrapper(nn.Module):
    """
    Wraps XAIguiFormer so Captum can attribute predictions to raw node features.

    The challenge: XAIguiFormer expects a PyTorch Geometric Data object (graph),
    but Captum needs a model that takes a tensor in and returns a tensor out.

    Solution: Hold the graph structure (edges, demographics, etc.) fixed, and
    only let Captum vary the node features `x`. Internally, we reconstruct
    the full Data object and run the original model.
    """

    def __init__(self, model, data_sample, target_pass=1):
        """
        Args:
            model: Trained XAIguiFormer model
            data_sample: A single PyG Data object to use as template
            target_pass: 0 = unguided (Pass 1), 1 = XAI-guided (Pass 2)
        """
        super().__init__()
        self.model = model
        self.target_pass = target_pass

        # Store fixed graph structure (these don't change during attribution)
        self.edge_index = data_sample.edge_index.clone()
        self.edge_attr = data_sample.edge_attr.clone()
        self.y = data_sample.y.clone()
        self.freqband_order = data_sample.freqband_order.clone()
        self.demographic_info = data_sample.demographic_info.clone()
        self.eid = data_sample.eid
        self.batch = torch.zeros(data_sample.x.shape[0], dtype=torch.long)

    def forward(self, x):
        """
        Args:
            x: Node features tensor. Can be:
               - (16, 16) single sample
               - (B, 16, 16) batch from Captum's IG interpolation
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        # Handle both single and batched inputs
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, 16, 16)

        batch_size = x.shape[0]
        all_logits = []

        for i in range(batch_size):
            xi = x[i]  # (16, 16)

            # Reconstruct the full Data object
            data = Data(
                x=xi,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=self.y,
                freqband_order=self.freqband_order,
                demographic_info=self.demographic_info,
                batch=self.batch,
            )
            data.eid = self.eid

            # Run full XAIguiFormer forward pass
            get_local.clear()
            out = self.model(data)
            all_logits.append(out[self.target_pass])

        return torch.cat(all_logits, dim=0)  # (B, num_classes)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Load Model and Data
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_data():
    """Load the trained model and test dataset."""
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/BEED_model.yaml')
    cfg.freeze()

    freqband = dict(cfg.connectome.frequency_band)
    freqband['beta'] = [freqband['theta'][0] / freqband['beta'][0],
                        freqband['theta'][1] / freqband['beta'][1]]

    model = XAIguiFormer(
        cfg.model.num_node_feat, cfg.model.num_edge_feat,
        cfg.model.dim_node_feat, cfg.model.dim_edge_feat,
        cfg.model.num_classes, cfg.model.num_gnn_layer,
        cfg.model.num_head, cfg.model.num_transformer_layer,
        torch.tensor(list(freqband.values())),
        cfg.model.gnn_type, act_func=GeGLU, norm=RmsNorm,
        dropout=cfg.model.dropout, explainer_type=cfg.model.explainer_type,
        mlp_ratio=cfg.model.mlp_ratio, init_values=cfg.model.init_values,
        attn_drop=cfg.model.attn_drop, droppath=cfg.model.droppath,
    )

    device = 'cpu'  # Attribution methods work best on CPU for stability
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_dataset = EEGBenchmarkDataset(cfg.root, cfg.dataset, 'test')

    return model, test_dataset, device


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Run Attribution Methods
# ═══════════════════════════════════════════════════════════════════════════════

def compute_attributions(model, test_dataset, device, max_samples=200):
    """
    Run 3 attribution methods on test samples.

    Returns:
        results: dict with keys 'ig', 'saliency', 'occlusion', each containing
                 a dict with keys 'pass1' and 'pass2', each containing a list of
                 (attribution_16, true_class, pred_class) tuples.
    """
    results = {
        'ig':        {'pass1': [], 'pass2': []},
        'saliency':  {'pass1': [], 'pass2': []},
        'occlusion': {'pass1': [], 'pass2': []},
    }

    num_samples = min(max_samples, len(test_dataset))
    print(f"   Running attribution on {num_samples} test samples...")

    for idx in range(num_samples):
        data = test_dataset[idx]
        true_class = data.y.argmax(dim=-1).item()
        x_input = data.x.clone().requires_grad_(True)
        baseline = torch.zeros_like(x_input)

        for pass_name, pass_idx in [('pass1', 0), ('pass2', 1)]:
            # Create wrapper for this sample and pass
            wrapper = InterpretableWrapper(model, data, target_pass=pass_idx)
            wrapper.eval()

            # Get predicted class
            with torch.no_grad():
                logits = wrapper(x_input)
                pred_class = logits.argmax(dim=-1).item()

            target_class = pred_class  # Attribute to the predicted class

            # --- Integrated Gradients ---
            ig = IntegratedGradients(wrapper)
            ig_attr = ig.attribute(
                x_input.unsqueeze(0),  # (1, 16, 16)
                baselines=baseline.unsqueeze(0),
                target=target_class,
                n_steps=25,
                internal_batch_size=1,
            ).squeeze(0).detach()  # (16, 16)

            # Aggregate to per-channel importance: sum absolute attributions across feature dim
            ig_channel = ig_attr.abs().sum(dim=1).numpy()  # (16,)

            # --- Saliency ---
            sal = Saliency(wrapper)
            sal_attr = sal.attribute(
                x_input.unsqueeze(0),
                target=target_class,
            ).squeeze(0).detach()
            sal_channel = sal_attr.abs().sum(dim=1).numpy()

            # --- Occlusion (Feature Ablation) ---
            # Create a mask that ablates one node (row) at a time
            fa = FeatureAblation(wrapper)
            # Feature mask: group by rows (each row = one EEG channel)
            feature_mask = torch.arange(16).unsqueeze(1).expand(16, 16)  # (16, 16)
            fa_attr = fa.attribute(
                x_input.unsqueeze(0),
                baselines=baseline.unsqueeze(0),
                target=target_class,
                feature_mask=feature_mask.unsqueeze(0),
            ).squeeze(0).detach()
            fa_channel = fa_attr.abs().sum(dim=1).numpy()

            results['ig'][pass_name].append((ig_channel, true_class, pred_class))
            results['saliency'][pass_name].append((sal_channel, true_class, pred_class))
            results['occlusion'][pass_name].append((fa_channel, true_class, pred_class))

        if (idx + 1) % 50 == 0:
            print(f"   ... processed {idx + 1}/{num_samples} samples")

    return results


def aggregate_per_class(results_list):
    """
    Aggregate attributions per class.

    Args:
        results_list: list of (attribution_16, true_class, pred_class) tuples
    Returns:
        per_class: dict {class_id: mean_attribution_16}
    """
    per_class = {}
    for cls in range(NUM_CLASSES):
        cls_attrs = [attr for attr, tc, pc in results_list if tc == cls]
        if cls_attrs:
            mean_attr = np.mean(cls_attrs, axis=0)
            # Normalize to [0, 1] for comparability
            if mean_attr.max() > 0:
                mean_attr = mean_attr / mean_attr.max()
            per_class[cls] = mean_attr
        else:
            per_class[cls] = np.zeros(16)
    return per_class


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Visualization Functions
# ═══════════════════════════════════════════════════════════════════════════════

def plot_per_class_importance(per_class_ig, per_class_sal, per_class_occ):
    """Plot 1: Per-class feature importance bar charts (using IG as primary)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Class EEG Channel Importance\n(Integrated Gradients Attribution)',
                 fontsize=16, fontweight='bold')

    for cls in range(NUM_CLASSES):
        ax = axes[cls // 2][cls % 2]
        importances = per_class_ig[cls]

        # Color by brain region
        colors = [REGION_COLORS[BRAIN_REGIONS[e]] for e in ELECTRODE_NAMES]
        bars = ax.bar(range(16), importances, color=colors, edgecolor='black',
                     linewidth=0.5, alpha=0.85)

        # Add value labels on top bars
        for i, (bar, val) in enumerate(zip(bars, importances)):
            if val > 0.3:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xticks(range(16))
        ax.set_xticklabels([f'{ELECTRODE_NAMES[i]}\n(X{i+1})' for i in range(16)],
                          fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('Normalized Importance', fontsize=10)
        ax.set_title(f'Class {cls}: {CLASS_NAMES[cls]}', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)

    # Add legend for brain regions
    legend_patches = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items()]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=10,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(SAVE_DIR, 'per_class_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {path}")


def plot_feature_heatmap(per_class_ig):
    """Plot 2: Feature importance heatmap (features × classes)."""
    matrix = np.array([per_class_ig[cls] for cls in range(NUM_CLASSES)])  # (4, 16)

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(16))
    ax.set_xticklabels([f'{ELECTRODE_NAMES[i]}\n(X{i+1})' for i in range(16)],
                      fontsize=9)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels([f'Class {c}: {CLASS_NAMES[c]}' for c in range(NUM_CLASSES)],
                      fontsize=11)

    # Add text annotations
    for i in range(NUM_CLASSES):
        for j in range(16):
            val = matrix[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=7, fontweight='bold')

    ax.set_title('Feature Importance Heatmap — EEG Channels × Epilepsy Classes\n'
                 '(Integrated Gradients, normalized per class)',
                 fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalized Importance', shrink=0.8)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'feature_importance_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {path}")


def plot_brain_topography(per_class_ig):
    """Plot 3: Brain topography maps showing which brain regions matter per class."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle('Brain Topography — EEG Channel Importance by Class\n'
                 '(Integrated Gradients Attribution)',
                 fontsize=16, fontweight='bold')

    xs, ys = get_all_positions()
    xs, ys = np.array(xs), np.array(ys)

    for cls in range(NUM_CLASSES):
        ax = axes[cls]
        importances = per_class_ig[cls]

        # Draw head outline
        head_circle = plt.Circle((0, 0), 0.52, fill=False, linewidth=2, color='#333')
        ax.add_patch(head_circle)

        # Draw nose
        nose_x = [(-0.05, 0.52), (0, 0.58), (0.05, 0.52)]
        ax.plot([p[0] for p in nose_x], [p[1] for p in nose_x], 'k-', linewidth=2)

        # Draw ears
        ear_l = plt.Circle((-0.55, 0), 0.04, fill=False, linewidth=1.5, color='#333')
        ear_r = plt.Circle(( 0.55, 0), 0.04, fill=False, linewidth=1.5, color='#333')
        ax.add_patch(ear_l)
        ax.add_patch(ear_r)

        # Interpolate for smooth heatmap
        grid_x, grid_y = np.mgrid[-0.55:0.55:100j, -0.55:0.55:100j]
        grid_z = griddata((xs, ys), importances, (grid_x, grid_y), method='cubic',
                         fill_value=0)

        # Mask outside the head
        dist = np.sqrt(grid_x**2 + grid_y**2)
        grid_z[dist > 0.52] = np.nan

        # Plot interpolated heatmap
        im = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='YlOrRd',
                        vmin=0, vmax=1, alpha=0.7)

        # Plot electrode positions
        scatter = ax.scatter(xs, ys, c=importances, cmap='YlOrRd',
                           s=180, edgecolors='black', linewidths=1.5,
                           vmin=0, vmax=1, zorder=5)

        # Label electrodes
        for i, name in enumerate(ELECTRODE_NAMES):
            ax.annotate(name, (xs[i], ys[i]), fontsize=6, ha='center',
                       va='bottom', fontweight='bold',
                       xytext=(0, 8), textcoords='offset points')

        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.65, 0.7)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Class {cls}: {CLASS_NAMES[cls]}', fontsize=12, fontweight='bold')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Normalized Importance', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.91, 0.93])
    path = os.path.join(SAVE_DIR, 'brain_topography.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {path}")


def plot_pass_comparison(model, test_dataset, max_samples=200):
    """
    Plot 4: Pass 1 vs Pass 2 comparison.

    Instead of comparing input-level attributions (which are similar because both
    passes share the ConnectomeEncoder), this shows WHERE the XAI actually makes
    a difference: at the prediction confidence and class-decision level.
    """
    pass1_confs = []    # confidence of predicted class in Pass 1
    pass2_confs = []    # confidence of predicted class in Pass 2
    pass1_preds = []
    pass2_preds = []
    true_labels = []
    pass1_correct = []
    pass2_correct = []

    num_samples = min(max_samples, len(test_dataset))

    for idx in range(num_samples):
        data = test_dataset[idx]
        true_class = data.y.argmax(dim=-1).item()
        true_labels.append(true_class)

        wrapper_p1 = InterpretableWrapper(model, data, target_pass=0)
        wrapper_p2 = InterpretableWrapper(model, data, target_pass=1)

        logits1 = wrapper_p1(data.x)
        logits2 = wrapper_p2(data.x)

        probs1 = torch.softmax(logits1, dim=-1).squeeze()
        probs2 = torch.softmax(logits2, dim=-1).squeeze()

        pred1 = probs1.argmax().item()
        pred2 = probs2.argmax().item()

        pass1_confs.append(probs1[pred1].item())
        pass2_confs.append(probs2[pred2].item())
        pass1_preds.append(pred1)
        pass2_preds.append(pred2)
        pass1_correct.append(pred1 == true_class)
        pass2_correct.append(pred2 == true_class)

    pass1_confs = np.array(pass1_confs)
    pass2_confs = np.array(pass2_confs)
    true_labels = np.array(true_labels)
    pass1_correct = np.array(pass1_correct)
    pass2_correct = np.array(pass2_correct)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('How XAI Guidance Changes Model Predictions\n'
                 'Pass 1 (Unguided) vs Pass 2 (XAI-Guided)',
                 fontsize=16, fontweight='bold')

    # --- Panel 1: Confidence scatter plot ---
    ax1 = fig.add_subplot(2, 2, 1)
    colors_scatter = ['#2ecc71' if p2c and not p1c else '#e74c3c' if p1c and not p2c
                      else '#3498db' for p1c, p2c in zip(pass1_correct, pass2_correct)]
    ax1.scatter(pass1_confs, pass2_confs, c=colors_scatter, alpha=0.5, s=30, edgecolors='black', linewidths=0.3)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='No change line')
    ax1.set_xlabel('Pass 1 Confidence (Unguided)', fontsize=11)
    ax1.set_ylabel('Pass 2 Confidence (XAI-Guided)', fontsize=11)
    ax1.set_title('Prediction Confidence: Pass 1 vs Pass 2', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='Fixed by XAI (wrong→right)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Broken by XAI (right→wrong)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Same correctness'),
    ]
    ax1.legend(handles=legend_elements, fontsize=8, loc='lower right')
    ax1.grid(alpha=0.3)

    # --- Panel 2: Per-class accuracy comparison ---
    ax2 = fig.add_subplot(2, 2, 2)
    p1_acc_per_class = []
    p2_acc_per_class = []
    for cls in range(NUM_CLASSES):
        mask = true_labels == cls
        if mask.sum() > 0:
            p1_acc_per_class.append(pass1_correct[mask].mean() * 100)
            p2_acc_per_class.append(pass2_correct[mask].mean() * 100)
        else:
            p1_acc_per_class.append(0)
            p2_acc_per_class.append(0)

    x_pos = np.arange(NUM_CLASSES)
    bar_width = 0.35
    ax2.bar(x_pos - bar_width/2, p1_acc_per_class, bar_width, label='Pass 1 (Unguided)',
            color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.bar(x_pos + bar_width/2, p2_acc_per_class, bar_width, label='Pass 2 (XAI-Guided)',
            color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add value labels
    for i in range(NUM_CLASSES):
        ax2.text(x_pos[i] - bar_width/2, p1_acc_per_class[i] + 1, f'{p1_acc_per_class[i]:.1f}%',
                ha='center', fontsize=8, fontweight='bold', color='#2c3e50')
        ax2.text(x_pos[i] + bar_width/2, p2_acc_per_class[i] + 1, f'{p2_acc_per_class[i]:.1f}%',
                ha='center', fontsize=8, fontweight='bold', color='#c0392b')
        # Show delta
        delta = p2_acc_per_class[i] - p1_acc_per_class[i]
        symbol = '+' if delta >= 0 else ''
        color = '#27ae60' if delta >= 0 else '#e74c3c'
        ax2.text(x_pos[i], max(p1_acc_per_class[i], p2_acc_per_class[i]) + 5,
                f'{symbol}{delta:.1f}%', ha='center', fontsize=9, fontweight='bold', color=color)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Class {c}\n{CLASS_NAMES[c]}' for c in range(NUM_CLASSES)], fontsize=9)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Per-Class Accuracy: Pass 1 vs Pass 2', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # --- Panel 3: Confidence improvement distribution ---
    ax3 = fig.add_subplot(2, 2, 3)
    conf_diff = pass2_confs - pass1_confs
    colors_hist = ['#2ecc71' if d > 0 else '#e74c3c' for d in conf_diff]

    # Plot as histogram
    bins = np.linspace(-0.5, 0.5, 40)
    pos_vals = conf_diff[conf_diff >= 0]
    neg_vals = conf_diff[conf_diff < 0]
    ax3.hist(pos_vals, bins=bins, color='#2ecc71', alpha=0.7, label=f'Improved ({len(pos_vals)} samples)', edgecolor='black', linewidth=0.3)
    ax3.hist(neg_vals, bins=bins, color='#e74c3c', alpha=0.7, label=f'Degraded ({len(neg_vals)} samples)', edgecolor='black', linewidth=0.3)
    ax3.axvline(x=0, color='black', linewidth=1.5, linestyle='--')
    ax3.axvline(x=conf_diff.mean(), color='#8e44ad', linewidth=2, linestyle='-',
               label=f'Mean shift: {conf_diff.mean():+.4f}')
    ax3.set_xlabel('Confidence Change (Pass 2 - Pass 1)', fontsize=11)
    ax3.set_ylabel('Number of Samples', fontsize=11)
    ax3.set_title('Distribution of Confidence Changes', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # --- Panel 4: Class flip analysis ---
    ax4 = fig.add_subplot(2, 2, 4)

    # Count prediction changes
    same_pred = sum(1 for p1, p2 in zip(pass1_preds, pass2_preds) if p1 == p2)
    wrong_to_right = sum(1 for p1, p2, tc in zip(pass1_preds, pass2_preds, true_labels)
                         if p1 != tc and p2 == tc)
    right_to_wrong = sum(1 for p1, p2, tc in zip(pass1_preds, pass2_preds, true_labels)
                         if p1 == tc and p2 != tc)
    wrong_to_wrong_diff = sum(1 for p1, p2, tc in zip(pass1_preds, pass2_preds, true_labels)
                              if p1 != p2 and p1 != tc and p2 != tc)

    categories = ['No Change', 'Fixed by XAI\n(wrong→right)', 'Broken by XAI\n(right→wrong)', 'Changed but\nstill wrong']
    values = [same_pred, wrong_to_right, right_to_wrong, wrong_to_wrong_diff]
    colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    bars = ax4.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val} ({val/num_samples*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

    ax4.set_ylabel('Number of Samples', fontsize=11)
    ax4.set_title('XAI Impact on Predictions (Class Flips)', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # Overall stats annotation
    overall_p1_acc = pass1_correct.mean() * 100
    overall_p2_acc = pass2_correct.mean() * 100
    stats_text = (f'Overall Accuracy:\n'
                  f'  Pass 1: {overall_p1_acc:.1f}%\n'
                  f'  Pass 2: {overall_p2_acc:.1f}%\n'
                  f'  Δ: {overall_p2_acc - overall_p1_acc:+.1f}%')
    ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(SAVE_DIR, 'pass1_vs_pass2_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {path}")


def plot_method_agreement(per_class_ig, per_class_sal, per_class_occ):
    """Plot 5: Cross-method agreement — do all 3 methods agree on important features?"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Cross-Method Agreement on Feature Importance\n'
                 'Integrated Gradients vs Saliency vs Occlusion',
                 fontsize=16, fontweight='bold')

    x_pos = np.arange(16)
    bar_width = 0.25

    for cls in range(NUM_CLASSES):
        ax = axes[cls // 2][cls % 2]

        ax.bar(x_pos - bar_width, per_class_ig[cls], bar_width,
               label='Integrated Gradients', color='#3498db', alpha=0.8,
               edgecolor='black', linewidth=0.5)
        ax.bar(x_pos, per_class_sal[cls], bar_width,
               label='Saliency', color='#2ecc71', alpha=0.8,
               edgecolor='black', linewidth=0.5)
        ax.bar(x_pos + bar_width, per_class_occ[cls], bar_width,
               label='Occlusion', color='#e67e22', alpha=0.8,
               edgecolor='black', linewidth=0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{ELECTRODE_NAMES[i]}\n(X{i+1})' for i in range(16)],
                          fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('Normalized Importance', fontsize=10)
        ax.set_title(f'Class {cls}: {CLASS_NAMES[cls]}', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.25)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(SAVE_DIR, 'method_agreement.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {path}")


def print_summary_table(per_class_ig):
    """Print a human-readable summary of the most important features per class."""
    print("\n" + "=" * 70)
    print("  HUMAN-INTERPRETABLE FEATURE IMPORTANCE SUMMARY")
    print("=" * 70)

    for cls in range(NUM_CLASSES):
        importances = per_class_ig[cls]
        ranked = np.argsort(importances)[::-1]

        print(f"\n  Class {cls}: {CLASS_NAMES[cls]}")
        print(f"  {'─' * 55}")
        print(f"  {'Rank':<6} {'Feature':<8} {'Electrode':<12} {'Region':<12} {'Importance':<10}")
        print(f"  {'─' * 55}")
        for rank, idx in enumerate(ranked[:5]):
            name = ELECTRODE_NAMES[idx]
            region = BRAIN_REGIONS[name]
            imp = importances[idx]
            bar = '█' * int(imp * 20)
            print(f"  {rank+1:<6} X{idx+1:<7} {name:<12} {region:<12} {imp:.4f}  {bar}")

    print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  XAIguiFormer — Human-Interpretable XAI Analysis")
    print("=" * 65)

    # 1. Load
    print("\n🔧 [1/5] Loading trained model and test data...")
    model, test_dataset, device = load_model_and_data()
    print(f"   Model loaded from: {MODEL_PATH}")
    print(f"   Test samples: {len(test_dataset)}")

    # 2. Compute attributions
    print("\n🧠 [2/5] Computing attributions (3 methods × 2 passes)...")
    print("   This will take ~15-20 minutes on CPU. Please wait...")
    results = compute_attributions(model, test_dataset, device, max_samples=200)

    # 3. Aggregate per class
    print("\n📊 [3/5] Aggregating attributions per class...")
    per_class_ig_p1 = aggregate_per_class(results['ig']['pass1'])
    per_class_ig_p2 = aggregate_per_class(results['ig']['pass2'])
    per_class_sal_p2 = aggregate_per_class(results['saliency']['pass2'])
    per_class_occ_p2 = aggregate_per_class(results['occlusion']['pass2'])

    # 4. Generate visualizations
    print("\n🎨 [4/5] Generating visualizations...")

    print("   📊 Plot 1: Per-class feature importance...")
    plot_per_class_importance(per_class_ig_p2, per_class_sal_p2, per_class_occ_p2)

    print("   🗺️  Plot 2: Feature importance heatmap...")
    plot_feature_heatmap(per_class_ig_p2)

    print("   🧠 Plot 3: Brain topography maps...")
    plot_brain_topography(per_class_ig_p2)

    print("   🔄 Plot 4: Pass 1 vs Pass 2 comparison...")
    plot_pass_comparison(model, test_dataset, max_samples=200)

    print("   ✓  Plot 5: Cross-method agreement...")
    plot_method_agreement(per_class_ig_p2, per_class_sal_p2, per_class_occ_p2)

    # 5. Print summary
    print("\n📋 [5/5] Summary of findings...")
    print_summary_table(per_class_ig_p2)

    print(f"\n{'=' * 65}")
    print(f"  All visualizations saved to: {SAVE_DIR}/")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
