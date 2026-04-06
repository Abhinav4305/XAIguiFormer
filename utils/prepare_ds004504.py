"""
Prepare the ds004504 (Alzheimer's/FTD/Healthy EEG) dataset for XAIguiFormer.

This script:
1. Reads preprocessed EEGLAB .set files from the ds004504 derivatives folder
2. Constructs brain connectomes (coherence + wPLI) across frequency bands
3. Extracts demographics (age, gender) from participants.tsv
4. Creates one-hot labels for binary classification (Alzheimer's vs Healthy)
5. Splits data into train/val/test sets
6. Saves everything in the directory structure expected by XAIguiFormer

Usage (run in the 'mne' conda environment):
    conda activate mne
    cd /path/to/XAIguiFormer
    python utils/prepare_ds004504.py

Requirements:
    - mne, mne_connectivity (installed in 'mne' conda env)
    - numpy, pandas, scikit-learn
"""

import os
import sys
import mne
import numpy as np
import pandas as pd
import mne_connectivity
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION — Update these paths as needed
# ============================================================
DATASET_ROOT = '/Users/abhinavbhargava/Downloads/Capstone'
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'EEGBenchmarkDataset')

PARTICIPANTS_FILE = os.path.join(DATASET_ROOT, 'participants.tsv')
DERIVATIVES_DIR = os.path.join(DATASET_ROOT, 'derivatives')

# Use binary classification: Alzheimer's (A) vs Healthy (C)
# Set to True to also include FTD (F) as a third class
USE_THREE_CLASSES = False

# Frequency bands (matching XAIguiFormer's TUAB config)
FREQ_LOWER = (2., 4., 8., 10., 12., 18., 21., 30., 12.)
FREQ_UPPER = (4., 8., 10., 12., 18., 21., 30., 45., 30.)
# Band names: delta, theta, low_alpha, high_alpha, low_beta, mid_beta, high_beta, gamma, beta
# Index 8 (beta) will be replaced with theta/beta ratio

# Channel name mapping: modern 10-20 names → classic names used by TUAB
CHANNEL_RENAME = {
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
    'FP1': 'Fp1', 'FP2': 'Fp2',
    'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
}

# Standard 19 EEG channels in 10-20 system (order matches TUAB)
STANDARD_19 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz'
]

# Epoch parameters
EPOCH_DURATION = 30.0   # seconds per long epoch
EPOCH_OVERLAP = 10.0    # overlap between long epochs
SHORT_EPOCH_DURATION = 3.0  # for connectivity estimation


def cal_tbr(matrix, theta_idx=1, beta_idx=8):
    """Calculate theta/beta ratio, handling division by zero."""
    theta = matrix[:, :, theta_idx]
    beta = matrix[:, :, beta_idx]
    tbr = np.divide(theta, beta, where=beta != 0, out=np.zeros_like(theta))
    tbr[np.isinf(tbr)] = 0
    tbr[np.isnan(tbr)] = 0
    return tbr


def find_set_file(sub_dir):
    """Find the .set file for a subject in their eeg directory."""
    eeg_dir = os.path.join(sub_dir, 'eeg')
    if not os.path.isdir(eeg_dir):
        return None
    for f in os.listdir(eeg_dir):
        if f.endswith('.set'):
            return os.path.join(eeg_dir, f)
    return None


def fix_channel_names(raw):
    """Rename channels to match standard 10-20 naming convention."""
    rename_map = {}
    for ch_name in raw.ch_names:
        # Check direct mapping
        if ch_name in CHANNEL_RENAME:
            rename_map[ch_name] = CHANNEL_RENAME[ch_name]
        # Check uppercase version
        elif ch_name.upper() in CHANNEL_RENAME:
            rename_map[ch_name] = CHANNEL_RENAME[ch_name.upper()]
    if rename_map:
        raw.rename_channels(rename_map)
    return raw


def pick_and_order_channels(raw):
    """Pick the 19 standard EEG channels and reorder them."""
    available = raw.ch_names
    channels_to_pick = []
    for ch in STANDARD_19:
        if ch in available:
            channels_to_pick.append(ch)
        else:
            # Try case-insensitive match
            found = False
            for avail_ch in available:
                if avail_ch.lower() == ch.lower():
                    channels_to_pick.append(avail_ch)
                    found = True
                    break
            if not found:
                print(f"  WARNING: Channel {ch} not found in data. Available: {available}")

    if len(channels_to_pick) < 19:
        print(f"  WARNING: Only {len(channels_to_pick)}/19 standard channels found")
        print(f"  Available channels: {available}")
        print(f"  Matched channels: {channels_to_pick}")

    raw.pick(channels_to_pick)
    # Reorder to standard order
    ordered = [ch for ch in STANDARD_19 if ch in raw.ch_names]
    if ordered:
        raw.reorder_channels(ordered)
    return raw


def construct_connectome(epochs_data, sfreq, ch_names):
    """Construct coherence and wPLI connectomes for a set of epochs."""
    coherence_list = []
    wpli_list = []

    n_epochs = len(epochs_data)

    for i, single_epoch in enumerate(epochs_data):
        print(f"    Processing epoch {i+1}/{n_epochs}...", end='', flush=True)

        # Create a RawArray from the single epoch data
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        epoch_raw = mne.io.RawArray(single_epoch, info, verbose=False)

        # Segment into 3s sub-epochs for connectivity estimation
        short_events = mne.make_fixed_length_events(
            epoch_raw, duration=SHORT_EPOCH_DURATION, id=1
        )

        if len(short_events) < 2:
            print(" skipped (too short)")
            continue

        short_epochs = mne.Epochs(
            epoch_raw, short_events, baseline=None,
            tmin=0, tmax=SHORT_EPOCH_DURATION - 1.0/sfreq,
            preload=True, proj=False, verbose=False
        )

        # Compute spectral connectivity
        try:
            con = mne_connectivity.spectral_connectivity_epochs(
                short_epochs, names=ch_names,
                method=['coh', 'wpli'], sfreq=sfreq,
                mode='multitaper',
                fmin=FREQ_LOWER, fmax=FREQ_UPPER,
                faverage=True, verbose=False
            )
        except Exception as e:
            print(f" error: {e}")
            continue

        coh = con[0]
        wpli = con[1]

        # Convert to dense numpy arrays: (n_nodes, n_nodes, n_freqs)
        coh_dense = coh.get_data(output='dense')
        wpli_dense = wpli.get_data(output='dense')

        # Replace beta band (index 8) with theta/beta ratio
        coh_dense[:, :, 8] = cal_tbr(coh_dense)
        wpli_dense[:, :, 8] = cal_tbr(wpli_dense)

        # Make symmetric
        coh_dense = coh_dense + np.transpose(coh_dense, (1, 0, 2))
        wpli_dense = wpli_dense + np.transpose(wpli_dense, (1, 0, 2))

        coherence_list.append(coh_dense)
        wpli_list.append(wpli_dense)
        print(" done")

    return coherence_list, wpli_list


def process_subject(sub_id, set_file, participants_df):
    """Process a single subject: read EEG, construct connectomes, extract demographics."""
    print(f"\n{'='*60}")
    print(f"Processing {sub_id}...")
    print(f"  File: {set_file}")

    # 1. Read the .set file
    try:
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return None

    print(f"  Channels: {raw.ch_names} ({len(raw.ch_names)} total)")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.n_times / raw.info['sfreq']:.1f}s")

    # 2. Fix channel names and pick 19 standard EEG channels
    fix_channel_names(raw)
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names}, verbose=False)
    pick_and_order_channels(raw)

    n_channels = len(raw.ch_names)
    print(f"  After channel selection: {raw.ch_names} ({n_channels} channels)")

    if n_channels < 19:
        print(f"  ERROR: Need 19 channels, only have {n_channels}. Skipping.")
        return None

    # 3. Set montage
    try:
        raw.set_montage('standard_1020', on_missing='warn', verbose=False)
    except Exception as e:
        print(f"  WARNING: Could not set montage: {e}")

    # 4. Create epochs (long segments)
    events = mne.make_fixed_length_events(
        raw, duration=EPOCH_DURATION, overlap=EPOCH_OVERLAP, id=1
    )
    if len(events) == 0:
        # If recording is too short for 30s epochs, use the whole recording
        print(f"  Recording too short for {EPOCH_DURATION}s epochs, using full recording")
        events = mne.make_fixed_length_events(raw, duration=raw.n_times/raw.info['sfreq'] - 0.1, id=1)

    epochs = mne.Epochs(
        raw, events, baseline=None,
        tmin=0, tmax=EPOCH_DURATION - 1.0/raw.info['sfreq'],
        preload=True, proj=False, verbose=False
    )
    print(f"  Created {len(epochs)} epochs of {EPOCH_DURATION}s")

    # 5. Construct connectomes
    print(f"  Constructing connectomes...")
    coherence_list, wpli_list = construct_connectome(
        epochs.get_data(), raw.info['sfreq'], list(raw.ch_names)
    )

    if len(coherence_list) == 0:
        print(f"  ERROR: No valid connectomes produced. Skipping.")
        return None

    # 6. Extract demographics from participants.tsv
    row = participants_df[participants_df['participant_id'] == sub_id]
    if row.empty:
        print(f"  ERROR: {sub_id} not found in participants.tsv. Skipping.")
        return None

    age = float(row['Age'].values[0])
    gender_str = row['Gender'].values[0]
    gender = 0 if gender_str == 'M' else 1  # 0=Male, 1=Female
    demographics = np.array([age, gender]).reshape(2, 1)

    # 7. Create label (one-hot)
    group = row['Group'].values[0]
    if USE_THREE_CLASSES:
        # 3-class: [A, C, F]
        if group == 'A':
            label = np.array([[1, 0, 0]])
        elif group == 'C':
            label = np.array([[0, 1, 0]])
        else:  # F
            label = np.array([[0, 0, 1]])
    else:
        # 2-class: [Abnormal/Alzheimer's, Normal/Healthy]
        if group == 'A':
            label = np.array([[1, 0]])
        else:  # C (Healthy)
            label = np.array([[0, 1]])

    print(f"  Demographics: age={age}, gender={'F' if gender else 'M'}")
    print(f"  Group: {group}, Label: {label}")
    print(f"  Connectomes: {len(coherence_list)} epochs, shape: {coherence_list[0].shape}")

    return {
        'coherence': coherence_list,
        'wpli': wpli_list,
        'demographics': demographics,
        'label': label,
        'group': group,
        'sub_id': sub_id
    }


def save_subject(result, split, output_dir):
    """Save a subject's data in the expected directory structure."""
    sub_id = result['sub_id']
    save_dir = os.path.join(output_dir, 'ds004504', 'raw', split, sub_id)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f'{sub_id}_EC_coherence.npy'), result['coherence'])
    np.save(os.path.join(save_dir, f'{sub_id}_EC_wpli.npy'), result['wpli'])
    np.save(os.path.join(save_dir, f'{sub_id}_EC_demographics.npy'), result['demographics'])
    np.save(os.path.join(save_dir, f'{sub_id}_EC_label.npy'), result['label'])

    print(f"  Saved {sub_id} to {split}/")


def main():
    print("=" * 60)
    print("ds004504 Data Preparation for XAIguiFormer")
    print("=" * 60)

    # Validate paths
    if not os.path.exists(PARTICIPANTS_FILE):
        print(f"ERROR: participants.tsv not found at {PARTICIPANTS_FILE}")
        sys.exit(1)
    if not os.path.isdir(DERIVATIVES_DIR):
        print(f"ERROR: derivatives directory not found at {DERIVATIVES_DIR}")
        sys.exit(1)

    # Load participants info
    participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
    print(f"\nParticipants file: {len(participants)} subjects")
    print(f"Groups: {participants['Group'].value_counts().to_dict()}")

    # Find available subjects in derivatives
    available_subs = sorted([
        d for d in os.listdir(DERIVATIVES_DIR)
        if os.path.isdir(os.path.join(DERIVATIVES_DIR, d)) and d.startswith('sub-')
    ])
    print(f"\nAvailable subjects in derivatives: {len(available_subs)}")
    print(f"  Subjects: {available_subs}")

    # Filter to relevant groups
    if USE_THREE_CLASSES:
        valid_groups = ['A', 'C', 'F']
    else:
        valid_groups = ['A', 'C']

    # Check which available subjects belong to valid groups
    valid_subs = []
    for sub_id in available_subs:
        row = participants[participants['participant_id'] == sub_id]
        if not row.empty and row['Group'].values[0] in valid_groups:
            valid_subs.append(sub_id)
        else:
            group = row['Group'].values[0] if not row.empty else 'unknown'
            print(f"  Skipping {sub_id} (group: {group}, not in {valid_groups})")

    print(f"\nValid subjects for classification: {len(valid_subs)}")

    # Check we have subjects from at least 2 classes
    sub_groups = []
    for sub_id in valid_subs:
        row = participants[participants['participant_id'] == sub_id]
        sub_groups.append(row['Group'].values[0])

    unique_groups = set(sub_groups)
    print(f"Groups represented: {unique_groups}")

    if len(unique_groups) < 2:
        print("\n" + "!" * 60)
        print("ERROR: Need subjects from at least 2 groups for classification!")
        print(f"Currently only have group(s): {unique_groups}")
        print()
        if 'C' not in unique_groups:
            print("You need to download HEALTHY CONTROL subjects (sub-037 to sub-065)")
        if 'A' not in unique_groups:
            print("You need to download ALZHEIMER'S subjects (sub-001 to sub-036)")
        print()
        print("To download more subjects, get the full ds004504 dataset from:")
        print("  https://openneuro.org/datasets/ds004504")
        print()
        print("Then place the subject folders in:")
        print(f"  {DERIVATIVES_DIR}/sub-XXX/eeg/")
        print("!" * 60)
        sys.exit(1)

    # Process each subject
    results = []
    for sub_id in valid_subs:
        sub_dir = os.path.join(DERIVATIVES_DIR, sub_id)
        set_file = find_set_file(sub_dir)
        if set_file is None:
            print(f"  WARNING: No .set file found for {sub_id}")
            continue

        result = process_subject(sub_id, set_file, participants)
        if result is not None:
            results.append(result)

    if len(results) < 3:
        print(f"\nERROR: Only {len(results)} subjects processed successfully. Need at least 3.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Successfully processed {len(results)} subjects")
    print(f"{'='*60}")

    # Split into train/val/test (stratified by group)
    groups = [r['group'] for r in results]
    sub_ids = [r['sub_id'] for r in results]

    # Check if we have enough samples per class for stratified split
    from collections import Counter
    group_counts = Counter(groups)
    print(f"Group distribution: {dict(group_counts)}")

    min_count = min(group_counts.values())
    if min_count < 2:
        print("WARNING: Some groups have fewer than 2 subjects. Using random split instead of stratified.")
        stratify = None
    else:
        stratify = groups

    # First split: 80% train+val, 20% test
    try:
        train_val_ids, test_ids, train_val_groups, _ = train_test_split(
            list(range(len(results))), groups,
            test_size=0.2, stratify=stratify, random_state=42
        )
    except ValueError:
        # Not enough samples for stratified split
        train_val_ids, test_ids = train_test_split(
            list(range(len(results))),
            test_size=0.2, random_state=42
        )
        train_val_groups = [groups[i] for i in train_val_ids]

    # Second split: train and val from train+val
    tv_stratify = train_val_groups if stratify is not None and len(set(train_val_groups)) > 1 else None
    try:
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=0.25, stratify=tv_stratify, random_state=42  # 0.25 of 0.8 = 0.2
        )
    except ValueError:
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=0.25, random_state=42
        )

    print(f"\nSplit: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Save all subjects
    output_dir = OUTPUT_ROOT
    print(f"\nSaving to: {output_dir}")

    for idx in train_ids:
        save_subject(results[idx], 'train', output_dir)
    for idx in val_ids:
        save_subject(results[idx], 'val', output_dir)
    for idx in test_ids:
        save_subject(results[idx], 'test', output_dir)

    # Also create processed directory (will be auto-populated by EEGBenchmarkDataset)
    processed_dir = os.path.join(output_dir, 'ds004504', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Switch to XAIguiFormer environment: conda activate XAIguiFormer")
    print(f"  2. Run training: python main.py --dataset ds004504")


if __name__ == '__main__':
    main()
