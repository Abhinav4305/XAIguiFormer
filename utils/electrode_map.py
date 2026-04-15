"""
EEG Electrode Mapping for BEED Dataset
=======================================
Maps BEED features X1-X16 to standard 10-20 electrode names,
2D scalp positions, and brain region labels.
"""

# Standard 16-channel 10-20 electrode subset
# Mapping assumes the common clinical montage order
ELECTRODE_NAMES = [
    'Fp1', 'Fp2',          # Prefrontal
    'F7', 'F3', 'Fz', 'F4', 'F8',  # Frontal
    'C3', 'C4',            # Central
    'T7', 'T8',            # Temporal
    'P3', 'Pz', 'P4',     # Parietal
    'O1', 'O2',            # Occipital
]

# Feature name → electrode name mapping
FEATURE_TO_ELECTRODE = {
    f'X{i+1}': name for i, name in enumerate(ELECTRODE_NAMES)
}

# Brain region for each electrode
BRAIN_REGIONS = {
    'Fp1': 'Frontal',  'Fp2': 'Frontal',
    'F7':  'Frontal',  'F3':  'Frontal', 'Fz': 'Frontal',
    'F4':  'Frontal',  'F8':  'Frontal',
    'C3':  'Central',  'C4':  'Central',
    'T7':  'Temporal', 'T8':  'Temporal',
    'P3':  'Parietal', 'Pz':  'Parietal', 'P4': 'Parietal',
    'O1':  'Occipital','O2':  'Occipital',
}

# 2D scalp positions for topographic plotting
# Coordinates are normalized to [-0.5, 0.5] range
# Origin (0,0) = center of head, +y = front (nose), +x = right
ELECTRODE_POSITIONS_2D = {
    'Fp1': (-0.15,  0.45),  'Fp2': ( 0.15,  0.45),
    'F7':  (-0.40,  0.25),  'F3':  (-0.18,  0.25),
    'Fz':  ( 0.00,  0.25),  'F4':  ( 0.18,  0.25),
    'F8':  ( 0.40,  0.25),
    'C3':  (-0.25,  0.00),  'C4':  ( 0.25,  0.00),
    'T7':  (-0.45,  0.00),  'T8':  ( 0.45,  0.00),
    'P3':  (-0.18, -0.25),  'Pz':  ( 0.00, -0.25),
    'P4':  ( 0.18, -0.25),
    'O1':  (-0.15, -0.45),  'O2':  ( 0.15, -0.45),
}

# Colors per brain region
REGION_COLORS = {
    'Frontal':   '#3498db',
    'Central':   '#2ecc71',
    'Temporal':  '#e67e22',
    'Parietal':  '#9b59b6',
    'Occipital': '#e74c3c',
}

CLASS_NAMES = {
    0: 'Healthy',
    1: 'Generalized Seizure',
    2: 'Focal Seizure',
    3: 'Seizure Events',
}


def get_electrode_info(feature_idx):
    """Get electrode name, position, and brain region for a feature index (0-15)."""
    name = ELECTRODE_NAMES[feature_idx]
    pos = ELECTRODE_POSITIONS_2D[name]
    region = BRAIN_REGIONS[name]
    color = REGION_COLORS[region]
    return name, pos, region, color


def get_all_positions():
    """Return arrays of x, y positions for all 16 electrodes in order."""
    xs = [ELECTRODE_POSITIONS_2D[e][0] for e in ELECTRODE_NAMES]
    ys = [ELECTRODE_POSITIONS_2D[e][1] for e in ELECTRODE_NAMES]
    return xs, ys
