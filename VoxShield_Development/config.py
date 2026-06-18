# config.py
"""
VoxShield - Configuration Module
--------------------------------
This module contains the global configuration, hyperparameters, and directory paths.
It is shared across all preprocessing, baseline, training, and evaluation scripts.

To review configurations or import:
    python -c "import config; print(config.SHARED_DIM)"
"""

import os

# Get the directory of config.py (VoxShield_Development)
BASE_DEV_DIR = os.path.dirname(os.path.abspath(__file__))

def find_dataset_la():
    # 1. Check local environment absolute path
    local_path = r"d:\Project Work phase 1\VoxShield\Dataset\LA"
    if os.path.exists(local_path):
        return local_path
        
    # 2. Check relative to code location (e.g., inside Dataset/LA)
    relative_path_1 = os.path.abspath(os.path.join(BASE_DEV_DIR, "..", "Dataset", "LA"))
    if os.path.exists(relative_path_1):
        return relative_path_1
        
    # 3. Check if LA is direct sibling of VoxShield_Development
    relative_path_2 = os.path.abspath(os.path.join(BASE_DEV_DIR, "..", "LA"))
    if os.path.exists(relative_path_2):
        return relative_path_2
        
    # 4. Search in Google Drive mounts
    drive_prefixes = [
        "/content/drive/MyDrive",
        "/content/drive/Shareddrives",
    ]
    for prefix in drive_prefixes:
        if os.path.exists(prefix):
            for root, dirs, files in os.walk(prefix):
                depth = root.replace(prefix, "").count(os.sep)
                if depth > 3:
                    continue
                for d in dirs:
                    if d.lower() == "la":
                        la_candidate = os.path.join(root, d)
                        # Ensure it contains protocols
                        if os.path.exists(os.path.join(la_candidate, "ASVspoof2019_LA_cm_protocols")):
                            return la_candidate
                        # Check also case-insensitive protocol folders
                        for subd in os.listdir(la_candidate):
                            if "protocol" in subd.lower():
                                return la_candidate
                            
    return local_path

def find_dataset_2021():
    # 1. Check local environment absolute path
    local_path = r"d:\Project Work phase 1\VoxShield\Dataset\ASVspoof2021_LA_eval\ASVspoof2021_LA_eval"
    if os.path.exists(local_path):
        return local_path
        
    # 2. Check relative to code location (nested ASVspoof2021_LA_eval)
    relative_path_1 = os.path.abspath(os.path.join(BASE_DEV_DIR, "..", "Dataset", "ASVspoof2021_LA_eval", "ASVspoof2021_LA_eval"))
    if os.path.exists(relative_path_1):
        return relative_path_1
        
    # 3. Check sibling with or without suffixes
    parent_dir = os.path.abspath(os.path.join(BASE_DEV_DIR, ".."))
    dataset_dir = os.path.join(parent_dir, "Dataset")
    search_dirs = [parent_dir]
    if os.path.exists(dataset_dir):
        search_dirs.append(dataset_dir)
        
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for item in os.listdir(s_dir):
                if "ASVspoof2021_LA_eval" in item:
                    candidate = os.path.join(s_dir, item)
                    if os.path.isdir(candidate):
                        nested_candidate = os.path.join(candidate, "ASVspoof2021_LA_eval")
                        if os.path.exists(nested_candidate):
                            return nested_candidate
                        return candidate
                    
    # 4. Search in Google Drive mounts
    drive_prefixes = [
        "/content/drive/MyDrive",
        "/content/drive/Shareddrives",
    ]
    for prefix in drive_prefixes:
        if os.path.exists(prefix):
            for root, dirs, files in os.walk(prefix):
                depth = root.replace(prefix, "").count(os.sep)
                if depth > 3:
                    continue
                for d in dirs:
                    if "ASVspoof2021_LA_eval" in d:
                        candidate = os.path.join(root, d)
                        nested_candidate = os.path.join(candidate, "ASVspoof2021_LA_eval")
                        if os.path.exists(nested_candidate):
                            return nested_candidate
                        return candidate
                        
    return local_path

# ==============================================================================
# 1. Directory and Dataset Paths
# ==============================================================================
# Absolute paths to raw datasets resolved dynamically
DATASET_LA_DIR = find_dataset_la()
DATASET_2021_DIR = find_dataset_2021()

# Path to save generated manifest files and checkpoints
MANIFEST_DIR = os.path.join(BASE_DEV_DIR, "manifests")
CHECKPOINT_DIR = os.path.join(BASE_DEV_DIR, "checkpoints")

os.makedirs(MANIFEST_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Manifest CSV file paths
TRAIN_2019_CSV = os.path.join(MANIFEST_DIR, "train_2019.csv")
DEV_2019_CSV = os.path.join(MANIFEST_DIR, "dev_2019.csv")
EVAL_2019_CSV = os.path.join(MANIFEST_DIR, "eval_2019.csv")
EVAL_2021_CSV = os.path.join(MANIFEST_DIR, "eval_2021.csv")

# ==============================================================================
# 2. Audio Preprocessing Parameters
# ==============================================================================
SAMPLE_RATE = 16000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)  # 64,000 samples

# ==============================================================================
# 3. Spectral Feature Parameters (LFCC / Log-Mel)
# ==============================================================================
# Options: "lfcc", "mel"
FEATURE_TYPE = "lfcc"

# LFCC configuration (torchaudio.transforms.LFCC)
LFCC_PARAMS = {
    "sample_rate": SAMPLE_RATE,
    "n_filter": 20,
    "n_lfcc": 20,
    "speckwargs": {
        "n_fft": 512,
        "win_length": 320,  # 20ms frame length
        "hop_length": 160,  # 10ms frame shift
    }
}

# Log-Mel spectrogram configuration
MEL_PARAMS = {
    "sample_rate": SAMPLE_RATE,
    "n_fft": 512,
    "win_length": 320,
    "hop_length": 160,
    "n_mels": 80
}

# ==============================================================================
# 4. Model Architecture Hyperparameters
# ==============================================================================
# Model dimensions
SHARED_DIM = 128            # Shared dimension D (projection target)
WAV2VEC_MODEL = "facebook/wav2vec2-base"

# CNN Front-end configuration
CNN_NUM_CHANNELS = [16, 32]  # Convolutional channel growth
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2

# Transformer block configuration
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_FF_DIM = 256
DROPOUT = 0.1

# Classification parameters
NUM_CLASSES = 2             # [bonafide, spoof]

# ==============================================================================
# 5. Training Strategy parameters
# ==============================================================================
SEED = 42
BATCH_SIZE = 2
EPOCHS = 12                 # Total training epochs (e.g. Stage 1 + Stage 2)
LR = 1e-4                   # General learning rate for new parameters
WEIGHT_DECAY = 1e-4

# Frozen vs Fine-tuning stages
FREEZE_EPOCHS = 6           # Stage 1: Keep wav2vec2 entirely frozen for N epochs
WAV2VEC_LR = 1e-5           # Stage 2: Unfreeze top layers of wav2vec2 with smaller LR
UNFREEZE_LAYERS = 2         # Number of top transformer layers of wav2vec2 to unfreeze in Stage 2
