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

# ==============================================================================
# 1. Directory and Dataset Paths
# ==============================================================================
# Absolute paths to raw datasets
DATASET_LA_DIR = r"d:\Project Work phase 1\VoxShield\Dataset\LA"
DATASET_2021_DIR = r"d:\Project Work phase 1\VoxShield\Dataset\ASVspoof2021_LA_eval\ASVspoof2021_LA_eval"

# Path to save generated manifest files and checkpoints
BASE_DEV_DIR = r"d:\Project Work phase 1\VoxShield\VoxShield_Development"
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
