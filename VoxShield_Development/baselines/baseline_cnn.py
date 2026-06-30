# baselines/baseline_cnn.py
"""
VoxShield - Baseline B: LFCC/Mel Spectrogram + 2D CNN Classifier (Phase 5)
--------------------------------------------------------------------------
This script trains a baseline audio deepfake detector using 2D CNN blocks that process
the 2D spectro-temporal features (default: LFCC) extracted from raw waveforms.

**Crash-Resilient**: Automatically saves full training state after every epoch and
backs up to Google Drive (on Colab). On restart, resumes from the last completed epoch.

To run training in Google Colab or terminal:
    python baselines/baseline_cnn.py

Command-line usage context:
    !python baselines/baseline_cnn.py
"""

import sys
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add parent directory to path so we can import config, dataset, and evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from dataset import ASVspoofDataset
from evaluate import compute_eer
from features import FeatureExtractor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

# ==============================================================================
# Google Drive backup paths (used only when running on Colab)
# ==============================================================================
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/VoxShield_Checkpoints"

def backup_to_drive(local_path, filename=None):
    """Copies a local checkpoint file to Google Drive for persistence across Colab sessions."""
    if not os.path.exists("/content"):
        return  # Not running on Colab, skip
    if not os.path.exists("/content/drive"):
        print("[Backup] Google Drive not mounted. Skipping backup.")
        return
    os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
    if filename is None:
        filename = os.path.basename(local_path)
    dst = os.path.join(DRIVE_CHECKPOINT_DIR, filename)
    try:
        shutil.copy2(local_path, dst)
        print(f"[Backup] Saved checkpoint to Google Drive: {dst}")
    except Exception as e:
        print(f"[Backup] Warning: Could not backup to Drive ({e}). Training continues.")

def restore_from_drive(local_path, filename=None):
    """Restores a checkpoint from Google Drive to local SSD if it exists."""
    if not os.path.exists("/content"):
        return  # Not on Colab
    if filename is None:
        filename = os.path.basename(local_path)
    src = os.path.join(DRIVE_CHECKPOINT_DIR, filename)
    if os.path.exists(src) and not os.path.exists(local_path):
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            shutil.copy2(src, local_path)
            print(f"[Restore] Copied checkpoint from Google Drive: {src} → {local_path}")
        except Exception as e:
            print(f"[Restore] Warning: Could not restore from Drive ({e}).")

# ==============================================================================
# Model Definition
# ==============================================================================
class SimpleCNNClassifier(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(SimpleCNNClassifier, self).__init__()
        
        # 2D CNN Blocks
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1, config.CNN_NUM_CHANNELS[0], kernel_size=config.CNN_KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(config.CNN_NUM_CHANNELS[0]),
            nn.ReLU(),
            nn.MaxPool2d(config.CNN_POOL_SIZE),
            
            # Block 2
            nn.Conv2d(config.CNN_NUM_CHANNELS[0], config.CNN_NUM_CHANNELS[1], kernel_size=config.CNN_KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(config.CNN_NUM_CHANNELS[1]),
            nn.ReLU(),
            nn.MaxPool2d(config.CNN_POOL_SIZE)
        )
        
        # Classifier Head
        self.flatten = nn.Flatten()
        self.fc = None  # Lazy initialized or calculated in forward pass

    def forward(self, x):
        # x shape: [Batch, Channels=1, F, T]
        features = self.conv_blocks(x)
        flattened = self.flatten(features)
        
        # Initialize linear layer on the first run
        if self.fc is None:
            in_features = flattened.shape[1]
            self.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, config.NUM_CLASSES)
            ).to(x.device)
            
        logits = self.fc(flattened)
        return logits

# ==============================================================================
# Training & Evaluation Functions
# ==============================================================================
def train_one_epoch(model, dataloader, feature_extractor, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize PyTorch GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    for batch_idx, (waveforms, _, labels, _) in enumerate(dataloader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        # GPU spectral feature extraction
        with torch.no_grad():
            feats = feature_extractor(waveforms)
            
        optimizer.zero_grad()
        
        # Runs forward pass under Autocast
        with torch.cuda.amp.autocast():
            logits = model(feats)
            loss = criterion(logits, labels)
            
        # Scales the loss and performs backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%")
            
    return total_loss / len(dataloader), correct / total

def evaluate_model(model, dataloader, feature_extractor):
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, _, labels, _ in dataloader:
            waveforms = waveforms.to(device)
            # GPU spectral feature extraction
            feats = feature_extractor(waveforms)
            
            with torch.cuda.amp.autocast():
                logits = model(feats)
            
            # Score represents probability of "bonafide" (class 0)
            probs = torch.softmax(logits, dim=1)
            bonafide_probs = probs[:, 0].cpu().numpy()
            
            all_scores.extend(bonafide_probs)
            all_labels.extend(labels.numpy())
            
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    bonafide_scores = all_scores[all_labels == 0]
    spoof_scores = all_scores[all_labels == 1]
    
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    return eer, threshold

# ==============================================================================
# Checkpoint Save / Load
# ==============================================================================
def save_training_checkpoint(epoch, model, optimizer, best_eer, resume_path):
    """Saves the full training state so training can be resumed after a crash."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eer": best_eer,
    }
    torch.save(checkpoint, resume_path)
    print(f"[Checkpoint] Saved training state (epoch {epoch}) to: {resume_path}")
    # Backup to Google Drive
    backup_to_drive(resume_path)

def load_training_checkpoint(resume_path, model, optimizer):
    """Loads a saved training checkpoint. Returns (start_epoch, best_eer) or (1, inf) if none found."""
    # Try restoring from Google Drive first (in case local SSD was wiped)
    restore_from_drive(resume_path)
    
    if not os.path.exists(resume_path):
        return 1, float("inf")
    
    try:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # Resume from the NEXT epoch
        best_eer = checkpoint["best_eer"]
        print(f"[Resume] ✅ Loaded checkpoint from epoch {checkpoint['epoch']}. Resuming from epoch {start_epoch}.")
        print(f"[Resume] Best EER so far: {best_eer:.4f}%")
        return start_epoch, best_eer
    except Exception as e:
        print(f"[Resume] Warning: Could not load checkpoint ({e}). Starting fresh.")
        return 1, float("inf")

# ==============================================================================
# Main
# ==============================================================================
def main():
    torch.manual_seed(config.SEED)
    
    # 1. Create Datasets and Dataloaders
    print("[Info] Loading datasets...")
    train_dataset = ASVspoofDataset(config.TRAIN_2019_CSV, return_feats=False)
    dev_dataset = ASVspoofDataset(config.DEV_2019_CSV, return_feats=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 2. Instantiate Model, Feature Extractor, Optimizer, Criterion
    print(f"[Info] Initializing Baseline B model (2D CNN processing {config.FEATURE_TYPE.upper()}) & GPU feature extractor...")
    model = SimpleCNNClassifier().to(device)
    feature_extractor = FeatureExtractor(config.FEATURE_TYPE).to(device)
    
    # Trigger model dry-run to instantiate lazy fc layer
    dummy_input = torch.randn(1, 1, 20 if config.FEATURE_TYPE == "lfcc" else 80, 401).to(device)
    _ = model(dummy_input)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    best_eer = float("inf")
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "baseline_cnn_best.pt")
    resume_path = os.path.join(config.CHECKPOINT_DIR, "baseline_cnn_resume.pt")
    
    # 3. Attempt to resume from a previous checkpoint
    start_epoch, best_eer = load_training_checkpoint(resume_path, model, optimizer)
    
    # Also restore best model checkpoint from Drive if needed
    restore_from_drive(checkpoint_path)
    
    if start_epoch > config.EPOCHS:
        print(f"[Info] Training already completed (all {config.EPOCHS} epochs done). Nothing to do.")
        return
    
    # 4. Training Loop (resumes from start_epoch)
    print(f"[Info] Starting training for epochs {start_epoch}–{config.EPOCHS} (total {config.EPOCHS})...")
    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, feature_extractor, optimizer, criterion)
        print(f"Epoch {epoch} Training Summary | Loss: {train_loss:.4f} | Acc: {100. * train_acc:.2f}%")
        
        print("[Info] Running evaluation on dev set...")
        eer, threshold = evaluate_model(model, dev_loader, feature_extractor)
        print(f"Epoch {epoch} Evaluation Summary | Dev EER: {eer:.4f}% | Decision Threshold: {threshold:.6f}")
        
        # Save best model (unchanged format for inference compatibility)
        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Success] New best Dev EER! Saved checkpoint to: {checkpoint_path}")
            backup_to_drive(checkpoint_path)
        
        # Save full training state for crash recovery (EVERY epoch)
        save_training_checkpoint(epoch, model, optimizer, best_eer, resume_path)
            
    print(f"\n[Finished] Training complete. Best Dev EER achieved: {best_eer:.4f}%")

if __name__ == "__main__":
    main()
