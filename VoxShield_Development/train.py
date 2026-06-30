# train.py
"""
VoxShield - Training Pipeline Module (Phase 6)
----------------------------------------------
This script executes the two-stage end-to-end training pipeline:
- Stage 1 (Epoch 1 to FREEZE_EPOCHS): Keep wav2vec2 backbone frozen, train other components.
- Stage 2 (Epoch FREEZE_EPOCHS+1 onwards): Unfreeze top transformer layers of wav2vec2
  and fine-tune with a low learning rate.

**Crash-Resilient**: Automatically saves full training state (epoch, model, optimizer,
scheduler, stage flag, best EER) after every epoch and backs up to Google Drive (on Colab).
On restart, resumes from the last completed epoch with correct training stage.

To run training in Google Colab or terminal:
    python train.py

Command-line usage context (for Colab / Local terminal):
    !python train.py
"""

import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import ASVspoofDataset
from evaluate import compute_eer
from features import FeatureExtractor
from models.voxshield import VoxShield

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
# Stage Management
# ==============================================================================
def unfreeze_top_w2v_layers(model):
    """Unfreezes the top N transformer layers and final LayerNorm of the wav2vec2 encoder."""
    print(f"[Info] Unfreezing the top {config.UNFREEZE_LAYERS} layers of wav2vec2 backbone...")
    
    # 1. Access encoder layers
    encoder_layers = model.wav2vec_branch.wav2vec.encoder.layers
    total_layers = len(encoder_layers)
    
    # Unfreeze the top layers
    for i in range(total_layers - config.UNFREEZE_LAYERS, total_layers):
        for param in encoder_layers[i].parameters():
            param.requires_grad = True
            
    # 2. Unfreeze the final LayerNorm of the encoder
    for param in model.wav2vec_branch.wav2vec.encoder.layer_norm.parameters():
        param.requires_grad = True
        
    print("[Info] Unfreezing complete.")

def get_optimizer(model, stage=1):
    """Returns optimizer tailored for Stage 1 or Stage 2 training parameters."""
    if stage == 1:
        # Stage 1: Only train non-backbone parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=config.LR, 
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        # Stage 2: Train unfrozen wav2vec2 parameters with lower LR, and rest with higher LR
        w2v_params = []
        rest_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "wav2vec_branch.wav2vec" in name:
                    w2v_params.append(param)
                else:
                    rest_params.append(param)
                    
        optimizer = torch.optim.AdamW([
            {"params": w2v_params, "lr": config.WAV2VEC_LR},
            {"params": rest_params, "lr": config.LR}
        ], weight_decay=config.WEIGHT_DECAY)
        
    return optimizer

# ==============================================================================
# Training & Evaluation Functions
# ==============================================================================
def train_one_epoch(model, dataloader, feature_extractor, optimizer, criterion, freeze_ssl=True):
    model.train()
    if freeze_ssl:
        model.wav2vec_branch.wav2vec.eval()  # Keep wav2vec batchnorm/dropout frozen
        
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
            spec_feats = feature_extractor(waveforms)
            
        optimizer.zero_grad()
        
        # Runs forward pass under Autocast
        with torch.cuda.amp.autocast():
            logits = model(waveforms, spec_feats, freeze_ssl=freeze_ssl)
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

def evaluate_model(model, dataloader, feature_extractor, freeze_ssl=True):
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, _, labels, _ in dataloader:
            waveforms = waveforms.to(device)
            # GPU spectral feature extraction
            spec_feats = feature_extractor(waveforms)
            
            with torch.cuda.amp.autocast():
                logits = model(waveforms, spec_feats, freeze_ssl=freeze_ssl)
            
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
def save_training_checkpoint(epoch, model, optimizer, scheduler, best_eer, is_w2v_frozen, resume_path):
    """Saves the full training state so training can be resumed after a crash."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_eer": best_eer,
        "is_w2v_frozen": is_w2v_frozen,
    }
    torch.save(checkpoint, resume_path)
    print(f"[Checkpoint] Saved training state (epoch {epoch}, stage={'frozen' if is_w2v_frozen else 'fine-tune'}) to: {resume_path}")
    # Backup to Google Drive
    backup_to_drive(resume_path)

def load_training_checkpoint(resume_path, model):
    """
    Loads a saved training checkpoint.
    Returns a dict with all state, or None if no checkpoint found.
    
    NOTE: Optimizer and scheduler are NOT loaded here because they need to be
    reconstructed based on the training stage (frozen vs fine-tuning).
    Model weights ARE loaded here.
    """
    # Try restoring from Google Drive first (in case local SSD was wiped)
    restore_from_drive(resume_path)
    
    if not os.path.exists(resume_path):
        return None
    
    try:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Resume] ✅ Loaded checkpoint from epoch {checkpoint['epoch']}.")
        print(f"[Resume] Best EER so far: {checkpoint['best_eer']:.4f}%")
        print(f"[Resume] Training stage: {'Stage 1 (Frozen SSL)' if checkpoint['is_w2v_frozen'] else 'Stage 2 (Fine-tuning)'}")
        return checkpoint
    except Exception as e:
        print(f"[Resume] Warning: Could not load checkpoint ({e}). Starting fresh.")
        return None

# ==============================================================================
# Main
# ==============================================================================
def main():
    torch.manual_seed(config.SEED)
    
    # 1. Load Data
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
    
    # 2. Instantiate Model and Feature Extractor
    print("[Info] Initializing VoxShield model & GPU feature extractor...")
    model = VoxShield().to(device)
    feature_extractor = FeatureExtractor(config.FEATURE_TYPE).to(device)
    
    # 3. Attempt to resume from a previous checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "voxshield_best.pt")
    resume_path = os.path.join(config.CHECKPOINT_DIR, "voxshield_resume.pt")
    
    # Also restore best model checkpoint from Drive if needed
    restore_from_drive(checkpoint_path)
    
    saved_checkpoint = load_training_checkpoint(resume_path, model)
    
    if saved_checkpoint is not None:
        start_epoch = saved_checkpoint["epoch"] + 1
        best_eer = saved_checkpoint["best_eer"]
        is_w2v_frozen = saved_checkpoint["is_w2v_frozen"]
        
        # Reconstruct correct training stage
        if not is_w2v_frozen:
            # We were in Stage 2 — need to unfreeze layers before creating optimizer
            unfreeze_top_w2v_layers(model)
            optimizer = get_optimizer(model, stage=2)
            scheduler = CosineAnnealingLR(optimizer, T_max=(config.EPOCHS - config.FREEZE_EPOCHS))
        else:
            optimizer = get_optimizer(model, stage=1)
            scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
        
        # Restore optimizer and scheduler state
        try:
            optimizer.load_state_dict(saved_checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(saved_checkpoint["scheduler_state_dict"])
            print(f"[Resume] Optimizer and scheduler state restored successfully.")
        except Exception as e:
            print(f"[Resume] Warning: Could not restore optimizer/scheduler state ({e}). Using fresh optimizer.")
        
        print(f"[Resume] Resuming from epoch {start_epoch}.")
    else:
        start_epoch = 1
        best_eer = float("inf")
        is_w2v_frozen = True
        optimizer = get_optimizer(model, stage=1)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    if start_epoch > config.EPOCHS:
        print(f"[Info] Training already completed (all {config.EPOCHS} epochs done). Nothing to do.")
        return
    
    # 4. Setup Criterion
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training loop (resumes from start_epoch)
    print(f"[Info] Starting end-to-end VoxShield training for epochs {start_epoch}–{config.EPOCHS} (total {config.EPOCHS})...")
    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        
        # Check if we should transit to Stage 2 (unfreeze wav2vec)
        if epoch > config.FREEZE_EPOCHS and is_w2v_frozen:
            print("[Info] Stage 1 finished. Transitioning to Stage 2 (fine-tuning)...")
            unfreeze_top_w2v_layers(model)
            optimizer = get_optimizer(model, stage=2)
            scheduler = CosineAnnealingLR(optimizer, T_max=(config.EPOCHS - config.FREEZE_EPOCHS))
            is_w2v_frozen = False
            
        print(f"Current stage: {'Stage 1 (Frozen SSL)' if is_w2v_frozen else 'Stage 2 (Fine-tuning)'}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, 
            train_loader, 
            feature_extractor,
            optimizer, 
            criterion, 
            freeze_ssl=is_w2v_frozen
        )
        print(f"Epoch {epoch} Training Summary | Loss: {train_loss:.4f} | Acc: {100. * train_acc:.2f}%")
        
        # Eval
        print("[Info] Evaluating on validation set...")
        eer, threshold = evaluate_model(model, dev_loader, feature_extractor, freeze_ssl=is_w2v_frozen)
        print(f"Epoch {epoch} Evaluation Summary | Dev EER: {eer:.4f}% | Decision Threshold: {threshold:.6f}")
        
        scheduler.step()
        
        # Save best model (unchanged format for inference compatibility)
        if eer < best_eer:
            best_eer = eer
            # Save complete state dict
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Success] New best Dev EER achieved! Saved checkpoint to: {checkpoint_path}")
            backup_to_drive(checkpoint_path)
        
        # Save full training state for crash recovery (EVERY epoch)
        save_training_checkpoint(epoch, model, optimizer, scheduler, best_eer, is_w2v_frozen, resume_path)
            
    print(f"\n[Finished] Training complete! Best Dev EER: {best_eer:.4f}%")

if __name__ == "__main__":
    main()
