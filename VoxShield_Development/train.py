# train.py
"""
VoxShield - Training Pipeline Module (Phase 6)
----------------------------------------------
This script executes the two-stage end-to-end training pipeline:
- Stage 1 (Epoch 1 to FREEZE_EPOCHS): Keep wav2vec2 backbone frozen, train other components.
- Stage 2 (Epoch FREEZE_EPOCHS+1 onwards): Unfreeze top transformer layers of wav2vec2
  and fine-tune with a low learning rate.

To run training in Google Colab or terminal:
    python train.py

Command-line usage context (for Colab / Local terminal):
    !python train.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import ASVspoofDataset
from evaluate import compute_eer
from models.voxshield import VoxShield

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

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

def train_one_epoch(model, dataloader, optimizer, criterion, freeze_ssl=True):
    model.train()
    if freeze_ssl:
        model.wav2vec_branch.wav2vec.eval()  # Keep wav2vec batchnorm/dropout frozen
        
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize PyTorch GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    for batch_idx, (waveforms, spec_feats, labels, _) in enumerate(dataloader):
        waveforms = waveforms.to(device)
        spec_feats = spec_feats.to(device)
        labels = labels.to(device)
        
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

def evaluate_model(model, dataloader, freeze_ssl=True):
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, spec_feats, labels, _ in dataloader:
            waveforms = waveforms.to(device)
            spec_feats = spec_feats.to(device)
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

def main():
    torch.manual_seed(config.SEED)
    
    # 1. Load Data
    print("[Info] Loading datasets...")
    train_dataset = ASVspoofDataset(config.TRAIN_2019_CSV)
    dev_dataset = ASVspoofDataset(config.DEV_2019_CSV)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Instantiate Model
    print("[Info] Initializing VoxShield model...")
    model = VoxShield().to(device)
    
    # 3. Setup Criterion and Optimizer (Stage 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, stage=1)
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    best_eer = float("inf")
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "voxshield_best.pt")
    
    is_w2v_frozen = True
    
    # 4. Training loop
    print(f"[Info] Starting end-to-end VoxShield training for {config.EPOCHS} epochs...")
    for epoch in range(1, config.EPOCHS + 1):
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
            optimizer, 
            criterion, 
            freeze_ssl=is_w2v_frozen
        )
        print(f"Epoch {epoch} Training Summary | Loss: {train_loss:.4f} | Acc: {100. * train_acc:.2f}%")
        
        # Eval
        print("[Info] Evaluating on validation set...")
        eer, threshold = evaluate_model(model, dev_loader, freeze_ssl=is_w2v_frozen)
        print(f"Epoch {epoch} Evaluation Summary | Dev EER: {eer:.4f}% | Decision Threshold: {threshold:.6f}")
        
        scheduler.step()
        
        # Save check point
        if eer < best_eer:
            best_eer = eer
            # Save complete state dict
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Success] New best Dev EER achieved! Saved checkpoint to: {checkpoint_path}")
            
    print(f"\n[Finished] Training complete! Best Dev EER: {best_eer:.4f}%")

if __name__ == "__main__":
    main()
