# baselines/baseline_wav2vec2.py
"""
VoxShield - Baseline A: Frozen wav2vec2 + MLP Classifier (Phase 5)
-----------------------------------------------------------------
This script trains a baseline audio deepfake detector using a completely frozen
wav2vec2-base model as an encoder, followed by average pooling and an MLP classifier.

To run training in Google Colab or terminal:
    python baselines/baseline_wav2vec2.py

Command-line usage context:
    !python baselines/baseline_wav2vec2.py
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model
import numpy as np

# Add parent directory to path so we can import config, dataset, and evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from dataset import ASVspoofDataset
from evaluate import compute_eer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

class Wav2Vec2MLPClassifier(nn.Module):
    def __init__(self, wav2vec_model_name=config.WAV2VEC_MODEL, num_classes=config.NUM_CLASSES):
        super(Wav2Vec2MLPClassifier, self).__init__()
        # Load pre-trained wav2vec2 model
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        
        # Freeze all wav2vec parameters
        for param in self.wav2vec.parameters():
            param.requires_grad = False
            
        # Classifier MLP head
        self.mlp = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, NumSamples]
        # wav2vec expects raw mono waveform
        with torch.no_grad():
            outputs = self.wav2vec(x)
            # last_hidden_state shape: [Batch, TimeFrames, 768]
            feats = outputs.last_hidden_state
            
            # Average pool across the time dimension to get a single vector per clip
            pooled = torch.mean(feats, dim=1)  # shape: [Batch, 768]
            
        logits = self.mlp(pooled)
        return logits

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    # Ensure wav2vec is kept in eval mode so Batch Normalization/Dropout behaves correctly
    model.wav2vec.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (waveforms, _, labels, _) in enumerate(dataloader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%")
            
    return total_loss / len(dataloader), correct / total

def evaluate_model(model, dataloader):
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, _, labels, _ in dataloader:
            waveforms = waveforms.to(device)
            logits = model(waveforms)
            
            # Score represents probability of "bonafide" (class 0)
            # High score means more likely real, low score means more likely spoof (fake)
            probs = torch.softmax(logits, dim=1)
            bonafide_probs = probs[:, 0].cpu().numpy()
            
            all_scores.extend(bonafide_probs)
            all_labels.extend(labels.numpy())
            
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Separate scores for EER computation
    bonafide_scores = all_scores[all_labels == 0]
    spoof_scores = all_scores[all_labels == 1]
    
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    return eer, threshold

def main():
    torch.manual_seed(config.SEED)
    
    # 1. Create Datasets and Dataloaders
    print("[Info] Loading datasets...")
    train_dataset = ASVspoofDataset(config.TRAIN_2019_CSV)
    dev_dataset = ASVspoofDataset(config.DEV_2019_CSV)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Instantiate Model, Optimizer, Criterion
    print("[Info] Initializing Baseline A model (Frozen wav2vec2)...")
    model = Wav2Vec2MLPClassifier().to(device)
    
    # Only optimize the parameters that require gradients (the MLP classifier head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    best_eer = float("inf")
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "baseline_wav2vec2_best.pt")
    
    # 3. Training Loop
    print(f"[Info] Starting training for {config.EPOCHS} epochs...")
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch} Training Summary | Loss: {train_loss:.4f} | Acc: {100. * train_acc:.2f}%")
        
        print("[Info] Running evaluation on dev set...")
        eer, threshold = evaluate_model(model, dev_loader)
        print(f"Epoch {epoch} Evaluation Summary | Dev EER: {eer:.4f}% | Decision Threshold: {threshold:.6f}")
        
        # Save best model
        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Success] New best Dev EER! Saved checkpoint to: {checkpoint_path}")
            
    print(f"\n[Finished] Training complete. Best Dev EER achieved: {best_eer:.4f}%")

if __name__ == "__main__":
    main()
