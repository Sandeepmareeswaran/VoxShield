# baselines/baseline_cnn.py
"""
VoxShield - Baseline B: LFCC/Mel Spectrogram + 2D CNN Classifier (Phase 5)
--------------------------------------------------------------------------
This script trains a baseline audio deepfake detector using 2D CNN blocks that process
the 2D spectro-temporal features (default: LFCC) extracted from raw waveforms.

To run training in Google Colab or terminal:
    python baselines/baseline_cnn.py

Command-line usage context:
    !python baselines/baseline_cnn.py
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add parent directory to path so we can import config, dataset, and evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from dataset import ASVspoofDataset
from evaluate import compute_eer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

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
        # LFCC is 20 coefficients. After 2 MaxPool2d (size 2), the frequency axis (F=20) is 20 // 4 = 5.
        # Hop length is 160. For 4s (64,000 samples), 64000 // 160 = 400 frames (+1).
        # After 2 MaxPool2d (size 2), the time axis (T=401) is 401 // 4 = 100.
        # So shape is config.CNN_NUM_CHANNELS[1] * 5 * 100
        # Let's dynamically calculate features to make it robust to different dimensions (e.g. Mel vs LFCC).
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

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (_, feats, labels, _) in enumerate(dataloader):
        feats = feats.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(feats)
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
        for _, feats, labels, _ in dataloader:
            feats = feats.to(device)
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

def main():
    torch.manual_seed(config.SEED)
    
    # 1. Create Datasets and Dataloaders
    print("[Info] Loading datasets...")
    train_dataset = ASVspoofDataset(config.TRAIN_2019_CSV)
    dev_dataset = ASVspoofDataset(config.DEV_2019_CSV)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Instantiate Model, Optimizer, Criterion
    print(f"[Info] Initializing Baseline B model (2D CNN processing {config.FEATURE_TYPE.upper()})...")
    model = SimpleCNNClassifier().to(device)
    
    # Trigger model dry-run to instantiate lazy fc layer
    dummy_input = torch.randn(1, 1, 20 if config.FEATURE_TYPE == "lfcc" else 80, 401).to(device)
    _ = model(dummy_input)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    best_eer = float("inf")
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "baseline_cnn_best.pt")
    
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
