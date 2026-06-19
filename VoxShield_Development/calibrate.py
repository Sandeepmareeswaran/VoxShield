# calibrate.py
"""
VoxShield - Logit Calibration Module (Phase 7)
----------------------------------------------
This script implements Temperature Scaling on the validation set (dev_2019)
to calibrate the model's logits, producing realistic probabilities.
It saves the calibrated temperature scalar parameter to `checkpoints/temperature.txt`.

To run calibration:
    python calibrate.py

Command-line usage context (for Colab / Local terminal):
    !python calibrate.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import config
from dataset import ASVspoofDataset
from features import FeatureExtractor
from models.voxshield import VoxShield

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_validation_logits_and_labels(model, dataloader, feature_extractor):
    """Gathers raw logits and labels from the validation dataset."""
    model.eval()
    all_logits = []
    all_labels = []
    
    print("[Info] Forwarding validation set through the model to gather logits...")
    with torch.no_grad():
        for waveforms, _, labels, _ in dataloader:
            waveforms = waveforms.to(device)
            # GPU spectral feature extraction
            spec_feats = feature_extractor(waveforms)
            
            # Use frozen-ssl mode for inference validation under Autocast
            with torch.cuda.amp.autocast():
                logits = model(waveforms, spec_feats, freeze_ssl=True)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            
    # Concatenate all lists into single tensors
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_logits, all_labels

class TemperatureScaler(nn.Module):
    """A simple wrapper to optimize a single parameter T (temperature) to scale logits."""
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        # Initialize temperature with 1.0
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        # Scale logits
        return logits / self.temperature

def calibrate_logits(logits, labels):
    """Optimizes temperature parameter T using LBFGS optimizer on NLL loss."""
    scaler = TemperatureScaler().to(logits.device)
    criterion = nn.CrossEntropyLoss()
    
    # Setup LBFGS optimizer (standard for calibration tasks)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    
    print("[Info] Optimizing temperature parameter T...")
    
    # We define a closure for the LBFGS optimizer
    def eval_closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_closure)
    
    calibrated_temp = scaler.temperature.item()
    print(f"[Success] Optimized temperature T: {calibrated_temp:.4f}")
    return calibrated_temp

def main():
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "voxshield_best.pt")
    if not os.path.exists(checkpoint_path):
        print(f"[Error] No trained checkpoint found at {checkpoint_path}. Please train the model first.")
        return
        
    # 1. Load Validation Data
    print("[Info] Loading validation set...")
    dev_dataset = ASVspoofDataset(config.DEV_2019_CSV, return_feats=False)
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 2. Load Model and Feature Extractor
    print("[Info] Loading trained VoxShield model checkpoint & GPU feature extractor...")
    model = VoxShield().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    feature_extractor = FeatureExtractor(config.FEATURE_TYPE).to(device)
    
    # 3. Gather Logits
    logits, labels = get_validation_logits_and_labels(model, dev_loader, feature_extractor)
    
    # 4. Calibrate
    # Move to GPU/CPU of choices
    logits = logits.to(device)
    labels = labels.to(device)
    
    T = calibrate_logits(logits, labels)
    
    # 5. Save temperature value
    temp_file = os.path.join(config.CHECKPOINT_DIR, "temperature.txt")
    with open(temp_file, "w") as f:
        f.write(str(T))
    print(f"[Success] Saved temperature calibration parameter to: {temp_file}")
    
    # Show example of calibration effect
    raw_probs = torch.softmax(logits[:5], dim=1)[:, 1].cpu().numpy()
    calib_probs = torch.softmax(logits[:5] / T, dim=1)[:, 1].cpu().numpy()
    
    print("\nCalibration sample results (Spoof probabilities):")
    for i in range(len(raw_probs)):
        print(f"  Sample {i+1} | Raw Prob: {raw_probs[i]:.4f} | Calibrated Prob: {calib_probs[i]:.4f} | Label: {labels[i].item()}")

if __name__ == "__main__":
    main()
