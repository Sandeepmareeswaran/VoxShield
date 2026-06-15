# predict.py
"""
VoxShield - Inference and Prediction Module (Phase 7)
-----------------------------------------------------
This script loads the trained VoxShield model and temperature calibration scalar,
runs inference on a target audio file, and outputs:
- Real/Fake (bonafide/spoof) status
- Calibrated spoof probability
- Calibrated Risk Band: Green (Low), Amber (Medium), Red (High)
- Frame-level suspicion map (using attention weights)

To run inference:
    python predict.py --audio_path path_to_file.flac

Command-line usage context (for Colab / Local terminal):
    !python predict.py --audio_path d:/Project\ Work\ phase\ 1/VoxShield/Dataset/LA/ASVspoof2019_LA_dev/flac/LA_D_1048473.flac
"""

import os
import argparse
import torch
import torchaudio
import numpy as np

import config
import features
from models.voxshield import VoxShield

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_single_file(filepath):
    """Loads and preprocesses a single audio file to match training inputs."""
    waveform, sr = torchaudio.load(filepath)
    
    # 1. Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 2. Resample to 16 kHz if necessary
    if sr != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=config.SAMPLE_RATE)
        waveform = resampler(waveform)
        
    waveform = waveform.squeeze(0)
    
    # 3. Crop/Pad to exactly 4s (64,000 samples)
    num_samples = waveform.shape[0]
    target_length = config.NUM_SAMPLES
    if num_samples >= target_length:
        waveform = waveform[:target_length]
    else:
        num_pad = target_length - num_samples
        # Use repeat padding
        repeats = (target_length // num_samples) + 1
        waveform = waveform.repeat(repeats)[:target_length]
        
    # 4. Amplitude Normalization (Zero-mean, Unit-variance)
    mean = waveform.mean()
    std = waveform.std()
    waveform = (waveform - mean) / (std + 1e-9)
    
    return waveform

def get_risk_band(prob):
    """Maps calibrated spoof probability to a risk band."""
    if prob < 0.15:
        return "GREEN (Low Risk)", "\033[92m"  # Green text in terminal
    elif prob < 0.75:
        return "AMBER (Medium Risk)", "\033[93m"  # Yellow text
    else:
        return "RED (High Risk)", "\033[91m"  # Red text

def main():
    parser = argparse.ArgumentParser(description="VoxShield Inference Script")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to input audio file (.flac or .wav)")
    args = parser.parse_args()
    
    # 1. Load model checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "voxshield_best.pt")
    if not os.path.exists(checkpoint_path):
        print(f"[Error] No trained checkpoint found at {checkpoint_path}. Please train the model first.")
        return
        
    print("[Info] Loading model architecture and weights...")
    model = VoxShield().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 2. Load Temperature Calibration Parameter
    temp_path = os.path.join(config.CHECKPOINT_DIR, "temperature.txt")
    temperature = 1.0
    if os.path.exists(temp_path):
        with open(temp_path, "r") as f:
            temperature = float(f.read().strip())
        print(f"[Info] Loaded temperature calibration factor T: {temperature:.4f}")
    else:
        print("[Warning] No temperature calibration file found. Proceeding with uncalibrated logits.")
        
    # 3. Preprocess audio
    print(f"[Info] Preprocessing audio file: {args.audio_path}")
    try:
        waveform = load_and_preprocess_single_file(args.audio_path)
    except Exception as e:
        print(f"[Error] Failed to load/preprocess audio file: {e}")
        return
        
    # Add batch dimensions
    waveform = waveform.unsqueeze(0).to(device)  # shape: [1, 64000]
    spec_feats = features.extract_features(waveform, feature_type=config.FEATURE_TYPE).to(device)  # shape: [1, 1, F, T]
    
    # 4. Forward pass (returning attention weights for suspicion map)
    with torch.no_grad():
        logits, weights = model(waveform, spec_feats, freeze_ssl=True, return_attention=True)
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        
        spoof_prob = probs[0, 1].item()  # Probability of class 1 (spoof)
        
    # 5. Determine label and risk band
    label = "SPOOF (Fake)" if spoof_prob >= 0.5 else "BONAFIDE (Genuine)"
    risk_band, color_code = get_risk_band(spoof_prob)
    
    print("\n==========================================================")
    print("                    VOXSHIELD REPORT                      ")
    print("==========================================================")
    print(f"File Path: {args.audio_path}")
    print(f"Result   : {label}")
    print(f"Confidence (Spoof Probability): {spoof_prob * 100:.2f}%")
    print(f"Risk Band: {color_code}{risk_band}\033[0m")
    
    # 6. Suspect frames / Suspicion Map visualization
    # weights shape: [1, T_ssl, 1] -> representing importance of each sequence step
    weights = weights.squeeze(0).squeeze(1).cpu().numpy()
    
    # Find top 3 most suspicious time segments (where model focused most)
    # wav2vec output length is usually 199 steps for 4 seconds of audio. Each step is ~20ms.
    # We map indices to seconds.
    total_steps = len(weights)
    step_duration = config.DURATION_SEC / total_steps
    
    top_indices = np.argsort(weights)[::-1][:3]
    print("\nSuspicion Map - Top 3 segments where spoofing is suspected:")
    for rank, idx in enumerate(top_indices):
        start_time = idx * step_duration
        end_time = (idx + 1) * step_duration
        importance = weights[idx] * 100
        print(f"  {rank+1}. Time: {start_time:.2f}s - {end_time:.2f}s (Attention Weight: {importance:.2f}%)")
    print("==========================================================\n")

if __name__ == "__main__":
    main()
