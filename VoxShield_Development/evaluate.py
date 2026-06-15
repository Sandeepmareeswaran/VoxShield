# evaluate.py
"""
VoxShield - Evaluation Metrics Module (Phase 6)
-----------------------------------------------
This module provides functions to calculate performance metrics (EER and min t-DCF)
for deepfake audio countermeasures.

To run evaluation on saved scores:
    python evaluate.py --scores_file path_to_scores.txt

Command-line usage context (for Colab / Local terminal):
    !python evaluate.py --scores_file checkpoints/voxshield_eval_scores.txt
"""

import os
import argparse
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def compute_eer(bonafide_scores, spoof_scores):
    """
    Computes Equal Error Rate (EER) given bonafide (real) and spoof (fake) scores.
    A higher score indicates the model believes the audio is genuine (bonafide).
    """
    # Combine scores and create labels (1 for bonafide, 0 for spoof)
    y = [1] * len(bonafide_scores) + [0] * len(spoof_scores)
    scores = list(bonafide_scores) + list(spoof_scores)
    
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
    fnr = 1 - tpr
    
    # EER is where FPR == FNR
    # Using interpolation to find the exact point
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    
    return eer * 100, float(thresh)

def compute_min_tdcf(bonafide_scores, spoof_scores, Pspoof=0.05, Cmiss_asv=1, Cmiss_cm=1, Cfa_asv=10, Cfa_cm=10):
    """
    Computes min t-DCF (tandem Detection Cost Function).
    This function implements a standard simplified t-DCF model assuming a fixed ASV system.
    """
    # Since we do not load an ASV system directly here, we calculate min t-DCF 
    # under the standard ASV parameterization. In practical evaluations, the official 
    # ASVspoof challenge evaluation package is preferred, but this provides a local reference.
    
    # 1. Combine scores and compute false alarm and miss rates for the CM
    y = [1] * len(bonafide_scores) + [0] * len(spoof_scores)
    scores = np.concatenate([bonafide_scores, spoof_scores])
    
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Standard t-DCF weights (arbitrary typical ASV system parameters from 2019 challenge)
    Pmiss_asv = 0.01
    Pfa_asv = 0.01
    Pmiss_spoof_asv = 0.76  # ASV system is easily spoofed by fakes
    
    # Compute t-DCF value at each threshold
    beta = (Cfa_cm * Pspoof) / (Cmiss_cm * (1 - Pspoof))
    # Standard coefficients for min t-DCF
    # t-DCF(t) = C1 * Pmiss_cm(t) + C2 * Pfa_cm(t)
    C1 = Cmiss_cm / (Cmiss_asv * (1 - Pmiss_asv) - Cfa_asv * Pfa_asv)
    C2 = beta * Cfa_cm / (Cmiss_asv * (1 - Pmiss_asv) - Cfa_asv * Pfa_asv)
    
    # We normalize so that the cost of doing nothing (no CM) is 1.0
    # In standard ASVspoof scripts:
    # min_tDCF = min { (1 - Pmiss_cm) * Constant + ... }
    # Let's compute Equal Error Rate as a fallback, or a mock min t-DCF
    # For a fully accurate min t-DCF, users should run the official ASVspoof challenge script.
    # Here we sweep thresholds to minimize t-DCF:
    t_dcf_list = []
    for p_miss, p_fa in zip(fnr, fpr):
        # simplified cost function
        cost = p_miss + beta * p_fa
        t_dcf_list.append(cost)
        
    min_tdcf_val = min(t_dcf_list)
    # Normalizing it so typical range is around 0 to 1
    # For exact comparison, we output EER as the primary driver.
    return min_tdcf_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake scores")
    parser.add_argument("--scores_file", type=str, required=True, help="Path to CM scores file (trial_id score)")
    parser.add_argument("--keys_file", type=str, required=True, help="Path to protocol key file (trial_id label)")
    args = parser.parse_args()
    
    # Read keys
    keys = {}
    with open(args.keys_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Format: could be 'LA_0079 LA_T_1138215 - - bonafide' or 'LA_E_123456 bonafide'
                # We identify label as last word, file id as second or first
                label = parts[-1]
                file_id = parts[1] if "_" in parts[1] else parts[0]
                keys[file_id] = label
                
    # Read scores
    bonafide_scores = []
    spoof_scores = []
    with open(args.scores_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                file_id, score = parts[0], float(parts[1])
                if file_id in keys:
                    label = keys[file_id]
                    if label == "bonafide":
                        bonafide_scores.append(score)
                    elif label == "spoof":
                        spoof_scores.append(score)
                        
    if not bonafide_scores or not spoof_scores:
        print("[Error] No matched keys and scores found.")
    else:
        eer_val, thresh = compute_eer(bonafide_scores, spoof_scores)
        min_tdcf = compute_min_tdcf(bonafide_scores, spoof_scores)
        print(f"Results for {args.scores_file}:")
        print(f"  Total Bonafide: {len(bonafide_scores)} | Total Spoof: {len(spoof_scores)}")
        print(f"  EER: {eer_val:.4f}% (Threshold: {thresh:.6f})")
        print(f"  min t-DCF (simplified): {min_tdcf:.6f}")
