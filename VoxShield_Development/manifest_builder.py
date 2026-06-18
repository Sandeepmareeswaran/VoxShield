# manifest_builder.py
"""
VoxShield - Manifest Builder Module (Phase 2)
---------------------------------------------
This script parses the raw protocol/keys files of ASVspoof 2019 LA and ASVspoof 2021 LA
and saves unified manifest CSV files. If the ASVspoof 2021 keys are missing, it
automatically downloads them from the official website.

To execute this script:
    python manifest_builder.py

Command-line usage context (for Colab / Local terminal):
    !python manifest_builder.py
"""

import os
import urllib.request
import tarfile
import ssl
import pandas as pd
import config

# Disable SSL verification for urllib (necessary for some local environments)
ssl._create_default_https_context = ssl._create_unverified_context

def find_subfolder(parent, prefix):
    if not os.path.exists(parent):
        return os.path.join(parent, prefix)
    for d in os.listdir(parent):
        if prefix.lower() in d.lower() and os.path.isdir(os.path.join(parent, d)):
            return os.path.join(parent, d)
    return os.path.join(parent, prefix)

def download_and_extract_2021_keys():
    """Downloads and extracts the official 2021 evaluation keys if they are not present."""
    target_dir = config.DATASET_2021_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    # Try recursively searching for trial_metadata.txt or similar, prioritizing CM folder
    metadata_path = None
    candidates = []
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if "trial_metadata" in f and f.endswith(".txt"):
                full_p = os.path.join(root, f)
                normalized_p = full_p.replace('\\', '/')
                if "la/cm" in normalized_p.lower():
                    metadata_path = full_p
                    break
                candidates.append(full_p)
        if metadata_path:
            break
            
    if not metadata_path and candidates:
        metadata_path = candidates[0]
            
    if metadata_path:
        print(f"[Info] Found ASVspoof 2021 keys at: {metadata_path}")
        return metadata_path
        
    tar_path = os.path.join(target_dir, "LA-keys-full.tar.gz")
    if not os.path.exists(tar_path):
        url = "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz"
        print(f"[Info] Downloading ASVspoof 2021 keys from: {url}")
        urllib.request.urlretrieve(url, tar_path)
        print("[Info] Download complete.")
        
    print(f"[Info] Extracting keys from {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print("[Info] Extraction complete.")
    
    # Check if we can locate the metadata file now, prioritizing CM
    candidates_after = []
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if "trial_metadata" in f and f.endswith(".txt"):
                full_p = os.path.join(root, f)
                normalized_p = full_p.replace('\\', '/')
                if "la/cm" in normalized_p.lower():
                    return full_p
                candidates_after.append(full_p)
                
    if candidates_after:
        return candidates_after[0]
            
    raise FileNotFoundError("Could not find trial_metadata.txt in the extracted archive.")

def build_2019_manifests():
    """Parses ASVspoof 2019 train, dev, and eval protocols and saves manifest CSVs."""
    protocols_dir = find_subfolder(config.DATASET_LA_DIR, "ASVspoof2019_LA_cm_protocols")
    
    splits = {
        "train": {
            "protocol_file": "ASVspoof2019.LA.cm.train.trn.txt",
            "audio_subfolder": "ASVspoof2019_LA_train",
            "output_csv": config.TRAIN_2019_CSV
        },
        "dev": {
            "protocol_file": "ASVspoof2019.LA.cm.dev.trl.txt",
            "audio_subfolder": "ASVspoof2019_LA_dev",
            "output_csv": config.DEV_2019_CSV
        },
        "eval": {
            "protocol_file": "ASVspoof2019.LA.cm.eval.trl.txt",
            "audio_subfolder": "ASVspoof2019_LA_eval",
            "output_csv": config.EVAL_2019_CSV
        }
    }
    
    for split_name, info in splits.items():
        # Find exact protocol filename (e.g. if it has a (1) or similar suffix)
        proto_file_actual = info["protocol_file"]
        if os.path.exists(protocols_dir):
            base_prefix = os.path.splitext(info["protocol_file"])[0]
            for f in os.listdir(protocols_dir):
                if base_prefix.lower() in f.lower() and f.endswith(".txt"):
                    proto_file_actual = f
                    break
                    
        proto_path = os.path.join(protocols_dir, proto_file_actual)
        if not os.path.exists(proto_path):
            print(f"[Warning] Protocol file {proto_path} not found. Skipping 2019 {split_name} split.")
            continue
            
        print(f"[Info] Building manifest for 2019 {split_name}...")
        records = []
        with open(proto_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker_id = parts[0]
                    utt_id = parts[1]
                    attack_id = parts[3]
                    label = parts[4]  # 'bonafide' or 'spoof'
                    
                    # Construct expected audio file path dynamically to tolerate suffix names
                    split_folder = find_subfolder(config.DATASET_LA_DIR, info["audio_subfolder"])
                    flac_folder = find_subfolder(split_folder, "flac")
                    filepath = os.path.join(flac_folder, f"{utt_id}.flac")
                    
                    records.append({
                        "utt_id": utt_id,
                        "filepath": filepath,
                        "label": label,
                        "split": split_name,
                        "dataset": "ASVspoof2019_LA",
                        "attack_id": attack_id if attack_id != "-" else "bonafide",
                        "speaker_id": speaker_id
                    })
                    
        df = pd.DataFrame(records)
        df.to_csv(info["output_csv"], index=False)
        print(f"[Success] Saved {len(df)} records to: {info['output_csv']}")

def build_2021_manifest():
    """Parses ASVspoof 2021 evaluation keys and saves eval manifest CSV."""
    trl_file = None
    if os.path.exists(config.DATASET_2021_DIR):
        for f in os.listdir(config.DATASET_2021_DIR):
            if "ASVspoof2021.LA.cm.eval.trl" in f and f.endswith(".txt"):
                trl_file = os.path.join(config.DATASET_2021_DIR, f)
                break
                
    if not trl_file:
        trl_file = os.path.join(config.DATASET_2021_DIR, "ASVspoof2021.LA.cm.eval.trl.txt")
        
    if not os.path.exists(trl_file):
        print(f"[Warning] 2021 trial list {trl_file} not found. Skipping 2021 split.")
        return
        
    try:
        keys_path = download_and_extract_2021_keys()
    except Exception as e:
        print(f"[Error] Failed to get 2021 keys: {e}")
        return
        
    print("[Info] Parsing 2021 evaluation keys metadata...")
    # trial_metadata columns: [speaker, trial, codec, trans, attack, label, trim, subset]
    meta_df = pd.read_csv(keys_path, sep=" ", header=None, 
                          names=["speaker_id", "utt_id", "codec", "trans", "attack_id", "label", "trim", "subset"])
    
    # Read the trial files that are actually in the evaluation protocol
    with open(trl_file, "r") as f:
        eval_trials = set(line.strip() for line in f if line.strip())
        
    print(f"[Info] Found {len(eval_trials)} evaluation trial IDs in protocol.")
    
    # Filter metadata to keep only relevant evaluation trials
    filtered_df = meta_df[meta_df["utt_id"].isin(eval_trials)].copy()
    
    # Find exact flac folder (e.g. flac or flac (1))
    flac_dir_name = "flac"
    if os.path.exists(config.DATASET_2021_DIR):
        for d in os.listdir(config.DATASET_2021_DIR):
            if d.lower().startswith("flac") and os.path.isdir(os.path.join(config.DATASET_2021_DIR, d)):
                flac_dir_name = d
                break
                
    # Construct expected file paths
    filtered_df["filepath"] = filtered_df["utt_id"].apply(
        lambda x: os.path.join(config.DATASET_2021_DIR, flac_dir_name, f"{x}.flac")
    )
    filtered_df["split"] = "eval"
    filtered_df["dataset"] = "ASVspoof2021_LA"
    
    # Select columns to match the 2019 manifest structure
    columns_to_keep = ["utt_id", "filepath", "label", "split", "dataset", "attack_id", "speaker_id"]
    final_df = filtered_df[columns_to_keep]
    
    final_df.to_csv(config.EVAL_2021_CSV, index=False)
    print(f"[Success] Saved {len(final_df)} records to: {config.EVAL_2021_CSV}")

if __name__ == "__main__":
    print("======================================================================")
    print("VoxShield - Generating CSV Manifests")
    print("======================================================================")
    build_2019_manifests()
    build_2021_manifest()
    print("======================================================================")
    print("All Manifest Generation Complete!")
    print("======================================================================")
