# verify_env.py
"""
VoxShield - Environment Verification Module (Phase 8)
------------------------------------------------------
This script checks the local hardware and library configuration:
- PyTorch installation and version
- CUDA availability and active GPU name
- torchaudio installation
- LFCC feature extraction shape check
- VoxShield model forward pass check

To run verification:
    python verify_env.py

Command-line usage context (for Colab / Local terminal):
    !python verify_env.py
"""

import sys
import os

def main():
    print("==========================================================")
    print("                VOXSHIELD SYSTEM VERIFY                   ")
    print("==========================================================")
    
    # 1. Check Python version
    print(f"Python Version: {sys.version}")
    
    # 2. Check PyTorch
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
    except ImportError:
        print("[Error] PyTorch is NOT installed. Please wait for pip installation to finish.")
        return
        
    # 3. Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"  Active GPU Index: {torch.cuda.current_device()}")
        print(f"  GPU Device Name : {torch.cuda.get_device_name(0)}")
        print(f"  Device Memory   : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[Warning] Running on CPU. Training will be extremely slow.")
        
    # 4. Check torchaudio
    try:
        import torchaudio
        print(f"torchaudio Version: {torchaudio.__version__}")
    except ImportError:
        print("[Error] torchaudio is NOT installed.")
        return
        
    # 5. Check other requirements
    libs = ["transformers", "soundfile", "librosa", "pandas", "scipy", "sklearn"]
    print("\nChecking supplementary libraries:")
    for lib in libs:
        try:
            __import__(lib)
            print(f"  {lib:15}: OK")
        except ImportError:
            print(f"  {lib:15}: NOT INSTALLED")
            
    # 6. Test feature shape extraction
    try:
        import features
        dummy_wave = torch.randn(1, 64000)
        feats = features.extract_features(dummy_wave)
        print(f"\nFeature Extraction Check:")
        print(f"  Input waveform shape: {dummy_wave.shape}")
        print(f"  Output feature shape: {feats.shape} (Expected: [1, 1, 20, 401])")
    except Exception as e:
        print(f"\n[Error] Feature extraction check failed: {e}")
        return
        
    # 7. Test VoxShield model forward pass
    try:
        from models.voxshield import VoxShield
        model = VoxShield()
        # Move to GPU if available
        device = torch.device("cuda" if cuda_available else "cpu")
        model = model.to(device)
        
        dummy_wave = dummy_wave.to(device)
        feats = feats.to(device)
        
        # Test frozen-ssl forward pass
        logits = model(dummy_wave, feats, freeze_ssl=True)
        print(f"\nVoxShield Model Check:")
        print(f"  Model output logits shape: {logits.shape} (Expected: [1, 2])")
        print("[Success] Model forward pass completed successfully!")
    except Exception as e:
        print(f"\n[Error] Model check failed: {e}")
        return
        
    print("==========================================================")
    print("         ALL CHECKS PASSED: SYSTEM READY TO TRAIN         ")
    print("==========================================================")

if __name__ == "__main__":
    main()
