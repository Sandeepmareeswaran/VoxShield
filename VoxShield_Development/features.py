# features.py
"""
VoxShield - Feature Extraction Module (Phase 3)
-----------------------------------------------
This module provides classes and functions to compute LFCC and Log-Mel Spectrogram features
from audio waveforms. It uses torchaudio transforms optimized for PyTorch pipelines.

To test feature extraction shape:
    python -c "import torch, features; wave = torch.randn(1, 64000); print(features.extract_features(wave).shape)"

Command-line usage context:
    Implemented as a library module. Imported into dataloader and evaluation scripts.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import config

class FeatureExtractor(nn.Module):
    def __init__(self, feature_type=config.FEATURE_TYPE):
        super(FeatureExtractor, self).__init__()
        self.feature_type = feature_type.lower()
        
        if self.feature_type == "lfcc":
            # Initialize LFCC transform
            self.transform = T.LFCC(
                sample_rate=config.LFCC_PARAMS["sample_rate"],
                n_filter=config.LFCC_PARAMS["n_filter"],
                n_lfcc=config.LFCC_PARAMS["n_lfcc"],
                speckwargs=config.LFCC_PARAMS["speckwargs"]
            )
        elif self.feature_type == "mel":
            # Initialize Mel Spectrogram transform
            self.transform = T.MelSpectrogram(
                sample_rate=config.MEL_PARAMS["sample_rate"],
                n_fft=config.MEL_PARAMS["n_fft"],
                win_length=config.MEL_PARAMS["win_length"],
                hop_length=config.MEL_PARAMS["hop_length"],
                n_mels=config.MEL_PARAMS["n_mels"]
            )
        else:
            raise ValueError(f"Unknown feature type: {feature_type}. Use 'lfcc' or 'mel'.")

    def forward(self, x):
        """
        Extract features from raw audio waveform.
        Args:
            x: Tensor of shape [Batch, NumSamples] or [NumSamples]
        Returns:
            features: Tensor of shape [Batch, 1, NumFeatures, NumFrames]
        """
        # Ensure input has batch dimension
        is_batched = x.dim() == 2
        if not is_batched:
            x = x.unsqueeze(0)
            
        feats = self.transform(x)
        
        # If Mel Spectrogram, compute log energy
        if self.feature_type == "mel":
            feats = torch.log(feats + 1e-9)
            
        # Add channel dimension: [Batch, Channels=1, Features, Frames]
        feats = feats.unsqueeze(1)
        
        if not is_batched:
            feats = feats.squeeze(0)
            
        return feats

# Reusable module instances
_lfcc_extractor = None
_mel_extractor = None

def extract_features(waveform, feature_type=None):
    """Utility function to quickly extract features using a cached extractor."""
    global _lfcc_extractor, _mel_extractor
    
    if feature_type is None:
        feature_type = config.FEATURE_TYPE
    feature_type = feature_type.lower()
    
    if feature_type == "lfcc":
        if _lfcc_extractor is None:
            _lfcc_extractor = FeatureExtractor("lfcc")
        return _lfcc_extractor(waveform)
    elif feature_type == "mel":
        if _mel_extractor is None:
            _mel_extractor = FeatureExtractor("mel")
        return _mel_extractor(waveform)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
