# models/voxshield.py
"""
VoxShield - Master Model Architecture Module (Phase 6)
------------------------------------------------------
This module integrates the raw waveform wav2vec2 branch, spectro-temporal CNN branch,
early feature fusion, temporal attention Transformer, attention pooling, and
the final MLP classification head into a single unified PyTorch model.

To import:
    from models.voxshield import VoxShield
"""

import torch
import torch.nn as nn
import config

from models.wav2vec_branch import Wav2VecBranch
from models.cnn_branch import CNNBranch
from models.fusion import FeatureFusion
from models.attention import TemporalAttention
from models.pooling import AttentionPooling

class VoxShield(nn.Module):
    def __init__(self, shared_dim=config.SHARED_DIM, num_classes=config.NUM_CLASSES):
        super(VoxShield, self).__init__()
        
        # 1. Dual-Branch Encoders
        self.wav2vec_branch = Wav2VecBranch(shared_dim=shared_dim)
        self.cnn_branch = CNNBranch(shared_dim=shared_dim)
        
        # 2. Early Fusion Layer
        self.fusion = FeatureFusion(shared_dim=shared_dim)
        
        # 3. Temporal Attention Block (Transformer Encoder)
        self.temporal_attention = TemporalAttention(d_model=shared_dim)
        
        # 4. Attention Pooling Layer
        self.pooling = AttentionPooling(d_model=shared_dim)
        
        # 5. MLP Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(shared_dim, num_classes)
        )

    def forward(self, waveform, spec_feats, freeze_ssl=True, return_attention=False):
        """
        Args:
            waveform: Raw audio waveform tensor of shape [Batch, NumSamples]
            spec_feats: Spectral feature tensor (LFCC/Log-Mel) of shape [Batch, Channels=1, F, T]
            freeze_ssl: If True, wav2vec2 parameters remain frozen during forward pass
            return_attention: If True, returns attention weights from pooling
        Returns:
            logits: Classification logits of shape [Batch, NumClasses]
            (optional) attention_weights: Attention weights of shape [Batch, SeqLength, 1]
        """
        # 1. Extract and project features from both branches
        # wav2vec_out shape: [Batch, T_ssl, D]
        wav2vec_out = self.wav2vec_branch(waveform, freeze=freeze_ssl)
        
        # cnn_out shape: [Batch, T_cnn_new, D]
        cnn_out = self.cnn_branch(spec_feats)
        
        # 2. Fuse views: shape [Batch, T_ssl, D]
        fused = self.fusion(wav2vec_out, cnn_out)
        
        # 3. Apply temporal modeling: shape [Batch, T_ssl, D]
        attended = self.temporal_attention(fused)
        
        # 4. Pooling: shape [Batch, D]
        if return_attention:
            clip_embedding, weights = self.pooling(attended, return_attention=True)
            logits = self.classifier(clip_embedding)
            return logits, weights
        else:
            clip_embedding = self.pooling(attended, return_attention=False)
            logits = self.classifier(clip_embedding)
            return logits
