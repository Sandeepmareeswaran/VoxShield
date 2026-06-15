# models/fusion.py
"""
VoxShield - Feature Fusion Module (Phase 6)
-------------------------------------------
This module aligns the time dimension of wav2vec2 and spectro-temporal CNN branches,
concatenates the aligned representations, and projects them back to the shared dimension D.

To import:
    from models.fusion import FeatureFusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class FeatureFusion(nn.Module):
    def __init__(self, shared_dim=config.SHARED_DIM):
        super(FeatureFusion, self).__init__()
        
        # Linear projection to reduce concatenated dimension [2 * D] back to [D]
        self.fusion_projection = nn.Sequential(
            nn.Linear(2 * shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, wav2vec_feats, cnn_feats):
        """
        Args:
            wav2vec_feats: Tensor of shape [Batch, T_ssl, D]
            cnn_feats: Tensor of shape [Batch, T_cnn, D]
        Returns:
            fused_feats: Tensor of shape [Batch, T_ssl, D]
        """
        batch_size, t_ssl, d_dim = wav2vec_feats.shape
        _, t_cnn, _ = cnn_feats.shape
        
        # 1. Align time dimensions if they differ
        # We project/interpolate the CNN features time axis to match the SSL branch
        if t_cnn != t_ssl:
            # interpolate expects input in shape [Batch, Channels, Length]
            cnn_aligned = cnn_feats.permute(0, 2, 1)  # shape: [Batch, D, T_cnn]
            cnn_aligned = F.interpolate(
                cnn_aligned, 
                size=t_ssl, 
                mode="linear", 
                align_corners=False
            )
            cnn_aligned = cnn_aligned.permute(0, 2, 1)  # shape: [Batch, T_ssl, D]
        else:
            cnn_aligned = cnn_feats
            
        # 2. Concatenate both views: shape [Batch, T_ssl, 2 * D]
        concatenated = torch.cat([wav2vec_feats, cnn_aligned], dim=2)
        
        # 3. Project back to shared dimension D: shape [Batch, T_ssl, D]
        fused = self.fusion_projection(concatenated)
        return fused
