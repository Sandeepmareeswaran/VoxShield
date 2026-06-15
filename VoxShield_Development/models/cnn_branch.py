# models/cnn_branch.py
"""
VoxShield - Spectro-Temporal CNN Branch (Phase 6)
-------------------------------------------------
This module processes 2D spectral representations (LFCC or Log-Mel) with 2D CNN layers,
collapses the frequency dimension, and projects the result to the shared dimension D.

To import:
    from models.cnn_branch import CNNBranch
"""

import torch
import torch.nn as nn
import config

class CNNBranch(nn.Module):
    def __init__(self, shared_dim=config.SHARED_DIM):
        super(CNNBranch, self).__init__()
        
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
        
        # Linear projection layer: channel dimension -> shared dimension D
        self.projection = nn.Sequential(
            nn.Linear(config.CNN_NUM_CHANNELS[1], shared_dim),
            nn.LayerNorm(shared_dim),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, x):
        """
        Args:
            x: Spectro-temporal features of shape [Batch, Channels=1, F_bins, T_frames]
        Returns:
            projected_features: Tensor of shape [Batch, T_cnn_new, SharedDim]
        """
        # 1. Forward pass through CNN blocks
        # Output shape: [Batch, Channels_out, F_new, T_new]
        feats = self.conv_blocks(x)
        
        # 2. Collapse the frequency dimension by average pooling
        # Output shape: [Batch, Channels_out, T_new]
        feats = torch.mean(feats, dim=2)
        
        # 3. Reshape for linear projection: [Batch, T_new, Channels_out]
        feats = feats.permute(0, 2, 1)
        
        # 4. Project channels to shared dimension D
        # Output shape: [Batch, T_new, D]
        projected = self.projection(feats)
        return projected
