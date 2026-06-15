# models/pooling.py
"""
VoxShield - Attention Pooling Module (Phase 6)
-----------------------------------------------
This module implements Self-Attention Pooling to convert a sequence of temporal frame-level
embeddings into a single unified clip-level embedding.

To import:
    from models.pooling import AttentionPooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class AttentionPooling(nn.Module):
    def __init__(self, d_model=config.SHARED_DIM):
        super(AttentionPooling, self).__init__()
        
        # Attention network layers
        self.attention_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        # Vector w to project to scalar attention score
        self.attention_vector = nn.Parameter(torch.randn(d_model, 1))

    def forward(self, x, return_attention=False):
        """
        Args:
            x: Temporal sequence features of shape [Batch, SeqLength, D]
            return_attention: If True, returns attention weights as well
        Returns:
            clip_embedding: Pooled clip embedding of shape [Batch, D]
            (optional) weights: Attention weights of shape [Batch, SeqLength, 1]
        """
        # 1. Project frame features: shape [Batch, SeqLength, D]
        proj = self.attention_proj(x)
        
        # 2. Compute scores by multiplying with the parameter vector
        # proj is [B, T, D], attention_vector is [D, 1] -> scores is [B, T, 1]
        scores = torch.matmul(proj, self.attention_vector)
        
        # 3. Softmax over the time dimension to get normalized weights: shape [B, T, 1]
        weights = F.softmax(scores, dim=1)
        
        # 4. Compute weighted sum of frames: shape [Batch, D]
        # x is [B, T, D], weights is [B, T, 1] -> element-wise mult & sum along time (dim=1)
        pooled = torch.sum(x * weights, dim=1)
        
        if return_attention:
            return pooled, weights
        return pooled
