# models/attention.py
"""
VoxShield - Temporal Attention Module (Phase 6)
-----------------------------------------------
This module implements sinusoidal positional encoding and a multi-layer
Transformer Encoder to model temporal dependencies across the fused feature sequence.

To import:
    from models.attention import TemporalAttention
"""

import math
import torch
import torch.nn as nn
import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [Batch, SequenceLength, D]
        x = x + self.pe[:, :x.size(1)]
        return x

class TemporalAttention(nn.Module):
    def __init__(self, d_model=config.SHARED_DIM, nhead=config.TRANSFORMER_HEADS, 
                 num_layers=config.TRANSFORMER_LAYERS, dim_feedforward=config.TRANSFORMER_FF_DIM):
        super(TemporalAttention, self).__init__()
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=config.DROPOUT,
            batch_first=True  # Ensure shape is [Batch, Seq, D]
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x: Fused features of shape [Batch, SeqLength, D]
        Returns:
            attended_features: Tensor of shape [Batch, SeqLength, D]
        """
        # 1. Add Positional Encodings
        x = self.pos_encoder(x)
        
        # 2. Forward through Transformer Encoder layers
        attended = self.transformer_encoder(x)
        return attended
