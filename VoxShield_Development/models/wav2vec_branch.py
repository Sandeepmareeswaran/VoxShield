# models/wav2vec_branch.py
"""
VoxShield - wav2vec2 Feature Branch (Phase 6)
---------------------------------------------
This module wraps the Hugging Face pre-trained wav2vec2 model and implements
a linear projection layer to project embeddings to the shared dimension D.

To import:
    from models.wav2vec_branch import Wav2VecBranch
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import config

class Wav2VecBranch(nn.Module):
    def __init__(self, model_name=config.WAV2VEC_MODEL, shared_dim=config.SHARED_DIM):
        super(Wav2VecBranch, self).__init__()
        # Load pre-trained wav2vec2 model
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        
        # Linear projection layer: 768 dimensions of wav2vec-base -> shared dimension D
        self.projection = nn.Sequential(
            nn.Linear(768, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, x, freeze=True):
        """
        Args:
            x: Raw waveforms of shape [Batch, NumSamples] (e.g., 64000 samples)
            freeze: If True, blocks gradients for the entire wav2vec2 backbone
        Returns:
            projected_features: Tensor of shape [Batch, T_ssl, SharedDim]
        """
        # Set gradients behavior for the backbone
        if freeze:
            with torch.no_grad():
                self.wav2vec.eval()
                outputs = self.wav2vec(x)
                feats = outputs.last_hidden_state  # shape: [Batch, T_ssl, 768]
        else:
            outputs = self.wav2vec(x)
            feats = outputs.last_hidden_state  # shape: [Batch, T_ssl, 768]
            
        # Project features to shared dimension D
        projected = self.projection(feats)  # shape: [Batch, T_ssl, D]
        return projected
