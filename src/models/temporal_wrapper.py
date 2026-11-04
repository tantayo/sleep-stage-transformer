import torch
import torch.nn as nn
from src.models.sleep_vit import FullyConvSleepViT


class SleepTemporalWrapper(nn.Module):
    def __init__(self, backbone_checkpoint_path, d_model=128, nhead=4, num_layers=2, num_classes=5):
        super().__init__()

        # 1. Load and freeze Phase 1 backbone
        self.backbone = FullyConvSleepViT(num_channels=2, d_model=d_model, nhead=nhead, num_layers=4,
                                          num_classes=num_classes)
        self.backbone.load_state_dict(torch.load(backbone_checkpoint_path))

        # Strip off classification head to output raw 128-dim [CLS] token features
        self.backbone.mlp_head = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False  # Freeze Phase 1 weights entirely

        # 2. Temporal Positional Embedding for the 21-epoch sequence window
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, 21, d_model) * 0.02)

        # 3. Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Sequence Classification Head
        self.sequence_mlp_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: [Batch, 21, 2, 3000]
        batch_size, seq_len, channels, samples = x.shape

        # Flatten sequence and batch dimensions to pass all epochs through the frozen backbone
        x = x.view(batch_size * seq_len, channels, samples)

        with torch.no_grad():
            features = self.backbone(x)  # Shape: [Batch * 21, 128]

        # Reshape back to sequence windows
        features = features.view(batch_size, seq_len, -1)  # Shape: [Batch, 21, 128]

        # Add temporal positional encoding
        features = features + self.temporal_pos_embedding

        # Pass through temporal transformer
        temporal_out = self.temporal_transformer(features)  # Shape: [Batch, 21, 128]

        # Compute logits for all epochs in the sequence
        logits = self.sequence_mlp_head(temporal_out)  # Shape: [Batch, 21, 5]
        return logits