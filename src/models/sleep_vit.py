import torch
import torch.nn as nn


class FullyConvSleepViT(nn.Module):
    def __init__(self, num_channels=2, d_model=128, nhead=4, num_layers=4, num_classes=5):
        super().__init__()

        # Input is [Batch, 2, 3000]
        self.backbone = nn.Sequential(
            # 0.25 seconds context at 100Hz - Output is [Batch, 32, 600]
            nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=25, stride=5, padding=12),
            nn.BatchNorm1d(32),
            nn.GELU(),

            # Output is [Batch, 64, 120]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=5, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),

            # Output is [Batch, 128, 15]
            nn.Conv1d(in_channels=64, out_channels=d_model, kernel_size=9, stride=8, padding=4)
        )

        self.ln_embed = nn.LayerNorm(d_model)

        self.num_tokens = 15
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens + 1, d_model) * 0.02)

        # Modern Transformer Stack (Pre-LN design for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,  # Subtle dropout to protect self-attention layers
            activation='gelu',
            norm_first=True,  # Crucial for stable gradient flow without deep residual drops
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final Classification Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        tokens = self.backbone(x)  # [Batch, 128, 15]
        tokens = tokens.transpose(1, 2)  # [Batch, 15, 128]
        tokens = self.ln_embed(tokens)

        # 1. Expand CLS token to match batch size: [Batch, 1, 128]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # 2. Concatenate CLS token to the sequence: [Batch, 16, 128]
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # 3. Add positional embeddings (which is now sized for 16 tokens)
        tokens = tokens + self.pos_embedding

        # 4. Pass through Transformer
        transformer_out = self.transformer(tokens)  # [Batch, 16, 128]

        # 5. Extract ONLY the CLS token (index 0) for classification
        cls_representation = transformer_out[:, 0, :]  # [Batch, 128]

        # 6. Compute logits
        logits = self.mlp_head(cls_representation)  # [Batch, 5]
        return logits


# --- Verification & Dimension Check ---
if __name__ == "__main__":
    model = FullyConvSleepViT()
    mock_eeg_batch = torch.randn(32, 2, 3000)
    output = model(mock_eeg_batch)

    print("Model verification test...")
    print(f"Input Shape:  {mock_eeg_batch.shape}")
    print(f"Output Shape: {output.shape}")  # Should be (32, 5)
    print("🚀 Validation successful! No dimension alignment exceptions.")
