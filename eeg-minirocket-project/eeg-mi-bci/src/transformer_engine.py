import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGTransformer(nn.Module):
    """
    A Convolutional Transformer designed for EEG Motor Imagery.
    Extracts spatial/temporal features via a small CNN block, 
    then passes them to a Multi-Head Self-Attention Transformer Encoder.
    """
    def __init__(self, n_classes=4, in_chans=22, input_window_samples=1000, 
                 embed_dim=40, depth=2, heads=8, drop_rate=0.5):
        super().__init__()
        
        # 1. Convolutional Block (Spatial & Temporal Filtering)
        self.conv_temporal = nn.Conv2d(1, 40, (1, 25), padding=(0, 12))
        self.conv_spatial = nn.Conv2d(40, 40, (in_chans, 1), bias=False)
        self.batchnorm = nn.BatchNorm2d(40)
        self.pooling = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(drop_rate)
        
        # Calculate sequence length after pooling
        out_len = ((input_window_samples - 75) // 15) + 1
        
        # 2. Transformer Encoder Block
        # We treat the flattened spatial filters as the embedding dimension
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, out_len + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=drop_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 3. Classification Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Timepoints)
        # Add dummy dimension for Conv2d -> (Batch, 1, Channels, Timepoints)
        x = x.unsqueeze(1)
        
        # Convolutional feature extraction
        x = self.conv_temporal(x)
        x = self.conv_spatial(x)
        x = self.batchnorm(x)
        x = F.elu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        
        # Reshape for Transformer: (Batch, SequenceLength, EmbeddingDim)
        x = x.squeeze(2).transpose(1, 2)
        
        # Add Class Token and Positional Embedding
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Pass through Transformer
        x = self.transformer_encoder(x)
        
        # Take the output of the CLS token for classification
        cls_out = x[:, 0, :]
        return self.mlp_head(cls_out)

if __name__ == "__main__":
    # Test the model with dummy data
    dummy_input = torch.randn(16, 22, 1000) # Batch=16, Chans=22, Time=1000
    model = EEGTransformer(n_classes=4, in_chans=22, input_window_samples=1000)
    out = model(dummy_input)
    print("Transformer Output Shape:", out.shape) # Should be (16, 4)
