import torch
import torch.nn as nn

# =====================================================
# ET TEMPORAL ENCODER
# =====================================================

class ETTemporalEncoder(nn.Module):

    def __init__(

        self,

        input_dim=3,

        cnn_channels=64,

        lstm_hidden=128,

        transformer_heads=4,

        transformer_layers=2,

        dropout=0.2
    ):

        super().__init__()

        # ==========================================
        # CNN
        # ==========================================

        self.cnn = nn.Sequential(

            nn.Conv1d(
                input_dim,
                cnn_channels,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(
                cnn_channels
            ),

            nn.ReLU(),

            nn.Conv1d(
                cnn_channels,
                cnn_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(
                cnn_channels
            ),

            nn.ReLU()
        )

        # ==========================================
        # BiLSTM
        # ==========================================

        self.lstm = nn.LSTM(

            input_size=cnn_channels,

            hidden_size=lstm_hidden,

            num_layers=2,

            batch_first=True,

            dropout=dropout,

            bidirectional=True
        )

        # ==========================================
        # TRANSFORMER
        # ==========================================

        enc_layer = nn.TransformerEncoderLayer(

            d_model=lstm_hidden * 2,

            nhead=transformer_heads,

            dim_feedforward=256,

            dropout=dropout,

            batch_first=True,

            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(

            enc_layer,

            num_layers=transformer_layers,

            enable_nested_tensor=False
        )

        # ==========================================
        # ATTENTION POOLING
        # ==========================================

        self.attn = nn.Sequential(

            nn.Linear(
                lstm_hidden * 2,
                128
            ),

            nn.Tanh(),

            nn.Linear(
                128,
                1
            )
        )

        # ==========================================
        # FINAL PROJECTION
        # ==========================================

        self.proj = nn.Sequential(

            nn.Linear(
                lstm_hidden * 2,
                128
            ),

            nn.ReLU(),

            nn.Dropout(dropout)
        )

    # =================================================
    # FORWARD
    # =================================================

    def forward(self, x):

        """
        x:
            [B, T, 3]
        """

        # =============================================
        # CNN
        # =============================================

        x = x.transpose(1, 2)

        x = self.cnn(x)

        x = x.transpose(1, 2)

        # =============================================
        # LSTM
        # =============================================

        x, _ = self.lstm(x)

        # =============================================
        # TRANSFORMER
        # =============================================

        x = self.transformer(x)

        # =============================================
        # ATTENTION
        # =============================================

        attn = self.attn(x)

        attn = torch.softmax(
            attn,
            dim=1
        )

        pooled = torch.sum(
            x * attn,
            dim=1
        )

        # =============================================
        # EMBEDDING
        # =============================================

        emb = self.proj(pooled)

        return {

            "et_emb": emb,

            "attn": attn.squeeze(-1)
        }
