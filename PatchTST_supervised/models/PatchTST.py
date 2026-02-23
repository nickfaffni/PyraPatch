__all__ = ['PatchTST']

import torch
from torch import nn
import math
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Channel, Length]
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # x: [Batch * Channel, n_patches, patch_len]

        x = self.value_embedding(x)
        x = x + self.position_embedding(x)
        return self.dropout(x)


class PatchMerging(nn.Module):
    """
    Merging Layer for PyraPatch.
    Merges 2 adjacent tokens and doubles the feature dimension.
    """

    def __init__(self, d_model):
        super().__init__()
        # Input dim: d_model. We merge 2, so raw dim is 2*d_model.
        # We project to 2*d_model.
        self.reduction = nn.Linear(2 * d_model, 2 * d_model, bias=False)
        self.norm = nn.LayerNorm(2 * d_model)

    def forward(self, x):
        """
        x: [Batch, Length, Dim]
        """
        B, L, D = x.shape

        # 1. Pad if length is odd
        if L % 2 != 0:
            x = torch.cat([x, x[:, -1:, :]], dim=1)

        # 2. Reshape to [Batch, Length/2, 2, Dim]
        x = x.view(B, -1, 2, D)

        # 3. Concatenate adjacent tokens -> [Batch, Length/2, 2*Dim]
        x = x.view(B, -1, 2 * D)

        # 4. Linear Projection & Norm
        x = self.norm(self.reduction(x))

        return x


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [Batch, n_vars, d_model, n_patches]
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = x.permute(0, 2, 1)  # [Batch, Target, n_vars]
        return x


class PatchTST_backbone(nn.Module):
    def __init__(self, c_in, context_window, target_window, patch_len, stride,
                 n_layers, d_model, n_heads, d_ff, dropout,
                 fc_dropout, head_dropout, attn_dropout,
                 num_stages=1, pct_start=0.3, verbose=False, **kwargs):

        super().__init__()

        # --- CONFIGURATION ---
        self.num_stages = num_stages

        # Distribute layers. If num_stages=1 (Baseline), all layers go to stage 0.
        # If num_stages=3 (Pyramid), layers are split (e.g., 3 layers -> [1, 1, 1])
        if self.num_stages > 1:
            layers_per_stage = [max(1, n_layers // self.num_stages)] * self.num_stages
        else:
            layers_per_stage = [n_layers]

        # 1. Patching
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, 0, dropout)

        # 2. Calculate initial patches
        n_patches = int((context_window - patch_len) / stride + 1)

        # 3. Build Stages
        self.stages = nn.ModuleList()
        self.merging_layers = nn.ModuleList()

        current_d_model = d_model
        current_n_patches = n_patches

        for i in range(self.num_stages):
            # --- Encoder for this stage ---
            stage_layers = [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=attn_dropout,
                                      output_attention=False),
                        current_d_model, n_heads),
                    current_d_model,
                    d_ff,
                    dropout=dropout,
                    activation='gelu'
                ) for _ in range(layers_per_stage[i])
            ]

            encoder = Encoder(stage_layers, norm_layer=torch.nn.LayerNorm(current_d_model))
            self.stages.append(encoder)

            # --- Merging Layer (only if Pyramid AND not the last stage) ---
            if i < self.num_stages - 1:
                self.merging_layers.append(PatchMerging(current_d_model))

                # Update dimensions for next stage
                current_d_model = current_d_model * 2
                current_n_patches = math.ceil(current_n_patches / 2)

        # 4. Final Head
        # The head sees the flattened representation of the LAST stage
        head_dim = current_n_patches * current_d_model
        self.head = FlattenHead(c_in, head_dim, target_window, head_dropout=head_dropout)

    def forward(self, x):
        # x: [Batch, Seq_Len, n_vars]

        # 1. Permute for Channel Independence
        # x: [Batch, Length, Vars] -> [Batch, Vars, Length]
        x = x.permute(0, 2, 1)

        # 2. Patching
        # u: [Batch * Vars, n_patches, d_model]
        u = self.patch_embedding(x)

        # 3. Encoder Loop
        for i in range(self.num_stages):
            # A. Pass through Transformer layers of this stage
            u, _ = self.stages[i](u)

            # B. Merge (reduce sequence, increase dim)
            if i < self.num_stages - 1:
                u = self.merging_layers[i](u)

        # u is now: [Batch * Vars, n_patches_final, d_model_final]

        # 4. Reshape for Head
        # [Batch * Vars, N, D] -> [Batch, Vars, N, D] -> [Batch, Vars, D, N]
        u = u.reshape(x.shape[0], x.shape[1], u.shape[1], u.shape[2])
        u = u.permute(0, 1, 3, 2)

        # 5. Prediction
        output = self.head(u)

        return output


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # DEBUG PRINT
        print(">>> LOADING PYRAPATCH MODEL (UPDATED)")

        # Use getattr to avoid crash if task_name is missing
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Model Hyperparameters
        num_stages = getattr(configs, 'num_stages', 3)
        default_patch_len = 8 if num_stages > 1 else 16
        patch_len = getattr(configs, 'patch_len', default_patch_len)
        stride = getattr(configs, 'stride', 4)

        self.model = PatchTST_backbone(
            c_in=configs.enc_in,
            context_window=configs.seq_len,
            target_window=configs.pred_len,
            patch_len=patch_len,
            stride=stride,
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            fc_dropout=configs.fc_dropout,
            head_dropout=configs.head_dropout,
            attn_dropout=configs.dropout,
            num_stages=num_stages,
            pct_start=0.3
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        dec_in = x_enc

        # Normalization (Instance Norm)
        means = dec_in.mean(1, keepdim=True).detach()
        dec_in = dec_in - means
        stdev = torch.sqrt(torch.var(dec_in, dim=1, keepdim=True, unbiased=False) + 1e-5)
        dec_in /= stdev

        # Run Backbone
        dec_out = self.model(dec_in)

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
