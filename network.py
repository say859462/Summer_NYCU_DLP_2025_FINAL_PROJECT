import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

#!! 可能可以加resnet在GCN
#iou in gnc可以調
#enhanced features
def build_overlap_graph(stroke_images, iou_thresh=0.1):
    """
    Build adjacency matrix based on bounding box IoU of strokes.
    
    stroke_images: (N, H, W) binary masks for each stroke
    returns: (N, N) adjacency matrix (0/1)
    """
    N, H, W = stroke_images.shape
    boxes = []

    # Step 1: bounding boxes
    for i in range(N):
        ys, xs = torch.where(stroke_images[i] > 0)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append((0,0,0,0))  # empty stroke
        else:
            x_min, x_max = xs.min().item(), xs.max().item()
            y_min, y_max = ys.min().item(), ys.max().item()
            boxes.append((x_min, y_min, x_max, y_max))

    # Step 2+3: IoU adjacency
    adj = torch.zeros((N, N), dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            xi1, yi1, xi2, yi2 = boxes[i]
            xj1, yj1, xj2, yj2 = boxes[j]

            # intersection
            inter_x1, inter_y1 = max(xi1, xj1), max(yi1, yj1)
            inter_x2, inter_y2 = min(xi2, xj2), min(yi2, yj2)
            inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

            # union
            area_i = (xi2 - xi1 + 1) * (yi2 - yi1 + 1)
            area_j = (xj2 - xj1 + 1) * (yj2 - yj1 + 1)
            union_area = area_i + area_j - inter_area

            iou = inter_area / union_area if union_area > 0 else 0.0

            if iou > iou_thresh:
                adj[i, j] = 1.0

    # Step 4: add self-loops
    adj += torch.eye(N)

    return adj


class SeqSpaFusion(nn.Module):
    """
    Sequential (BiLSTM) + Spatial (GCN) Encoder Fusion
    Input:
        stroke_embeds: (B, N, D)  # ordered stroke embeddings
        adj_matrix: (B, N, N)     # overlap adjacency graph
    Output:
        enhanced_embeds: (B, N, D_out)
    """

    def __init__(self, d_model=256, hidden_size=128, gcn_hidden=128, out_dim=256, num_gcn_layers=2):
        super(SeqSpaFusion, self).__init__()

        # Sequential Encoder (BiLSTM)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.seq_proj = nn.Linear(hidden_size * 2, d_model)

        # Spatial Encoder (GCN)
        self.gcn_layers = nn.ModuleList([
            nn.Linear(d_model if i == 0 else gcn_hidden, gcn_hidden)
            for i in range(num_gcn_layers)
        ])
        self.gcn_proj = nn.Linear(gcn_hidden, d_model)

        # Fusion MLP
        fusion_in_dim = d_model * 3   # original + seq + spa
        self.mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        

    def forward(self, stroke_embeds, adj_matrix, mask = None):
        B, N, D = stroke_embeds.shape

        # Sequential encoding (temporal order preserved)
        seq_out, _ = self.lstm(stroke_embeds)   # (B, N, 2*hidden)
        seq_feat = self.seq_proj(seq_out)       # (B, N, D)

        # Spatial encoding (message passing using adjacency)
        spa_feat = stroke_embeds
        for idx, layer in enumerate(self.gcn_layers):
            agg = torch.bmm(adj_matrix, spa_feat) / (adj_matrix.sum(-1, keepdim=True) + 1e-6)
            new_feat = layer(agg)

            # residual connection
            if new_feat.shape == spa_feat.shape:   # same dimension, safe residual
                spa_feat = F.relu(new_feat + spa_feat)
            else:
                spa_feat = F.relu(new_feat)

        # 🔹 always project back to d_model
        spa_feat = self.gcn_proj(spa_feat)   # (B, N, 256)

        # Fusion
        fused = torch.cat([stroke_embeds, seq_feat, spa_feat], dim=-1)  # (B, N, 3D)
        enhanced = self.mlp(fused)                                      # (B, N, D_out)
        if mask is not None:
            enhanced = enhanced * mask.unsqueeze(-1).float()
        return enhanced


class AddCoords(nn.Module):
    """
    將座標通道 (x, y, 和可選的 r) 添加到輸入張量中。
    """

    def __init__(self, x_dim=256, y_dim=256, with_r=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: (batch, channels, x_dim, y_dim)
        """
        batch_size = input_tensor.size(0)

        # 產生 x, y 座標
        xx_channel = torch.arange(self.x_dim, device=input_tensor.device).float()
        xx_channel = xx_channel.repeat(batch_size, 1, self.y_dim, 1).transpose(2, 3)

        yy_channel = torch.arange(self.y_dim, device=input_tensor.device).float()
        yy_channel = yy_channel.repeat(batch_size, 1, self.x_dim, 1)

        # 正規化到 [-1, 1]
        xx_channel = (xx_channel / (self.x_dim - 1)) * 2 - 1
        yy_channel = (yy_channel / (self.y_dim - 1)) * 2 - 1

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    CoordConv 層，結合了 AddCoords 和一個標準的卷積模組。
    """

    def __init__(self, x_dim, y_dim, in_channels, out_channels, with_r=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=with_r)

        extra_channels = 3 if with_r else 2
        self.conv = nn.Conv2d(in_channels + extra_channels, out_channels, **kwargs)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret


class AutoencoderEmbed(nn.Module):
    """
    用於筆劃嵌入的自動編碼器模型。
    包含一個編碼器和兩個解碼器（一個用於重建，一個用於距離場預測）。
    """

    def __init__(self, code_size, x_dim, y_dim, root_feature):
        super(AutoencoderEmbed, self).__init__()

        # 編碼器
        self.encoder = nn.Sequential(
            CoordConv(x_dim, y_dim, 1, root_feature, kernel_size=3, padding=1),
            nn.BatchNorm2d(root_feature),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128
            nn.Conv2d(root_feature, root_feature * 2, 3, padding=1),
            nn.BatchNorm2d(root_feature * 2),
            nn.ReLU(True),
            nn.Conv2d(root_feature * 2, root_feature * 2, 3, padding=1),
            nn.BatchNorm2d(root_feature * 2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64
            nn.Conv2d(root_feature * 2, root_feature * 4, 3, padding=1),
            nn.BatchNorm2d(root_feature * 4),
            nn.ReLU(True),
            nn.Conv2d(root_feature * 4, root_feature * 4, 3, padding=1),
            nn.BatchNorm2d(root_feature * 4),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32
            nn.Conv2d(root_feature * 4, root_feature * 8, 3, padding=1),
            nn.BatchNorm2d(root_feature * 8),
            nn.ReLU(True),
            nn.Conv2d(root_feature * 8, root_feature * 8, 3, padding=1),
            nn.BatchNorm2d(root_feature * 8),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16
            nn.Conv2d(root_feature * 8, root_feature * 16, 3, padding=1),
            nn.BatchNorm2d(root_feature * 16),
            nn.ReLU(True),
            nn.Conv2d(root_feature * 16, root_feature * 16, 3, padding=1),
            nn.BatchNorm2d(root_feature * 16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8
            nn.Flatten(),
            nn.Linear(root_feature * 16 * 8 * 8, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, code_size),
            nn.Sigmoid(),
        )

        # 解碼器通用部分
        self.decoder_fc = nn.Sequential(
            nn.Linear(code_size, 4096),
            nn.Linear(4096, root_feature * 16 * 8 * 8),
            nn.Unflatten(1, (root_feature * 16, 8, 8)),
        )

        # 解碼器卷積部分
        decoder_conv = nn.Sequential(
            nn.Conv2d(root_feature * 16, root_feature * 16, 3, padding=1),
            nn.BatchNorm2d(root_feature * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(root_feature * 16, root_feature * 8, 2, stride=2),  # 16
            nn.Conv2d(root_feature * 8, root_feature * 8, 3, padding=1),
            nn.BatchNorm2d(root_feature * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(root_feature * 8, root_feature * 4, 2, stride=2),  # 32
            nn.Conv2d(root_feature * 4, root_feature * 4, 3, padding=1),
            nn.BatchNorm2d(root_feature * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(root_feature * 4, root_feature * 2, 2, stride=2),  # 64
            nn.Conv2d(root_feature * 2, root_feature * 2, 3, padding=1),
            nn.BatchNorm2d(root_feature * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(root_feature * 2, root_feature, 2, stride=2),  # 128
            nn.Conv2d(root_feature, root_feature, 3, padding=1),
            nn.BatchNorm2d(root_feature),
            nn.ReLU(True),
            nn.ConvTranspose2d(root_feature, 1, 2, stride=2),  # 256
        )

        # 兩個解碼器分支
        self.decoder_for_cons = nn.Sequential(self.decoder_fc, decoder_conv)
        self.decoder_for_dist = nn.Sequential(
            self.decoder_fc, nn.Sequential(*list(decoder_conv.children()))
        )  # 複製一份

    def forward(self, input_tensor):
        code = self.encoder(input_tensor)
        reconstruction = self.decoder_for_cons(code)
        distance_field = self.decoder_for_dist(code)
        return reconstruction, distance_field

    def encode(self, input_tensor):
        return self.encoder(input_tensor)


# --- Transformer 模型 ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(k.size(-1), dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += mask * -1e9

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        return self.dense(output), attention_weights


class PointWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask):
        # Pre-LN
        x_norm = self.layernorm1(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, mask)
        out1 = x + self.dropout1(attn_output)

        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        out2 = out1 + self.dropout2(ffn_output)

        return out2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        x_norm = self.layernorm1(x)
        attn1, attn_weights_block1 = self.mha1(x_norm, x_norm, x_norm, look_ahead_mask)
        out1 = x + self.dropout1(attn1)

        out1_norm = self.layernorm2(out1)
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1_norm, padding_mask
        )
        out2 = out1 + self.dropout2(attn2)

        out2_norm = self.layernorm3(out2)
        ffn_output = self.ffn(out2_norm)
        out3 = out2 + self.dropout3(ffn_output)

        return out3, attn_weights_block1, attn_weights_block2


def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = torch.from_numpy(angle_rads[np.newaxis, ...]).float()
    return pos_encoding


class Encoder(nn.Module):
    def __init__(
        self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(rate)

    def forward(self, x, mask):
        seq_len = x.size(1)
        x += self.pos_encoding[:, :seq_len, :].to(x.device)
        # x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1
    ):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.size(1)
        attention_weights = {}
        x += self.pos_encoding[:, :seq_len, :].to(x.device)
        # x = self.dropout(x)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask
            )
            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2
        return x, attention_weights


class Discriminator(nn.Module):
    def __init__(self, d_model):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, 1024), nn.Linear(1024, 512), nn.Linear(512, d_model)
        )

    def forward(self, enc_output, dec_output):
        x = self.network(dec_output)
        # 矩陣點積: [N, nb_s, d_model] * [N, nb_g, d_model] -> [N, nb_g, nb_s]
        output = torch.einsum("bsd,bgd->bgs", enc_output, x)
        return output


class GpTransformer(nn.Module):
    def __init__(
        self, num_layers, d_model, num_heads, dff, pe_input, pe_target, rate=0.1
    ):
        super(GpTransformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_target, rate)
        self.disc_layer = Discriminator(d_model)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
        self.seqspa_fusion = SeqSpaFusion(d_model=d_model)
    def forward(self, inp, adj_matrix, mask, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        inp = self.seqspa_fusion(inp, adj_matrix, mask)
        enc_output = self.encoder(inp, enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask
        )
        dec_output = self.layernorm(dec_output)
        final_output = self.disc_layer(enc_output, dec_output)
        return final_output, attention_weights
