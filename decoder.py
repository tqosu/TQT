
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, tgt):
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        return self.norm(tgt + self.dropout(tgt2))

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, memory):
        out, _ = self.cross_attn(query, memory, memory)
        return self.norm(query + self.dropout(out))

class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x2 = self.linear2(F.relu(self.linear1(x)))
        return self.norm(x + self.dropout(x2))

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dropout)
        self.self_attn = SelfAttentionLayer(d_model, nhead, dropout)
        self.ffn = FFNLayer(d_model, dim_feedforward, dropout)
    def forward(self, query, memory):
        query = self.cross_attn(query, memory)
        query = self.self_attn(query)
        query = self.ffn(query)
        return query

class TemporalMaskFormer_V1(nn.Module):
    # Only use last middle output
    def __init__(self, num_queries, d_model, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)  # 可换成prototype query
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.mask_head = nn.Linear(d_model, d_model)  # 可替换
        self.class_head = nn.Linear(d_model, num_queries)

    def forward(self, frame_features):
        # frame_features: [B, T, C]
        B, T, C = frame_features.shape
        Q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, D]
        memory = frame_features  # [B, T, D]

        for layer in self.decoder_layers:
            Q = layer(Q, memory)

        # Mask Head: Q * K^T 作为 similarity，每个 query 对所有帧打分
        # Q: [B, num_queries, D], memory: [B, T, D] -> [B, num_queries, T]
        mask_logits = torch.einsum('bqd,btd->bqt', Q, memory)
        # Class Head: [B, num_queries, num_classes]，可选
        class_logits = self.class_head(Q)  # [B, num_queries, num_queries]
        return mask_logits, class_logits

class TemporalMaskFormer_V2(nn.Module):
    #  V1 + Use all middle output
    def __init__(self, num_queries, d_model, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)  # 可换成prototype query
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.mask_head = nn.Linear(d_model, d_model)  # 可替换
        self.class_head = nn.Linear(d_model, num_queries)
        self.num_layers = num_layers

    def forward(self, frame_features_list):  
        # frame_features: [B, T, C]
        # print(frame_features_list.shape)
        L, B, T, C = frame_features_list.shape
        Q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, D]
        memory_list = frame_features_list[-self.num_layers:]  # [B, T, D]

        assert len(memory_list) == len(self.decoder_layers), \
            "Length of memory_list should match decoder_layers"

        for i, layer in enumerate(self.decoder_layers):
            memory = memory_list[i]
            # print(memory.shape,  memory_list.shape, Q.shape)
            # exit()
            Q = layer(Q, memory)

        # Mask Head: Q * K^T 作为 similarity，每个 query 对所有帧打分
        # Q: [B, num_queries, D], memory: [B, T, D] -> [B, num_queries, T]
        mask_logits = torch.einsum('bqd,btd->bqt', Q, memory)                       # TODO: return all 3
        # Class Head: [B, num_queries, num_classes]，可选
        class_logits = self.class_head(Q)  # [B, num_queries, num_queries]
        return mask_logits, class_logits

class TemporalMaskFormer_V3(nn.Module):
    # V2 + Apply all loss on all Q
    def __init__(self, num_queries, d_model, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)  # 可换成prototype query
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.mask_head = nn.Linear(d_model, d_model)  # 可替换
        self.class_head = nn.Linear(d_model, num_queries)
        self.num_layers = num_layers

    def forward(self, frame_features_list):  
        # frame_features_list: [L, B, T, C]
        L, B, T, C = frame_features_list.shape
        Q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]

        memory_list = frame_features_list[-self.num_layers:]  # 每层一个 memory

        intermediate_Qs = []
        for i, layer in enumerate(self.decoder_layers):
            memory = memory_list[i]  # [B, T, D]
            Q = layer(Q, memory)     # [B, Q, D]
            intermediate_Qs.append(Q)

        mask_logits_list = []
        class_logits_list = []

        for i, Q_i in enumerate(intermediate_Qs):
            memory = memory_list[i]  # 对应 memory
            mask_logits = torch.einsum('bqd,btd->bqt', Q_i, memory)  # [B, Q, T]
            class_logits = self.class_head(Q_i)                      # [B, Q, num_classes]
            mask_logits_list.append(mask_logits)
            class_logits_list.append(class_logits)

        return mask_logits_list, class_logits_list


class TemporalMaskFormer_V4(nn.Module):
    # V3 + Only use last feature
    def __init__(self, num_queries, d_model, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)  # 可换成prototype query
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.mask_head = nn.Linear(d_model, d_model)  # 可替换
        self.class_head = nn.Linear(d_model, num_queries)
        self.num_layers = num_layers

    def forward(self, frame_features_list):  
        # frame_features_list: [L, B, T, C]
        L, B, T, C = frame_features_list.shape
        Q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]

        memory_list = frame_features_list[-self.num_layers:]  # 每层一个 memory

        intermediate_Qs = []
        for i, layer in enumerate(self.decoder_layers):
            memory = memory_list[-1]  # [B, T, D]
            Q = layer(Q, memory)     # [B, Q, D]
            intermediate_Qs.append(Q)

        mask_logits_list = []
        class_logits_list = []

        for i, Q_i in enumerate(intermediate_Qs):
            memory = memory_list[-1]  # 对应 memory
            mask_logits = torch.einsum('bqd,btd->bqt', Q_i, memory)  # [B, Q, T]
            class_logits = self.class_head(Q_i)                      # [B, Q, num_classes]
            mask_logits_list.append(mask_logits)
            class_logits_list.append(class_logits)

        return mask_logits_list, class_logits_list

class TemporalMaskFormer_V5(nn.Module):
    # V4 + SHIFT CENTER
    def __init__(self, num_queries, d_model, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)  # 可换成prototype query
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.mask_head = nn.Linear(d_model, d_model)  # 可替换
        self.class_head = nn.Linear(d_model, num_queries)
        self.num_layers = num_layers

    def forward(self, frame_features_list):  
        # frame_features_list: [L, B, T, C]
        L, B, T, C = frame_features_list.shape
        Q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]

        memory_list = frame_features_list[-self.num_layers:]  # 每层一个 memory

        intermediate_Qs = []
        for i, layer in enumerate(self.decoder_layers):
            memory = memory_list[-1]  # [B, T, D]
            Q = layer(Q, memory)     # [B, Q, D]
            intermediate_Qs.append(Q)

        mask_logits_list = []
        class_logits_list = []

        for i, Q_i in enumerate(intermediate_Qs):
            memory = memory_list[-1]  # 对应 memory
            mask_logits = torch.einsum('bqd,btd->bqt', Q_i, memory)  # [B, Q, T]
            class_logits = self.class_head(Q_i)                      # [B, Q, num_classes]
            mask_logits_list.append(mask_logits)
            class_logits_list.append(class_logits)

        return mask_logits_list, class_logits_list