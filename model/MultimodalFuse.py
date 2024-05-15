import torch
import torch.nn as nn
import torch.functional as F
from einops.layers.torch import Rearrange


class MultimodalFuse(nn.Module):
    def __init__(self, visual_size, text_size, hidden_size,dropout):
        super().__init__(MultimodalFuse, self)
        self.visual_linear = nn.Linear(visual_size, hidden_size)
        self.question_linear = nn.Linear(text_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, visual, question):
        visual = self.dropout(self.relu(self.visual_linear(visual)))
        question = self.dropout(self.relu(self.question_linear(question)))
        output = torch.matmul(visual, question)

        return output


class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other key
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


# 定义模态融合模型
class ModalityFusionModel(nn.Module):
    def __init__(self, text_embed_size, image_embed_size, output_size, heads):
        super(ModalityFusionModel, self).__init__()
        self.text_embed_size = text_embed_size
        self.image_embed_size = image_embed_size
        self.output_size = output_size
        self.heads = heads

        # Project text and image features to the same size as the output
        self.text_projection = nn.Linear(text_embed_size, output_size)
        self.image_projection = nn.Linear(image_embed_size, output_size)

        # Cross-Attention Fusion
        self.cross_attention = CrossAttention(output_size, heads)

        # Final layer to produce the output
        self.fc_out = nn.Linear(output_size, output_size)

    def forward(self, text_features, image_features, text_mask, image_mask):
        # Project text and image features
        projected_text = self.text_projection(text_features)
        projected_image = self.image_projection(image_features)

        # Create a mask that combines text and image masks
        combined_mask = None  # You need to create a combined mask if needed

        # Apply cross-attention to fuse text and image features
        fused_features = self.cross_attention(
            projected_image,
            projected_image,
            projected_text,
            combined_mask
        )

        # Apply a final layer to produce the output
        output = self.fc_out(fused_features)

        return output


# Example usage:
# text_embed_size = 768
# image_embed_size = 2048
# output_size = 1024
# heads = 8
# fusion_model = ModalityFusionModel(text_embed_size, image_embed_size, output_size, heads)
# text_features = torch.rand(1, 10, text_embed_size)  # BERT-like features
# image_features = torch.rand(1, 49, image_embed_size)  # ResNet-like features
# text_mask = None  # or a mask tensor for text if needed
# image_mask = None  # or a mask tensor for image if needed
# fused_features = fusion_model(text_features, image_features, text_mask, image_mask)


class CrossAttentionFuse(nn.Module):
    def __init__(self, visual_size, text_size, hidden_size,dropout):
        super().__init__(CrossAttentionFuse, self)
        self.visual_embedding = nn.Linear(visual_size, hidden_size)
        self.question_embedding = nn.Linear(text_size, hidden_size)
        











