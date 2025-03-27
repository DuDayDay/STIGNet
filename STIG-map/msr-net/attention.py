from os.path import exists

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
# from backbone import farthest_point_sample, index_points, closest_points
from einops import rearrange
from einops.layers.torch import Rearrange

def FeedForward(dim, hidden_dim):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
    )
# class Attention(nn.Module):
#     def __init__(self, dim, heads = 4, dim_head = 64):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.norm = nn.LayerNorm(dim)
#
#         self.attend = nn.Softmax(dim = -1)
#
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#         self.to_out = nn.Linear(inner_dim, dim, bias = False)
#
#     def forward(self, x):
#         # x = self.norm(x)
#
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
#
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#
#         attn = self.attend(dots)
#
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#
#         return self.to_out(out), v
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5  # 缩放因子
        self.norm = nn.LayerNorm(dim)  # 规范化
        self.attend = nn.Softmax(dim=-1)  # 注意力权重计算

    def forward(self, x):
        # 可选：对输入进行 LayerNorm
        # x = self.norm(x)  # [batch, n, dim]

        # 将输入直接作为 q, k, v
        q, k, v = x, x, x

        # 计算点积注意力
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [batch, n, n]
        attn = self.attend(dots)  # [batch, n, n]

        # 注意力权重乘以值
        out = torch.matmul(attn, v) + v # [batch, n, dim]

        return out, attn
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.input_dim = input_dim
        self.dim_head = attention_dim // num_heads  # 每个头的维度
        self.layer_norm = nn.LayerNorm(input_dim)
        assert attention_dim % num_heads == 0, "Attention dimension must be divisible by number of heads"

        # 定义用于查询、键、值的线性变换（多头）
        self.query_linear = nn.Linear(input_dim, attention_dim)
        self.key_linear = nn.Linear(input_dim, attention_dim)
        self.value_linear = nn.Linear(input_dim, attention_dim)

        # 输出的线性层
        self.out_linear = nn.Linear(attention_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 计算查询、键和值 (batch_size, seq_len, attention_dim)
        Q = self.query_linear(x)  # (batch_size, seq_len, attention_dim)
        K = self.key_linear(x)  # (batch_size, seq_len, attention_dim)
        V = self.value_linear(x)  # (batch_size, seq_len, attention_dim)

        # 将 Q, K, V 分成多个头，形状变为 (batch_size, num_heads, seq_len, dim_head)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1,
                                                                                 2)  # (batch_size, num_heads, seq_len, dim_head)
        K = K.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1,
                                                                                 2)  # (batch_size, num_heads, seq_len, dim_head)
        V = V.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1,
                                                                                 2)  # (batch_size, num_heads, seq_len, dim_head)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)

        # 使用Softmax归一化得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # 计算加权的值
        weighted_values = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, dim_head)

        # 合并多个头的输出 (batch_size, seq_len, attention_dim)
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attention_dim)

        # 输出层
        attention_output = self.out_linear(weighted_values)
        output_residual = attention_output + x  # (batch_size, seq_len, input_dim)

        return output_residual, attention_weights

class Multi_Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        # x shape: [batch_size, seq_len, channels, num_parts] -> [16, 16, 128, 3]

        batch_size, seq_len, channels, num_parts = x.shape

        # Rearrange to merge the last two dimensions
        x = rearrange(x, 'b s c p -> (b p) s c')  # Shape now [batch_size * num_parts, seq_len, channels]

        x = self.norm(x)  # Normalize over the merged input

        # qkv calculation, we expect shape [batch_size * num_parts, seq_len, channels]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Attention computation
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        #使用 to_out 映射到 [batch_size * num_parts, seq_len, dim * num_parts]
        out = self.to_out(out)

        # 将输出恢复为原始形状 [batch_size, seq_len, dim, num_parts]
        out = rearrange(out, '(b p) s c -> b s c p', b=batch_size, p=num_parts)


        return out, v


class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttention, self).__init__()
        # self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5  # 缩放因子

    def forward(self, query, key, value):
        # 线性变换
        Q = query  # [B, N, D]
        K = self.key_proj(key)  # [B, N, D]
        V = self.value_proj(value)  # [B, N, D]

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # [B, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N, N]

        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # [B, N, D]

        # 输出投影
        output = self.output_proj(attn_output)  # [B, N, D]
        return output
# class PCT_multi(nn.Module):
#     def __init__(self, token, n_points, channels= 192, first='Ture', head=4, dim_head=64, model = 'attention'):
#         super(PCT_multi, self).__init__()
#         self.model = model
#         self.first = first
#         self.token = token
#         self.n_points = n_points
#         self.channels = channels
#         self.norm = nn.LayerNorm(n_points*channels)
#         self.attn = Attention(n_points*channels, heads=head, dim_head=dim_head)
#         self.M_attn = Multi_Attention(n_points, heads=head, dim_head=dim_head)
#         self.fe = FeedForward(n_points*channels, 1024)
#         self.conv_blocks = nn.ModuleList()
#         self.bn_blocks = nn.ModuleList()
#         self.layers = nn.Sequential(
#             nn.Linear(2, 128),  # 第一层: 输入维度 2 -> 隐含维度 32
#             nn.ReLU(),
#             nn.Linear(128, 64),  # 第二层: 隐含维度 32 -> 隐含维度 16
#             nn.ReLU(),
#             nn.Linear(64, 3),  # 第三层: 隐含维度 16 -> 输出维度 3
#         )
#     def forward(self, xyz, feature):
#         C, _, _ = feature.shape
#         if self.first == 'Ture':
#             feature = self.layers(feature) + xyz
#         else:
#             feature = feature
#         xyz_sample = index_points(xyz, farthest_point_sample(xyz, self.token))
#         feature_token = index_points(feature, closest_points(self.n_points, xyz, xyz_sample))
#         if self.model == 'attention':
#             feature_token = feature_token.view(C, self.token, -1)
#             feature_token = self.norm(feature_token)
#             x, _ = self.attn(feature_token)
#             feature_token = rearrange(x, "b s (c p) -> b s c p", c=self.n_points, p=self.channels)
#         elif self.model == 'Multi_attention':
#             feature_token, _ = self.M_attn(feature_token)
#         feature_token = rearrange(feature_token, 'b s c p -> b (s c) p')
#         return feature_token
class PCT(nn.Module):
    def __init__(self, channels = 192, head=4, dim_head=64):
        super(PCT, self).__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.attn = Attention(channels, heads=head, dim_head=dim_head)
    def forward(self,feature):
        C, _, _ = feature.shape
        feature_token = self.norm(feature)
        x, _ = self.attn(feature_token)
        return x

class AttentionFusion(nn.Module):
    def __init__(self, input_channels, local_channels, output_channels, scaling_factor=1.0):
        """
        :param input_channels: 全局特征通道数，例如 256
        :param local_channels: 局部特征通道数，例如 64
        :param output_channels: 最终输出的通道数，例如 128
        :param scaling_factor: 注意力缩放因子，默认为 1.0
        """
        super(AttentionFusion, self).__init__()

        # 可变的超参数
        self.input_channels = input_channels  # 全局特征的通道数
        self.local_channels = local_channels  # 局部特征的通道数
        self.output_channels = output_channels  # 输出的通道数
        self.scaling_factor = scaling_factor  # 注意力缩放因子

        # 全局特征的查询变换（Q）
        self.query_transform = nn.Linear(self.input_channels, self.output_channels)

        # 局部特征的键（K）和值（V）变换
        self.key_transform = nn.Linear(self.local_channels, self.output_channels)
        self.value_transform = nn.Linear(self.local_channels, self.output_channels)
        self.out_transform = nn.Linear(self.local_channels, self.output_channels)
    def forward(self, global_features, local_features):
        """
        :param global_features: 全局特征，形状为 [batch_size, input_channels, 64]，例如 [16, 256, 64]
        :param local_features: 局部特征，形状为 [batch_size, 16, 64, local_channels]，例如 [16, 16, 64, 64]
        """
        B, S, P, C = local_features.shape
        # 1. 将全局特征通过变换得到查询（Q）
        global_features = F.normalize(global_features, p=2, dim=1)
        Q = self.query_transform(
            global_features.permute(0, 2, 1))  # [16, 64, input_channels] -> [16, 64, output_channels]

        # 2. 将局部特征通过变换得到键（K）和值（V）
        K = self.key_transform(local_features.view(-1,
                                                   self.local_channels))  # [16 * 16 * 64, local_channels] -> [16 * 16 * 64, output_channels]
        K = K.view(global_features.size(0), S, P, self.output_channels)  # [16, 16, 64, output_channels]

        V = self.value_transform(local_features.view(-1,
                                                     self.local_channels))  # [16 * 16 * 64, local_channels] -> [16 * 16 * 64, output_channels]
        V = V.view(global_features.size(0), S, P, self.output_channels)  # [16, 16, 64, output_channels]

        # 3. 计算注意力得分，进行点积和缩放
        attention_scores = torch.einsum('bqc,blkc->blq', Q,
                                        K)  # [16, 64, output_channels] . [16, 16, 64, output_channels] -> [16, 16, 64]
        attention_scores = attention_scores / (self.output_channels ** 0.5)  # 缩放

        # 4. 使用 softmax 得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [16, 16, 64]

        # 5. 使用注意力权重加权值（V）
        fused_features = torch.einsum('blq,blkc->blkc', attention_weights, V)  # [16, 16, 64, output_channels]

        # 6. 最终的融合特征：与局部特征拼接
        fused_features = torch.cat([local_features, fused_features], -1)  # [16, 16, 64, local_channels] + [16, 16, 64, output_channels]
        # fused_features = self.out_transform(fused_features) + local_features
        return fused_features


if '__main__' == __name__:
    feature = torch.randn(8, 64, 640)
    points = torch.randn(8, 512, 3)
    model = PCT(640)
    model.forward(feature)
    print(feature.shape)