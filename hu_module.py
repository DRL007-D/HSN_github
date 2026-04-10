# hu_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusionHU(nn.Module):
    """
    基于通道注意力的混合单元 (HU)。
    动机: 让模型自动学习CNN特征和SNN特征在融合时各自的“重要性”权重。
    """

    def __init__(self, cnn_channels, snn_channels, reduction_ratio=16):
        super(AttentionFusionHU, self).__init__()
        # 全局平均池化，获取通道描述符
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 注意力生成网络
        # 输入: 来自两个分支的特征通道描述符之和
        # 输出: 2个权重 (cnn_weight, snn_weight)
        self.attention_net = nn.Sequential(
            nn.Linear(cnn_channels + snn_channels, (cnn_channels + snn_channels) // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((cnn_channels + snn_channels) // reduction_ratio, 2, bias=False),  # 输出两个分支的权重
            nn.Softmax(dim=1)  # 权重归一化，和为1
        )

    def forward(self, cnn_feat, snn_feat):
        """
        Args:
            cnn_feat: [B, C_cnn, H, W]
            snn_feat: [B, C_snn, H, W] (SNN分支通常已做时序聚合，如平均池化)
        Returns:
            fused_feat: [B, C_cnn, H, W]  (融合后的特征，保持与CNN特征相同维度以便后续处理)
        """
        batch_size = cnn_feat.size(0)

        # 获取每个特征的通道描述符
        c_desc = self.gap(cnn_feat).view(batch_size, -1)  # [B, C_cnn]
        s_desc = self.gap(snn_feat).view(batch_size, -1)  # [B, C_snn]

        # 生成注意力权重
        combined_desc = torch.cat([c_desc, s_desc], dim=1)  # [B, C_cnn+C_snn]
        attention_weights = self.attention_net(combined_desc)  # [B, 2]
        weight_cnn, weight_snn = attention_weights[:, 0], attention_weights[:, 1]

        # 将权重变形为 [B, 1, 1, 1] 以便广播
        weight_cnn = weight_cnn.view(batch_size, 1, 1, 1)
        weight_snn = weight_snn.view(batch_size, 1, 1, 1)

        # 加权融合
        # 注意: 假设C_cnn == C_snn。如果不相等，需要一个1x1卷积对齐维度。
        fused_feat = weight_cnn * cnn_feat + weight_snn * snn_feat

        return fused_feat


class SimpleAdditiveHU(nn.Module):
    """
    简单的加性混合单元，类似HSN论文中的基础HU。
    公式: fused = SF(t) + Conv1x1( ΔDF(Δt) )
    这里我们用一个1x1卷积来转换和匹配SNN特征的维度。
    """

    def __init__(self, cnn_channels, snn_channels):
        super(SimpleAdditiveHU, self).__init__()
        # 将SNN特征转换到与CNN特征相同的空间和通道维度
        self.snn_transform = nn.Conv2d(snn_channels, cnn_channels, kernel_size=1)
        # 可选的，在相加后加一个非线性激活
        self.act = nn.ReLU(inplace=True)

    def forward(self, cnn_feat, snn_feat):
        transformed_snn = self.snn_transform(snn_feat)
        fused = cnn_feat + transformed_snn
        return self.act(fused)