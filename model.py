# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from snn_model import ResNet2StageSNN
from hu_module import AttentionFusionHU, SimpleAdditiveHU  # 添加这行导入语句


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, last_relu=False, downsample=None, stride2=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2 if stride2 else 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.last_relu = last_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x) if self.downsample is not None else x
        out += residual
        out = out[:, :, 1:-1, 1:-1].contiguous()
        return out if not self.last_relu else self.relu(out)


class ResNet2Stage(nn.Module):
    def __init__(self, firstchannels=64, channels=(64, 128), inchannel=3, block_num=(3, 4)):
        self.inplanes = firstchannels
        super(ResNet2Stage, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, firstchannels, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(firstchannels)
        self.relu = nn.ReLU(inplace=True)
        self.stage1 = self._make_layer(channels[0], block_num[0], last_relu=True, stride2=True)
        self.stage2 = self._make_layer(channels[1], block_num[1], last_relu=True, stride2=True)
        self.conv_out = nn.Conv2d(channels[1] * 4, channels[1] * 4, kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, planes, blocks, last_relu, stride2=False):
        block = Bottleneck
        downsample = None
        if self.inplanes != planes * block.expansion or stride2:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=3,
                          stride=2 if stride2 else 1, padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, last_relu=True, downsample=downsample, stride2=stride2)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, last_relu=(last_relu if i == (blocks - 1) else True)))

        return nn.Sequential(*layers)

    def step(self, x):
        x = self.conv1(x)  # stride = 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.stage1(x)  # stride = 4
        x = self.stage2(x)  # stride = 8
        x = self.conv_out(x)
        return x

    def forward(self, net_in):
        return torch.stack([self.step(net_in[..., step]) for step in range(net_in.shape[-1])], -1)


class DVS_SiamFC(torch.nn.Module):
    """DVS手势识别的混合神经网络（CNN+SNN）"""

    def __init__(self, num_gestures=11, hu_type='attention'):
        super().__init__()
        # 特征提取骨干网络
        self.aps_net = ResNet2Stage(inchannel=1, block_num=[1, 1])  # CNN分支，处理静态外观
        self.dvs_net = ResNet2StageSNN(inchannel=2, block_num=[1, 1])  # SNN分支，处理时序动态

        # --- 修改1: 定义并集成HU模块 ---
        cnn_out_channels = 512  # 根据ResNet2Stage的输出维度设定
        snn_out_channels = 512  # 根据ResNet2StageSNN的输出维度设定

        if hu_type == 'attention':
            self.hu = AttentionFusionHU(cnn_out_channels, snn_out_channels)
        elif hu_type == 'additive':
            self.hu = SimpleAdditiveHU(cnn_out_channels, snn_out_channels)
        else:
            raise ValueError(f"Unsupported HU type: {hu_type}")

        # --- 修改2: 优化分类头，增加正则化 ---
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(cnn_out_channels, 256),  # 输入维度是融合后特征的通道数
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_gestures)
        )

        # 批归一化层，稳定训练
        self.cnn_bn = nn.BatchNorm2d(cnn_out_channels)
        self.snn_bn = nn.BatchNorm2d(snn_out_channels)

    def forward(self, aps, dvs, aps_loc, dvs_loc, training=True):
        """前向传播"""
        # 1. 提取特征
        # CNN分支: 静态外观特征
        cnn_feat = self.aps_net.step(aps)  # [B, C_cnn, H, W]
        cnn_feat = self.cnn_bn(cnn_feat)

        # SNN分支: 时序动态特征
        snn_feat_volume = self.dvs_net(dvs)  # [B, C_snn, H, W, T]
        # 聚合时序维度: 使用平均池化 (也可尝试最大池化)
        snn_feat = snn_feat_volume.mean(dim=-1)  # [B, C_snn, H, W]
        snn_feat = self.snn_bn(snn_feat)

        # 2. 通过HU进行特征融合 (核心修改)
        fused_feat = self.hu(cnn_feat, snn_feat)  # [B, C_cnn, H, W]

        # 3. 分类
        logits = self.classifier(fused_feat)  # [B, num_gestures]

        if training:
            return {"logits": logits, "loss": torch.tensor(0.0, device=logits.device)}
        else:
            return {"logits": logits, "pred_gesture": torch.argmax(logits, dim=1)}

    def corr_up(self, x, k):
        """计算相关性（保持原有框架）"""
        c = torch.nn.functional.conv2d(x, k).unflatten(1, (x.shape[0], k.shape[0] // x.shape[0])).diagonal().permute(3,
                                                                                                                     0,
                                                                                                                     1,
                                                                                                                     2)
        return c

    @staticmethod
    def extract_clip(ff, clip_loc, clip_size):
        """提取特征片段（保持原有框架）"""
        bs, fs, h, w = ff.shape
        ch, cw = clip_size

        tenHorizontal = torch.linspace(-1.0, 1.0, cw).expand(1, 1, ch, cw) * cw / w
        tenVertical = torch.linspace(-1.0, 1.0, ch).unsqueeze(-1).expand(1, 1, ch, cw) * ch / h
        tenGrid = torch.cat([tenHorizontal, tenVertical], 1).to(ff.device)

        clip_loc[..., 0] /= w / 2
        clip_loc[..., 1] /= h / 2
        tenDis = clip_loc.unsqueeze(-1).unsqueeze(-1).type(torch.float32)

        tenGrid = (tenGrid.unsqueeze(1) + tenDis).permute(1, 0, 3, 4, 2)
        target_list = [F.grid_sample(input=ff, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=True) for
                       grid in tenGrid]

        return torch.stack(target_list, 1).flatten(end_dim=1)

    @staticmethod
    def gen_gt_cm(target_loc, map_size):
        """生成真实相关性图（保持原有框架）"""
        w, h = map_size

        tenHorizontal = torch.arange(0, w).expand(1, 1, 1, h, w) - w / 2 + 0.5
        tenVertical = torch.arange(0, h).unsqueeze(-1).expand(1, 1, 1, h, w) - h / 2 + 0.5
        tenGrid = torch.stack([tenHorizontal, tenVertical], 2).to(target_loc.device)

        target_loc = target_loc.unsqueeze(-1).unsqueeze(-1).type(torch.float32)
        dist = torch.norm(tenGrid - target_loc, dim=2)
        gt_cm = -1 + (dist < 2) * 1 + (dist < 1) * 1
        return gt_cm.permute(0, 1, 3, 4, 2)

