import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import warnings
import numpy as np
from tqdm import tqdm
from spikingjelly.datasets import cifar10_dvs as sj_cifar10


class CIFAR10DVSSJDataset(Dataset):
    """
    使用 SpikingJelly 的 CIFAR10-DVS 数据集适配器（支持手动划分训练/测试集）
    兼容 split 和 train 参数，完美对接现有训练框架
    """
    def __init__(self, data_path: str, split: str = 'train', train: bool = True,
                 T: int = 10, apply_augmentation: bool = True, **kwargs):
        super().__init__()
        # 参数兼容：优先使用 split，否则使用 train
        if split is not None:
            self.train = (split == 'train')
        else:
            self.train = train

        self.data_path = data_path
        self.T = T
        self.apply_augmentation = apply_augmentation and self.train

        # 定义数据增强（仅训练时有效）
        if self.apply_augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
            ])
        else:
            self.transform = None

        # 加载完整数据集（帧模式，自动缓存）
        print(f"正在从 SpikingJelly 加载 CIFAR10-DVS 完整数据集（首次运行会自动转换事件为帧）...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.full_dataset = sj_cifar10.CIFAR10DVS(
                root=self.data_path,
                data_type='frame',
                frames_number=self.T,
                split_by='number'
            )
        print(f"总样本数: {len(self.full_dataset)}")

        # 手动划分训练集和测试集（分层采样，保证类别分布一致）
        self._build_train_test_indices()

    def _build_train_test_indices(self):
        """根据标签分层划分训练集（80%）和测试集（20%）"""
        # 获取所有标签（一次性遍历，建立索引列表）
        labels = []
        indices_by_class = {}
        print("正在读取所有样本标签以进行分层划分...")
        for i in tqdm(range(len(self.full_dataset)), desc="扫描标签"):
            _, label = self.full_dataset[i]
            labels.append(label)
            indices_by_class.setdefault(label, []).append(i)

        # 对每个类别进行划分
        train_indices = []
        test_indices = []
        np.random.seed(42)  # 固定随机种子，保证可复现
        for label, idx_list in indices_by_class.items():
            # 随机打乱
            np.random.shuffle(idx_list)
            n_train = int(len(idx_list) * 0.8)
            train_indices.extend(idx_list[:n_train])
            test_indices.extend(idx_list[n_train:])

        if self.train:
            self.indices = train_indices
        else:
            self.indices = test_indices

        print(f"数据集划分完成: {'训练' if self.train else '测试'}集大小 = {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        frames, label = self.full_dataset[real_idx]
        frames_tensor = torch.from_numpy(frames).float()

        # 确保形状为 [T, 2, H, W]（SpikingJelly 默认是 [T, 2, 128, 128]）
        if frames_tensor.shape[0] != self.T:
            # 若形状为 [2, 128, 128, T] 则进行转置
            frames_tensor = frames_tensor.permute(3, 0, 1, 2)

        # --- 构建 APS 输入（CNN分支）---
        aps_frame = frames_tensor.sum(dim=0, keepdim=True)   # [1, 2, H, W]
        aps_frame = aps_frame[:, 0:1, :, :].squeeze(0)       # [1, H, W]
        if aps_frame.max() > 0:
            aps_frame = aps_frame / aps_frame.max()

        # --- 构建 DVS 输入（SNN分支）---
        dvs_volume = frames_tensor.permute(1, 2, 3, 0)       # [2, H, W, T]

        # --- 位置信息（固定中心点，切断标签与位置的虚假关联）---
        H, W = frames_tensor.shape[2], frames_tensor.shape[3]
        aps_loc = torch.tensor([[H // 2, W // 2]], dtype=torch.float32).repeat(3, 1)  # [3, 2]
        dvs_loc = aps_loc.unsqueeze(-1).repeat(1, 1, self.T)                         # [3, 2, T]

        # --- 应用数据增强（仅对APS帧）---
        if self.transform is not None:
            aps_frame = self.transform(aps_frame)

        return aps_frame, dvs_volume, aps_loc, dvs_loc, torch.tensor(label, dtype=torch.long)


# 简单测试代码（可选）
if __name__ == "__main__":
    dataset = CIFAR10DVSSJDataset(data_path="./cifar10_dvs_sj_data", split='train', T=10)
    print(f"训练集大小: {len(dataset)}")
    aps, dvs, aps_loc, dvs_loc, label = dataset[0]
    print(f"APS 形状: {aps.shape}")
    print(f"DVS 形状: {dvs.shape}")

