# n_mnist_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tonic  # 用于数据转换
import warnings

class NMNISTDataset(Dataset):
    """N-MNIST数据集适配器，遵循工程框架"""

    def __init__(self, data_path: str, train: bool = True, T: int = 10, apply_augmentation: bool = True):
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.T = T
        self.apply_augmentation = apply_augmentation and train

        # 定义数据增强（针对APS帧）
        if self.apply_augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.transform = None

        # 1. 检查并准备数据（触发自动下载或转换）
        self._check_and_prepare_dataset(data_path)

        # 使用Tonic加载N-MNIST数据集
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 注意：此处不设置 transform，我们后面自己转换
            self.dataset = tonic.datasets.NMNIST(
                save_to=data_path,
                train=self.train,
                transform=None
            )

        # --- 新增：创建事件到帧的转换器 ---
        from tonic.transforms import ToFrame
        # N-MNIST 的传感器尺寸是 34x34，有2个极性通道
        self.sensor_size = tonic.datasets.NMNIST.sensor_size
        self.to_frame = ToFrame(sensor_size=self.sensor_size, n_time_bins=self.T)

        print(f"N-MNIST {'Train' if train else 'Test'} 加载完成: {len(self.dataset)} 个样本")

    def _check_and_prepare_dataset(self, data_path: str):
        """检查数据路径，如果为空则打印提示"""
        expected_npz_dir = os.path.join(data_path, f"frames_number_{self.T}_split_by_number")
        if not os.path.exists(expected_npz_dir):
            print(f"\n提示: N-MNIST 预处理数据目录不存在: {expected_npz_dir}")
            print("当首次实例化 `tonic.datasets.NMNIST` 时，它会自动：")
            print("  1. 下载原始 .aedat 文件（如果本地没有）")
            print("  2. 将其转换为 .npz 帧数据（如果本地没有）")
            print("  3. 保存到上述目录中")
            print("这可能需要一些时间，请确保网络连接并耐心等待...\n")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        """
        返回格式: (aps_frame, dvs_volume, aps_loc, dvs_loc, label)
        与 DVSGestureDataset 保持接口一致。
        """
        # 从Tonic数据集获取事件和标签
        # events 可能是一个结构化数组，label是整数
        events, label = self.dataset[index]

        # --- 关键修改：将事件转换为帧 ---
        # 使用之前定义好的转换器
        frames = self.to_frame(events)  # frames 形状: [T, 2, 34, 34]
        frames_tensor = torch.from_numpy(frames).float()  # 现在frames是标准数组，可以转换

        # --- 后续保持原有代码不变 ---
        # 构建 APS 输入 (CNN分支)
        aps_frame = frames_tensor.sum(dim=0, keepdim=True)  # [1, 2, 34, 34]
        aps_frame = aps_frame[:, 0:1, :, :]  # 只取正极性通道 [1, 1, 34, 34]
        aps_frame = aps_frame.squeeze(0)  # 变为 [1, 34, 34]
        # 归一化
        aps_max = aps_frame.max()
        if aps_max > 0:
            aps_frame = aps_frame / aps_max

        # --- 构建 DVS 输入 (SNN分支) ---
        dvs_volume = frames_tensor.permute(1, 2, 3, 0)  # [2, 34, 34, T]

        # --- 位置信息 (固定值) ---
        height, width = 34, 34
        aps_loc = torch.tensor([[height // 2, width // 2]], dtype=torch.float32).repeat(3, 1)  # [3, 2]
        dvs_loc = aps_loc.unsqueeze(-1).repeat(1, 1, self.T)  # [3, 2, T]

        # --- 应用数据增强 (仅对APS帧) ---
        if self.transform is not None:
            aps_frame = self.transform(aps_frame)

        return aps_frame, dvs_volume, aps_loc, dvs_loc, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    # 简单测试
    dataset = NMNISTDataset(data_path="./n_mnist_data", train=True, T=10)
    if len(dataset) > 0:
        aps, dvs, aps_loc, dvs_loc, label = dataset[0]
        print(f"测试通过!")
        print(f"  APS形状: {aps.shape}")  # 应为 [1, 34, 34]
        print(f"  DVS形状: {dvs.shape}")  # 应为 [2, 34, 34, 10]
        print(f"  标签: {label}")