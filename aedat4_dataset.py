import os
import numpy as np
import torch
import warnings
import tarfile
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Tuple

#  添加全局变量跟踪导入状态
_DVS128GESTURE_IMPORTED = False
_DVS128GESTURE_IMPORT_PRINTED = False

# 导入 spikingjelly 的 DVS128Gesture
try:
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    _DVS128GESTURE_IMPORTED = True
except ImportError as e:
    # 只有在第一次导入失败时才打印错误信息
    if not _DVS128GESTURE_IMPORT_PRINTED:
        print(f"无法导入 DVS128Gesture: {e}")
        print("请安装: pip install spikingjelly")
        _DVS128GESTURE_IMPORT_PRINTED = True
    # 不立即退出，允许程序继续运行但会在后续使用时报错


class DVSGestureDataset(Dataset):
    """DVS128 Gesture手势分类数据集适配器"""

    def __init__(self, data_path: str, train: bool = True, T: int = 10, apply_augmentation: bool = True) -> None:
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.T = T
        self.apply_augmentation = apply_augmentation and train  # 仅训练时增强

        # 定义数据增强 (简单示例，可根据需要扩展)
        if self.apply_augmentation:
            # self.transform = transforms.Compose([
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     # 注意: 针对DVS事件数据的增强需要更精细的设计，此处为简单示例
            # ])
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),  # 增加随机旋转
                transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),  # 增加随机裁剪缩放
                # 可谨慎添加：transforms.ColorJitter(brightness=0.1, contrast=0.1), # 微弱亮度对比度变化
            ])
        else:
            self.transform = None

        # 检查导入状态
        if not _DVS128GESTURE_IMPORTED:
            print("错误: 未成功导入 DVS128Gesture 模块")
            print("请先安装: pip install spikingjelly")
            raise ImportError("DVS128Gesture 模块未导入")

        # 只在第一次成功导入时打印消息
        if not _DVS128GESTURE_IMPORT_PRINTED and _DVS128GESTURE_IMPORTED:
            print("成功导入 DVS128Gesture (spikingjelly)")
            globals()['_DVS128GESTURE_IMPORT_PRINTED'] = True

        # 检查数据集是否存在
        self._check_and_prepare_dataset(data_path)

        # 使用 spikingjelly 加载 DVS128Gesture 数据集
        # 注意：这里会捕获警告，防止spikingjelly尝试自动下载
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.dataset = DVS128Gesture(
                root=data_path,
                train=train,
                data_type='frame',
                frames_number=T,
                split_by='number'
            )

        # 只在第一次创建实例时打印加载完成消息
        if not hasattr(DVSGestureDataset, '_load_printed'):
            print(f"DVS128 Gesture {'Train' if train else 'Test'} 加载完成: {len(self.dataset)} 个样本")
            DVSGestureDataset._load_printed = True

    def _check_and_prepare_dataset(self, data_path: str):
        """检查数据集文件是否存在，如果不存在则提示手动下载"""
        download_dir = os.path.join(data_path, "download")
        tar_file = os.path.join(download_dir, "DvsGesture.tar.gz")

        # 检查下载目录
        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)
            if not hasattr(self, '_dir_created'):
                print(f"创建下载目录: {download_dir}")
                self._dir_created = True

        # 检查tar文件是否存在
        if not os.path.exists(tar_file):
            if not hasattr(self, '_file_not_found_printed'):
                print(f"未找到数据集文件: {tar_file}")
                print("\n请手动执行以下步骤:")
                print("1. 下载 DvsGesture.tar.gz 文件")
                print("2. 下载链接: https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794")
                print("3. 将下载的文件放置到以下位置:")
                print(f"   {tar_file}")
                self._file_not_found_printed = True

            # 创建模拟数据以允许代码继续运行（仅用于测试）
            self._create_dummy_data(tar_file)
        else:
            if not hasattr(self, '_file_found_printed'):
                print(f"找到数据集文件: {tar_file}")
                print(f"文件大小: {os.path.getsize(tar_file) / (1024 ** 3):.2f} GB")
                self._file_found_printed = True

            # 检查是否已解压
            extracted_dir = os.path.join(data_path, "DvsGesture")
            if not os.path.exists(extracted_dir):
                if not hasattr(self, '_extracting_printed'):
                    print("正在解压数据集文件...")
                    self._extracting_printed = True
                self._extract_tar_file(tar_file, data_path)
            else:
                if not hasattr(self, '_extracted_printed'):
                    print(f"✓ 数据集已解压到: {extracted_dir}")
                    self._extracted_printed = True

    def _create_dummy_data(self, tar_file: str):
        """创建虚拟数据文件，用于测试代码结构"""
        if not hasattr(self, '_dummy_data_created'):
            print("\n⚠ 创建虚拟数据文件用于测试（实际训练需真实数据）")
            with open(tar_file, 'wb') as f:
                f.write(b"Dummy dataset file - Please replace with real DvsGesture.tar.gz")
            print(f"创建虚拟文件: {tar_file}")
            print("注意：虚拟数据无法用于实际训练，请尽快下载真实数据集")
            self._dummy_data_created = True

    def _extract_tar_file(self, tar_file: str, extract_to: str):
        """解压tar.gz文件"""
        try:
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(path=extract_to)
            if not hasattr(self, '_extract_success_printed'):
                print(f"✓ 解压完成: {extract_to}")
                self._extract_success_printed = True
        except Exception as e:
            if not hasattr(self, '_extract_error_printed'):
                print(f"✗ 解压失败: {e}")
                print("请尝试手动解压:")
                print(f"  tar -xzf \"{tar_file}\" -C \"{extract_to}\"")
                self._extract_error_printed = True

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回 (aps_frame, dvs_volume, aps_loc, dvs_loc, label)
        关键修改：
        1. APS: 使用事件累积的静态帧，表示手势的整体外观 (空间特征)。
        2. DVS: 保留原始事件流或其紧凑表示，表示手势的动态过程 (时空特征)。
        3. 位置: 返回固定值，彻底切断标签与位置的虚假关联，迫使模型学习视觉特征。
        """
        # 从原始数据集获取数据
        frames, label = self.dataset[item]  # frames: [T, 2, 128, 128]
        frames_tensor = torch.from_numpy(frames).float()

        # --- 修改1: 优化 APS 输入 (CNN分支) ---
        # 方案A: 对所有时间步的事件进行累积，生成一个能反映手势完整轮廓的静态帧
        # 方案A: 对所有时间步的事件进行累积，生成一个能反映手势完整轮廓的静态帧
        aps_frame = frames_tensor.sum(dim=0, keepdim=True)  # [1, 2, 128, 128]
        aps_frame = aps_frame[:, 0:1, :, :]  # 只使用正极性通道 [1, 1, 128, 128]

        # 关键修改：移除第一个维度，使其变为 [1, 128, 128]
        aps_frame = aps_frame.squeeze(0)  # 从 [1, 1, 128, 128] 变为 [1, 128, 128]

        # 归一化
        aps_max = aps_frame.max()
        if aps_max > 0:
            aps_frame = aps_frame / aps_max

        # --- 修改2: 优化 DVS 输入 (SNN分支) ---
        # 保持原始的时空体积表示，以供SNN处理时序动态
        dvs_volume = frames_tensor.permute(1, 2, 3, 0)  # [2, 128, 128, T]

        # --- 修改3: 移除硬编码位置映射 ---
        # 返回固定位置（如图像中心），迫使模型必须从视觉内容中学习特征
        # 这是解决过拟合最关键的一步
        height, width = 128, 128
        aps_loc = torch.tensor([[height // 2, width // 2]], dtype=torch.float32).repeat(3, 1)  # [3, 2] 保持形状兼容

        # 创建时序位置 (所有时间步位置相同，因为目标是固定的)
        dvs_loc = self._create_temporal_positions(aps_loc, self.T)

        # --- 可选: 应用数据增强 (需谨慎处理事件数据) ---
        if self.transform is not None:
            # 这里主要对APS帧做增强。DVS事件的增强更复杂，可后续添加。
            aps_frame = self.transform(aps_frame)

        return aps_frame, dvs_volume, aps_loc, dvs_loc, torch.tensor(label, dtype=torch.long)

    def _create_temporal_positions(self, aps_loc: torch.Tensor, T: int) -> torch.Tensor:
        """创建时序位置标签 (位置在所有时间步不变)"""
        dvs_loc = aps_loc.unsqueeze(-1).repeat(1, 1, T)  # [3, 2, T]
        return dvs_loc

    def _gesture_label_to_position(self, label: int) -> torch.Tensor:
        """将手势标签转换为模拟的目标位置"""
        # 11个手势类别映射到图像中的不同位置
        positions_map = {
            0: [[40, 40], [80, 40], [120, 40]],  # hand_clapping
            1: [[40, 80], [80, 80], [120, 80]],  # right_hand_wave
            2: [[40, 120], [80, 120], [120, 120]],  # left_hand_wave
            3: [[60, 40], [100, 40], [140, 40]],  # right_arm_clockwise
            4: [[60, 80], [100, 80], [140, 80]],  # right_arm_counter_clockwise
            5: [[60, 120], [100, 120], [140, 120]],  # left_arm_clockwise
            6: [[20, 40], [60, 40], [100, 40]],  # left_arm_counter_clockwise
            7: [[20, 80], [60, 80], [100, 80]],  # arm_roll
            8: [[20, 120], [60, 120], [100, 120]],  # air_drums
            9: [[100, 100], [140, 100], [180, 100]],  # air_guitar
            10: [[64, 64], [64, 128], [128, 64]]  # other
        }

        # 返回对应手势的位置
        return torch.tensor(positions_map.get(label, [[64, 64], [64, 128], [128, 64]]), dtype=torch.float32)

    def _create_temporal_positions(self, aps_loc: torch.Tensor, T: int) -> torch.Tensor:
        """创建时序位置标签"""
        # aps_loc: [3, 2]
        # 扩展到时序维度: [3, 2, T]
        dvs_loc = aps_loc.unsqueeze(-1).repeat(1, 1, T)
        return dvs_loc


def abspath(path: str) -> str:
    """获取绝对路径"""
    return os.path.join(os.path.dirname(__file__), path)


if __name__ == "__main__":
    # 测试数据集加载
    dataset_path = "c:/HSN/HSN-hybrid sensing networks/aedat4_data"

    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        print("正在创建目录...")
        os.makedirs(dataset_path, exist_ok=True)

    dataset = DVSGestureDataset(
        data_path=dataset_path,
        train=True,
        T=10
    )

    if len(dataset) > 0:
        aps, dvs, aps_loc, dvs_loc = dataset[0]
        print(f"\n✓ 数据集加载成功")
        print(f"  APS形状: {aps.shape}")
        print(f"  DVS形状: {dvs.shape}")
        print(f"  APS位置: {aps_loc.shape}")
        print(f"  DVS位置: {dvs_loc.shape}")
    else:
        print(" 数据集为空")