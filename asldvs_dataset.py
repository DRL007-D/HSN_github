# asldvs_dataset.py
import os
import numpy as np
import torch
import warnings
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple
import tarfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
from torchvision.datasets.utils import extract_archive
from pathlib import Path
import scipy.io
import glob

# ========== 进度条支持 ==========
try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    # 如果未安装 tqdm，定义一个虚拟的 tqdm 函数
    def tqdm(iterable, desc="", **kwargs):
        print(desc)
        return iterable
# ===================================

# 导入spikingjelly的utils模块，用于np_savez
try:
    from spikingjelly.datasets import utils
    _SJ_UTILS_IMPORTED = True
except ImportError:
    _SJ_UTILS_IMPORTED = False
    print("警告: 无法导入spikingjelly.utils，将使用替代方案")


class ASLDVSDataset(Dataset):
    """ASL-DVS手语字母数据集适配器，支持预处理缓存"""

    def __init__(self, data_path: str, train: bool = True, T: int = 10,
                 apply_augmentation: bool = True, img_size: tuple = (180, 240),
                 train_split_ratio: float = 0.8, random_seed: int = 42,
                 cache_dir: str = None):
        """
        初始化ASL-DVS数据集适配器

        参数:
            data_path: 数据根目录路径
            train: 是否为训练模式
            T: 时间步数（事件流分割的帧数）
            apply_augmentation: 是否应用数据增强
            img_size: 图片大小 (height, width)，ASL-DVS原始为(180, 240)
            train_split_ratio: 训练集划分比例（默认0.8）
            random_seed: 随机种子，确保每次划分一致
            cache_dir: 预处理缓存目录（如果为None，则不使用缓存）
        """
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.T = T
        self.apply_augmentation = apply_augmentation and train
        self.img_size = img_size
        self.train_split_ratio = train_split_ratio
        self.random_seed = random_seed
        self.cache_dir = cache_dir

        # 设置随机种子
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)

        # ASL-DVS数据集有24个手语字母类别（A-Y，排除J）
        self.classes = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) != 'J']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        # 定义数据增强（仅用于训练时动态应用）
        if self.apply_augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            ])
        else:
            self.transform = None

        # 检查并准备数据集
        self._check_and_prepare_dataset(data_path)

        # 加载并划分数据
        print("\n开始预处理 ASL-DVS 数据集...")
        self.samples = self._load_and_split_samples()

        if len(self.samples) == 0:
            raise RuntimeError(f"在 {data_path} 中未找到任何ASL-DVS数据文件")

        # 如果启用缓存，创建缓存目录
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"预处理缓存目录: {self.cache_dir}")

        # 打印统计信息
        self._print_dataset_statistics()

    def _check_and_prepare_dataset(self, data_path: str):
        """检查数据集文件是否存在，并处理必要的转换"""
        # 检查数据目录结构
        expected_dirs = [
            os.path.join(data_path, "train"),
            os.path.join(data_path, "test"),
            os.path.join(data_path, "raw")
        ]

        # 如果没有任何预期目录，尝试查找.mat或.npz文件
        if not any(os.path.exists(d) for d in expected_dirs):
            print(f"警告: 在 {data_path} 中未找到标准ASL-DVS目录结构")
            print("ASL-DVS数据集需要手动准备:")
            print("1. 从OpenI镜像下载: pip install openi")
            print("2. 执行: openi dataset download OpenI/ASLDVS --local_dir ./ASLDVS --max_workers 10")
            print("3. 解压 ASLDVS.zip 获取 ICCV2019_DVS_dataset.zip")
            print("4. 进一步解压得到按字母分类的.mat文件")
            print("5. 将数据组织为: data_path/A/, data_path/B/, ...")

            # 检查是否有.mat文件可以直接使用
            mat_files = glob.glob(os.path.join(data_path, "**", "*.mat"), recursive=True)
            if mat_files:
                print(f"找到 {len(mat_files)} 个.mat文件，将尝试直接处理")
            else:
                # 创建虚拟数据用于测试
                self._create_dummy_data(data_path)

    def _create_dummy_data(self, data_path: str):
        """创建虚拟数据文件，用于测试代码结构"""
        print(f"\n⚠ 创建虚拟ASL-DVS数据用于测试（实际训练需真实数据）")

        # 为每个类别创建虚拟目录
        for class_name in self.classes[:3]:  # 只创建前3个类别用于测试
            class_dir = os.path.join(data_path, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # 创建少量虚拟.mat文件
            for i in range(5):
                mat_file = os.path.join(class_dir, f"sample_{i}.mat")
                if not os.path.exists(mat_file):
                    # 创建简单的虚拟MAT文件结构
                    import h5py
                    with h5py.File(mat_file, 'w') as f:
                        # 创建与原始ASL-DVS相似的结构
                        f.create_dataset('x', data=np.random.randint(0, 240, 100))
                        f.create_dataset('y', data=np.random.randint(0, 180, 100))
                        f.create_dataset('ts', data=np.random.rand(100) * 1000)
                        f.create_dataset('pol', data=np.random.randint(0, 2, 100))

        print(f"创建虚拟数据在: {data_path}")
        print("注意：虚拟数据无法用于实际训练，请尽快下载真实数据集")

    def _load_and_split_samples(self):
        """加载所有样本并按类别划分训练集和测试集"""
        import random

        # 收集所有样本
        all_samples_by_class = {cls: [] for cls in self.classes}

        print("正在扫描并加载 ASL-DVS 数据文件...")

        # 使用 tqdm 遍历所有类别，显示进度
        iterator = tqdm(self.classes, desc="处理类别", unit="类") if _TQDM_AVAILABLE else self.classes
        for class_name in iterator:
            class_dir = os.path.join(self.data_path, class_name)

            if not os.path.isdir(class_dir):
                # 尝试在子目录中查找
                pattern = os.path.join(self.data_path, "**", class_name, "*.mat")
                mat_files = glob.glob(pattern, recursive=True)

                if not mat_files:
                    # 尝试查找.npz文件（已转换的格式）
                    pattern = os.path.join(self.data_path, "**", class_name, "*.npz")
                    npz_files = glob.glob(pattern, recursive=True)

                    if not npz_files:
                        # 最后尝试查找任何.mat或.npz文件
                        pattern_all = os.path.join(self.data_path, "**", f"*{class_name}*.mat")
                        mat_files = glob.glob(pattern_all, recursive=True)

                        pattern_all = os.path.join(self.data_path, "**", f"*{class_name}*.npz")
                        npz_files = glob.glob(pattern_all, recursive=True)

                files = mat_files + npz_files
            else:
                # 在类别目录中查找文件
                mat_files = glob.glob(os.path.join(class_dir, "*.mat"))
                npz_files = glob.glob(os.path.join(class_dir, "*.npz"))
                files = mat_files + npz_files

            label = self.class_to_idx[class_name]
            for file_path in files:
                all_samples_by_class[class_name].append((file_path, label))

        # 按类别划分训练集和测试集
        train_samples = []
        test_samples = []

        for class_name, samples in all_samples_by_class.items():
            if not samples:
                continue

            # 随机打乱当前类别的样本
            random.shuffle(samples)

            # 计算划分点
            split_idx = int(len(samples) * self.train_split_ratio)

            if self.train:
                train_samples.extend(samples[:split_idx])
            else:
                test_samples.extend(samples[split_idx:])

        return train_samples if self.train else test_samples

    def _print_dataset_statistics(self):
        """打印数据集统计信息"""
        # 统计每个类别的样本数
        class_counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            class_counts[class_name] += 1

        mode = "训练" if self.train else "测试"
        print(f"\n{'=' * 60}")
        print(f"ASL-DVS {mode}集统计信息")
        print(f"{'=' * 60}")
        print(f"总样本数: {len(self.samples)}")
        print(f"类别分布:")

        total = len(self.samples)
        for class_name in self.classes:
            count = class_counts[class_name]
            if total > 0:
                percentage = (count / total * 100)
                print(f"  {class_name:3s}: {count:4d} ({percentage:5.1f}%)")

        print(f"{'=' * 60}")

    def _load_events(self, file_path: str):
        """加载事件数据，支持.mat和.npz格式"""
        if file_path.endswith('.npz'):
            # 加载已转换的.npz文件
            data = np.load(file_path)
            events = {
                'x': data['x'],
                'y': data['y'],
                't': data['t'],
                'p': data['p']
            }
        else:
            # 加载.mat文件
            try:
                import h5py
                with h5py.File(file_path, 'r') as f:
                    # 处理HDF5格式的.mat文件
                    events = {
                        'x': f['x'][:].squeeze(),
                        'y': f['y'][:].squeeze(),
                        't': f['ts'][:].squeeze(),
                        'p': f['pol'][:].squeeze()
                    }
            except:
                # 尝试用scipy.io加载旧格式
                data = scipy.io.loadmat(file_path)
                events = {
                    'x': 239 - data['x'].squeeze(),  # 翻转x坐标
                    'y': 179 - data['y'].squeeze(),  # 翻转y坐标
                    't': data['ts'].squeeze(),
                    'p': data['pol'].squeeze()
                }

        return events

    def _events_to_frames(self, events, T: int = 10):
        """将事件流转换为帧序列"""
        # 提取事件数据
        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        t = events['t']
        p = events['p'].astype(np.float32)  # 0或1，表示极性

        # 确保坐标在有效范围内
        height, width = self.img_size
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        # 如果没有时间戳，创建均匀分布
        if len(t) == 0:
            return np.zeros((T, 2, height, width), dtype=np.float32)

        # 归一化时间戳到[0, 1]
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min)
        else:
            t_norm = np.zeros_like(t)

        # 初始化帧
        frames = np.zeros((T, 2, height, width), dtype=np.float32)

        # 将事件分配到时间仓
        time_bins = np.floor(t_norm * T).astype(np.int32)
        time_bins = np.clip(time_bins, 0, T - 1)

        # 累积事件到帧
        for i in range(len(x)):
            t_bin = time_bins[i]
            polarity = int(p[i])
            frames[t_bin, polarity, y[i], x[i]] += 1.0

        # 归一化每帧
        for t in range(T):
            frame_max = frames[t].max()
            if frame_max > 0:
                frames[t] = frames[t] / frame_max

        return frames

    def _get_cache_path(self, index: int) -> str:
        """生成缓存文件的路径"""
        # 使用样本索引作为缓存文件名，确保唯一性
        return os.path.join(self.cache_dir, f"sample_{index}.pt")

    def _preprocess_sample(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行完整的预处理（不包含数据增强），返回标准化张量。
        该结果将被缓存。
        """
        file_path, label = self.samples[index]

        # 加载事件数据
        events = self._load_events(file_path)

        # 将事件转换为帧
        frames_np = self._events_to_frames(events, self.T)  # [T, 2, H, W]
        frames_tensor = torch.from_numpy(frames_np).float()

        # --- 构建 APS 输入 (CNN分支) ---
        # 对所有时间步的事件进行累积，生成静态帧
        aps_frame = frames_tensor.sum(dim=0, keepdim=True)  # [1, 2, H, W]
        aps_frame = aps_frame[:, 0:1, :, :]                # 只使用正极性通道 [1, 1, H, W]
        aps_frame = aps_frame.squeeze(0)                   # 变为 [1, H, W]

        # 归一化
        aps_max = aps_frame.max()
        if aps_max > 0:
            aps_frame = aps_frame / aps_max

        # --- 构建 DVS 输入 (SNN分支) ---
        dvs_volume = frames_tensor.permute(1, 2, 3, 0)  # [2, H, W, T]

        # --- 位置信息（固定中心点）---
        height, width = self.img_size
        aps_loc = torch.tensor([[height // 2, width // 2]], dtype=torch.float32).repeat(3, 1)  # [3, 2]
        dvs_loc = aps_loc.unsqueeze(-1).repeat(1, 1, self.T)  # [3, 2, T]

        label_tensor = torch.tensor(label, dtype=torch.long)

        return aps_frame, dvs_volume, aps_loc, dvs_loc, label_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回格式: (aps_frame, dvs_volume, aps_loc, dvs_loc, label)
        如果启用了缓存且缓存存在，直接加载缓存；否则执行预处理并保存到缓存。
        数据增强在缓存加载后动态应用。
        """
        # 尝试从缓存加载（带损坏检测）
        if self.cache_dir:
            cache_path = self._get_cache_path(index)
            if os.path.exists(cache_path):
                try:
                    cached = torch.load(cache_path)
                    required_keys = ['aps_frame', 'dvs_volume', 'aps_loc', 'dvs_loc', 'label']
                    if all(k in cached for k in required_keys):
                        aps_frame = cached['aps_frame']
                        dvs_volume = cached['dvs_volume']
                        aps_loc = cached['aps_loc']
                        dvs_loc = cached['dvs_loc']
                        label = cached['label']
                    else:
                        raise ValueError("缓存文件缺少必要字段")
                except Exception as e:
                    # 缓存损坏或无效：删除并重新生成
                    print(f"警告: 缓存文件损坏或无效 ({cache_path})，错误: {e}，将重新生成")
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
                    # 重新预处理
                    aps_frame, dvs_volume, aps_loc, dvs_loc, label = self._preprocess_sample(index)
                    try:
                        torch.save({
                            'aps_frame': aps_frame,
                            'dvs_volume': dvs_volume,
                            'aps_loc': aps_loc,
                            'dvs_loc': dvs_loc,
                            'label': label
                        }, cache_path)
                    except Exception as save_err:
                        print(f"警告: 保存缓存失败 {cache_path}: {save_err}")
            else:
                # 预处理并保存到缓存
                aps_frame, dvs_volume, aps_loc, dvs_loc, label = self._preprocess_sample(index)
                try:
                    torch.save({
                        'aps_frame': aps_frame,
                        'dvs_volume': dvs_volume,
                        'aps_loc': aps_loc,
                        'dvs_loc': dvs_loc,
                        'label': label
                    }, cache_path)
                except Exception as e:
                    print(f"警告: 保存缓存失败 {cache_path}: {e}")
        else:
            # 不使用缓存，直接预处理
            aps_frame, dvs_volume, aps_loc, dvs_loc, label = self._preprocess_sample(index)

        # # --- 动态应用数据增强（仅对 APS 帧）---
        # if self.transform is not None:
        #     aps_frame = self.transform(aps_frame)

        return aps_frame, dvs_volume, aps_loc, dvs_loc, label


def test_asldvs_dataset():
    """测试ASL-DVS数据集适配器（含缓存）"""
    # 测试路径
    data_path = "./asldvs_data"
    cache_dir = "./asldvs_cache_test"

    print("测试ASL-DVS数据集适配器（缓存模式）...")

    try:
        # 创建训练集实例（启用缓存）
        train_dataset = ASLDVSDataset(
            data_path=data_path,
            train=True,
            T=10,
            img_size=(180, 240),
            cache_dir=cache_dir
        )

        # 创建测试集实例（不启用缓存）
        test_dataset = ASLDVSDataset(
            data_path=data_path,
            train=False,
            T=10,
            img_size=(180, 240),
            cache_dir=None
        )

        print(f"\n训练集样本数: {len(train_dataset)}")
        print(f"测试集样本数: {len(test_dataset)}")

        if len(train_dataset) > 0 and len(test_dataset) > 0:
            # 测试缓存加载（第二次访问应使用缓存）
            print(f"\n首次访问样本0（预处理并缓存）...")
            aps_frame, dvs_volume, aps_loc, dvs_loc, label = train_dataset[0]
            print(f"  APS帧形状: {aps_frame.shape}")
            print(f"  DVS体积形状: {dvs_volume.shape}")
            print(f"  APS位置形状: {aps_loc.shape}")
            print(f"  DVS位置形状: {dvs_loc.shape}")
            print(f"  标签: {label.item()}, 类别: {train_dataset.idx_to_class[label.item()]}")

            # 再次访问同一索引，应直接从缓存读取（速度极快）
            import time
            start = time.time()
            _ = train_dataset[0]
            cache_load_time = time.time() - start
            print(f"  缓存加载耗时: {cache_load_time*1000:.2f} ms")

            # 检查输出形状
            assert aps_frame.shape == (1, 180, 240), f"APS形状错误: {aps_frame.shape}"
            assert dvs_volume.shape == (2, 180, 240, 10), f"DVS形状错误: {dvs_volume.shape}"
            assert aps_loc.shape == (3, 2), f"APS位置形状错误: {aps_loc.shape}"
            assert dvs_loc.shape == (3, 2, 10), f"DVS位置形状错误: {dvs_loc.shape}"

            return True
        else:
            print("错误: 数据集为空")
            return False

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_asldvs_dataset()
    if success:
        print("\n✓ ASL-DVS数据集适配器测试通过!")
    else:
        print("\n✗ ASL-DVS数据集适配器测试失败，请检查数据路径和格式。")