import os
import torch
from torch.utils.data import Dataset
import tonic
from tonic.transforms import ToFrame
import warnings
import torch.nn.functional as F
import random
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class NCaltech101Dataset(Dataset):
    """
    N-Caltech101 数据集适配器（带缓存加速版本 + 数据增强）
    """

    def __init__(self, data_path: str, split: str = "train", T: int = 10, apply_augmentation: bool = True):
        super().__init__()

        assert split in ["train", "val", "test"]
        self.split = split
        self.data_path = data_path
        self.T = T
        self.apply_augmentation = apply_augmentation and (split == "train")

        # 缓存路径
        self.cache_dir = os.path.join(self.data_path, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(
            self.cache_dir,
            f"{split}_T{T}.pt"
        )

        # 如果缓存存在，直接加载
        if os.path.exists(self.cache_file):
            print(f"✅ 加载缓存数据: {self.cache_file}")
            self.data = torch.load(self.cache_file)
        else:
            print("⚡ 未找到缓存，开始预处理数据（仅首次运行）...")
            self._check_dataset()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_dataset = tonic.datasets.NCALTECH101(
                    save_to=self.data_path,
                    transform=None,
                )

            # 构建划分
            class_indices = {}
            for idx in range(len(full_dataset)):
                _, label = full_dataset[idx]
                class_indices.setdefault(label, []).append(idx)

            train_idx, val_idx, test_idx = [], [], []
            random.seed(42)

            for label, indices in class_indices.items():
                random.shuffle(indices)
                n = len(indices)
                n_train = int(0.8 * n)
                n_val = int(0 * n)
                train_idx += indices[:n_train]
                val_idx += indices[n_train:n_train + n_val]
                test_idx += indices[n_train + n_val:]

            if split == "train":
                self.indices = train_idx
            elif split == "val":
                self.indices = val_idx
            else:
                self.indices = test_idx

            self.raw_dataset = full_dataset
            self.class_names = sorted(list(class_indices.keys()))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
            self.sensor_size = tonic.datasets.NCALTECH101.sensor_size
            self.to_frame = ToFrame(sensor_size=self.sensor_size, n_time_bins=self.T)

            # 预处理并缓存
            self.data = []
            self._preprocess_and_cache()

        # ========== 数据增强定义 ==========
        if self.apply_augmentation:
            # 几何变换：水平翻转、旋转、缩放裁剪、平移
            self.geometric_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(size=(128, 128), scale=(0.7, 1.0)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
            # 时间掩码概率
            self.time_mask_prob = 0.3
        else:
            self.geometric_transform = None
            self.time_mask_prob = 0.0

    def _check_dataset(self):
        expected_folder = os.path.join(self.data_path, "NCALTECH101")
        if not os.path.exists(expected_folder):
            os.makedirs(expected_folder, exist_ok=True)
            print("未检测到 N-Caltech101 数据，程序将自动下载...")

    def _preprocess_and_cache(self):
        target_size = (128, 128)
        for idx in tqdm(self.indices, desc=f"预处理 {self.split}"):
            events, label_str = self.raw_dataset[idx]
            frames = self.to_frame(events)
            frames_tensor = torch.from_numpy(frames).float()

            # APS
            aps_frame = frames_tensor.sum(dim=0, keepdim=True)
            aps_frame = aps_frame[:, 0:1, :, :].squeeze(0)
            if aps_frame.max() > 0:
                aps_frame = aps_frame / aps_frame.max()

            # DVS
            dvs_volume = frames_tensor.permute(1, 2, 3, 0)  # [C, H, W, T]

            # label
            if isinstance(label_str, str):
                label = self.class_to_idx[label_str]
            else:
                label = int(label_str)

            # resize
            aps_frame = F.interpolate(
                aps_frame.unsqueeze(0), size=target_size,
                mode='bilinear', align_corners=False
            ).squeeze(0)
            dvs_volume = F.interpolate(
                dvs_volume.permute(3, 0, 1, 2), size=target_size,
                mode='bilinear', align_corners=False
            ).permute(1, 2, 3, 0)

            aps_frame = aps_frame.contiguous()
            dvs_volume = dvs_volume.contiguous()

            H, W = target_size
            aps_loc = torch.tensor([[H // 2, W // 2]], dtype=torch.float32).repeat(3, 1)
            dvs_loc = aps_loc.unsqueeze(-1).repeat(1, 1, self.T)

            self.data.append((
                aps_frame, dvs_volume, aps_loc, dvs_loc,
                torch.tensor(label, dtype=torch.long)
            ))

        torch.save(self.data, self.cache_file)
        print(f"✅ 预处理完成，缓存已保存: {self.cache_file}")

    def _transform_volume(self, volume, transform, seed):
        """
        对 DVS 体积 [C, H, W, T] 应用相同的几何变换
        """
        C, H, W, T = volume.shape
        transformed_frames = []
        for t in range(T):
            frame = volume[:, :, :, t]  # [C, H, W]
            # 为保证与 APS 变换一致，使用相同的随机种子
            torch.manual_seed(seed)
            # 转为 PIL 图像并应用变换
            frame_pil = TF.to_pil_image(frame)
            frame_transformed = transform(frame_pil)
            frame_tensor = TF.to_tensor(frame_transformed)
            transformed_frames.append(frame_tensor)
        return torch.stack(transformed_frames, dim=-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        aps_frame, dvs_volume, aps_loc, dvs_loc, label = self.data[index]

        if self.geometric_transform is not None:
            # 固定随机种子，确保 APS 和 DVS 的几何变换一致
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)

            # 对 APS 帧应用几何变换（APS 是 [1, H, W] 形状，需要增加通道维度）
            aps_pil = TF.to_pil_image(aps_frame)
            aps_transformed = self.geometric_transform(aps_pil)
            aps_frame = TF.to_tensor(aps_transformed)  # [1, H, W]

            # 对 DVS 体积应用相同的变换
            dvs_volume = self._transform_volume(dvs_volume, self.geometric_transform, seed)

            # 时间掩码：随机丢弃部分时间步（将对应帧置零）
            if random.random() < self.time_mask_prob:
                mask_t = torch.rand(self.T) > 0.2  # 保留 80% 的时间步
                dvs_volume = dvs_volume * mask_t.float().view(1, 1, 1, -1)

        return aps_frame, dvs_volume, aps_loc, dvs_loc, label


# ================================
# 测试
# ================================
if __name__ == "__main__":
    dataset = NCaltech101Dataset(
        data_path="./n_caltech101_data",
        split="train",
        T=10,
        apply_augmentation=True
    )

    print("Dataset size:", len(dataset))

    aps, dvs, aps_loc, dvs_loc, label = dataset[0]
    print("APS:", aps.shape)
    print("DVS:", dvs.shape)
    print("Label:", label)