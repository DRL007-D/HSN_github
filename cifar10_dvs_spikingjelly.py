import os
import warnings
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from spikingjelly.datasets import cifar10_dvs as sj_cifar10


class CIFAR10DVSSJDataset(Dataset):
    """
    CIFAR10-DVS loader with cached stratified split and lightweight augmentation.

    Output format:
      aps: [1, H, W]
      dvs: [2, H, W, T]
      aps_loc: [3, 2]
      dvs_loc: [3, 2, T]
      label: scalar LongTensor
    """

    def __init__(
        self,
        data_path: str,
        split=None,
        train=None,
        T: int = 10,
        apply_augmentation: bool = True,
        **kwargs,
    ):
        super().__init__()

        if split is not None:
            self.train = split.lower() == "train"
        elif train is not None:
            self.train = bool(train)
        else:
            self.train = True

        self.data_path = data_path
        self.T = T
        self.apply_augmentation = apply_augmentation and self.train

        self.aps_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(128, padding=4),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
            ]
        ) if self.apply_augmentation else None

        print("加载 CIFAR10-DVS...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.full_dataset = sj_cifar10.CIFAR10DVS(
                root=self.data_path,
                data_type="frame",
                frames_number=self.T,
                split_by="number",
            )

        print(f"总样本数: {len(self.full_dataset)}")

        self._load_or_create_split()
        self.indices = self.train_indices if self.train else self.test_indices
        print(f"{'训练' if self.train else '测试'}集大小: {len(self.indices)}")

    def _load_or_create_split(self):
        split_file = os.path.join(self.data_path, "split_indices.npz")

        if os.path.exists(split_file):
            print("加载已有划分")
            data = np.load(split_file, allow_pickle=True)
            self.train_indices = data["train"].tolist()
            self.test_indices = data["test"].tolist()
            return

        print("首次运行，扫描标签并构建分层划分...")
        indices_by_class: Dict[int, List[int]] = {}

        for i in tqdm(range(len(self.full_dataset)), desc="扫描标签"):
            _, label = self.full_dataset[i]
            indices_by_class.setdefault(int(label), []).append(i)

        rng = np.random.default_rng(42)
        train_indices = []
        test_indices = []

        for _, idx_list in sorted(indices_by_class.items()):
            idx_array = np.array(idx_list, dtype=np.int64)
            rng.shuffle(idx_array)
            n_train = int(len(idx_array) * 0.8)
            train_indices.extend(idx_array[:n_train].tolist())
            test_indices.extend(idx_array[n_train:].tolist())

        self.train_indices = train_indices
        self.test_indices = test_indices
        np.savez(split_file, train=np.array(train_indices), test=np.array(test_indices))
        print("划分已保存")

    def _temporal_jitter(self, frames: torch.Tensor) -> torch.Tensor:
        if self.T <= 2:
            return frames
        shift = int(torch.randint(low=-1, high=2, size=(1,)).item())
        return torch.roll(frames, shifts=shift, dims=0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        frames, label = self.full_dataset[real_idx]

        frames = torch.from_numpy(frames).float()
        if frames.shape[0] != self.T:
            frames = frames.permute(3, 0, 1, 2)

        if self.apply_augmentation:
            frames = self._temporal_jitter(frames)

        # Use both event polarities to construct APS-like guidance.
        aps = frames.mean(dim=0).sum(dim=0, keepdim=True)
        aps = aps / (aps.amax(dim=(1, 2), keepdim=True) + 1e-6)

        dvs = frames.permute(1, 2, 3, 0).contiguous()
        mean = dvs.mean(dim=(1, 2, 3), keepdim=True)
        std = dvs.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        dvs = (dvs - mean) / std

        if self.aps_transform is not None:
            aps = self.aps_transform(aps)

        _, h, w = aps.shape
        aps_loc = torch.tensor([[h // 2, w // 2]], dtype=torch.float32).repeat(3, 1)
        dvs_loc = aps_loc.unsqueeze(-1).repeat(1, 1, self.T)

        return aps, dvs, aps_loc, dvs_loc, torch.tensor(int(label), dtype=torch.long)