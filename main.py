import inspect
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from args_parser import parse_args, print_args
from dataset_manager import DatasetManager
from model import DVS_SiamFC


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def interactive_menu():
    print("\n" + "=" * 60)
    print("    CNN-SNN混合神经网络 - 交互式启动菜单")
    print("=" * 60)

    print("\n请选择要使用的数据集 (输入数字 1-3):")
    print("  [1] dvs_gesture128 (DVS128 Gesture)")
    print("  [2] cifar10_dvs_sj (CIFAR10-DVS)")
    print("  [3] n_mnist (N-MNIST)")

    dataset_map = {"1": "dvs_gesture128", "2": "cifar10_dvs_sj", "3": "n_mnist"}
    mode_map = {"1": "train", "2": "test", "3": "both"}

    dataset_choice = None
    while dataset_choice not in dataset_map:
        dataset_choice = input("\n请输入数据集编号 (1-3): ").strip()
        if dataset_choice not in dataset_map:
            print("输入无效，请重新输入数字 1, 2, 3。")
    dataset_name = dataset_map[dataset_choice]

    print("\n请选择运行模式 (输入 1, 2 或 3):")
    print("  [1] train")
    print("  [2] test")
    print("  [3] both")

    mode_choice = None
    while mode_choice not in mode_map:
        mode_choice = input("\n请输入模式编号 (1, 2 或 3): ").strip().lower()
        if mode_choice not in mode_map:
            print("输入无效，请重新输入 1, 2 或 3。")
    mode = mode_map[mode_choice]

    data_path = None
    use_custom = input("\n是否使用自定义数据路径? (y/n, 默认n): ").strip().lower()
    if use_custom == "y":
        data_path = input("请输入您的自定义数据路径: ").strip()
        if not os.path.exists(data_path):
            print(f"警告: 路径 '{data_path}' 不存在")

    model_path = None
    if mode in ["test", "both"]:
        use_custom_model = input("\n是否指定模型文件路径? (y/n, 默认n): ").strip().lower()
        if use_custom_model == "y":
            model_path = input("请输入模型文件(.ckpt)完整路径: ").strip()
            if not os.path.exists(model_path):
                print(f"警告: 模型文件 '{model_path}' 不存在")

    class Args:
        def __init__(self):
            self.dataset = dataset_name
            self.mode = mode
            self.data_path = data_path
            self.model_path = model_path

    print("\n" + "=" * 60)
    print("选择完成，开始执行...")
    print("=" * 60)
    return Args()


def safe_torch_load(load_ckpt_path):
    load_kwargs = {"map_location": "cpu"}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = True
    except Exception:
        pass
    return torch.load(load_ckpt_path, **load_kwargs)


class FocalLabelSmoothingLoss(nn.Module):
    def __init__(self, class_weights=None, gamma: float = 1.5, smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, logits, target):
        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        log_prob = torch.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)
        focal_factor = (1.0 - prob).pow(self.gamma)
        loss = -(true_dist * focal_factor * log_prob)

        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        return loss.sum(dim=1).mean()


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def stratified_split(dataset, val_ratio=0.2, seed=42):
    labels = []
    for i in tqdm(range(len(dataset)), desc="提取标签用于分层划分"):
        labels.append(int(dataset[i][4].item()))

    indices = np.arange(len(dataset))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, labels))
    return train_idx.tolist(), val_idx.tolist(), labels


def train_siamfc(dataset_name="dvs_gesture128", data_path=None, model_path=None):
    DatasetManager.print_dataset_info(dataset_name)
    num_classes = DatasetManager.get_num_classes(dataset_name)
    best_ckpt_path, save_ckpt_path = DatasetManager.get_model_paths(dataset_name)

    if model_path:
        save_ckpt_path = model_path
        best_ckpt_path = model_path.replace(".ckpt", "_best.ckpt")

    if data_path is None:
        data_path = DatasetManager.get_default_data_path(dataset_name)

    DatasetClass = DatasetManager.get_dataset_class(dataset_name)

    epoch_num = 180
    save_period = 5
    batch_size = 16
    early_stop_patience = 30
    early_stop_min_delta = 0.15

    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(f"summary/train_{dataset_name}_{int(time.time())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"使用设备: {device}")

    net = DVS_SiamFC(num_gestures=num_classes, hu_type="attention").to(device)

    full_train_dataset = DatasetClass(data_path, split="train", T=10, apply_augmentation=True)
    train_idx, val_idx, labels = stratified_split(full_train_dataset, val_ratio=0.2, seed=42)

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    label_array = np.array(labels)
    class_counts = np.bincount(label_array, minlength=num_classes).astype(np.float32)
    class_weights = class_counts.sum() / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    train_labels = label_array[np.array(train_idx)]
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_data = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=2e-3)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=8e-4,
        epochs=epoch_num,
        steps_per_epoch=len(train_data),
        pct_start=0.15,
        div_factor=10,
        final_div_factor=100,
    )
    criterion = FocalLabelSmoothingLoss(class_weights=class_weights_t, gamma=1.5, smoothing=0.05)

    best_val_acc = 0.0
    early_stop_counter = 0
    best_model_state = None

    print(f"开始训练，共 {epoch_num} 个 epoch")
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, batch_size: {batch_size}")

    for epoch in range(epoch_num):
        net.train()
        total_loss, correct, total = 0.0, 0, 0

        mixup_enabled = epoch < int(epoch_num * 0.6)

        with tqdm(train_data, desc=f"Epoch {epoch + 1}/{epoch_num}") as train_bar:
            for data in train_bar:
                aps, dvs, aps_loc, dvs_loc, true_labels = [d.to(device, non_blocking=True) for d in data]

                if mixup_enabled:
                    aps, labels_a, labels_b, lam = mixup_data(aps, true_labels, alpha=0.2)
                    dvs, _, _, _ = mixup_data(dvs, true_labels, alpha=0.2)
                else:
                    labels_a, labels_b, lam = true_labels, true_labels, 1.0

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    net_out = net(aps, dvs, aps_loc, dvs_loc, training=True)
                    logits = net_out["logits"]

                    if mixup_enabled:
                        loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                    else:
                        loss = criterion(logits, true_labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += true_labels.size(0)
                correct += (predicted == true_labels).sum().item()

        avg_loss = total_loss / max(1, len(train_data))
        train_acc = 100.0 * correct / max(1, total)

        net.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for data in val_data:
                aps, dvs, aps_loc, dvs_loc, true_labels = [d.to(device, non_blocking=True) for d in data]
                net_out = net(aps, dvs, aps_loc, dvs_loc, training=False)
                logits = net_out["logits"]

                loss = criterion(logits, true_labels)
                val_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                val_total += true_labels.size(0)
                val_correct += (predicted == true_labels).sum().item()

        avg_val_loss = val_loss / max(1, len(val_data))
        val_acc = 100.0 * val_correct / max(1, val_total)

        writer.add_scalars("Loss", {"train": avg_loss, "val": avg_val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"\nEpoch {epoch + 1:3d}/{epoch_num} | "
            f"Train Loss: {avg_loss:.4f}, Acc: {train_acc:6.2f}% | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:6.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val_acc + early_stop_min_delta:
            best_val_acc = val_acc
            early_stop_counter = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save(best_model_state, best_ckpt_path)
            print(f"  -> 新最佳验证准确率: {best_val_acc:.2f}%")
        else:
            early_stop_counter += 1
            print(f"  -> 早停计数: {early_stop_counter}/{early_stop_patience}")

        if (epoch + 1) % save_period == 0:
            torch.save(net.state_dict(), save_ckpt_path)

        if early_stop_counter >= early_stop_patience:
            print(f"\n早停触发: epoch {epoch + 1}")
            break

    writer.close()

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    torch.save(net.state_dict(), save_ckpt_path)
    print(f"\n训练完成。最终模型: {save_ckpt_path}")
    print(f"最佳模型: {best_ckpt_path}")

    return net, dataset_name, data_path


def test_siamfc(dataset_name="dvs_gesture128", data_path=None, model_path=None):
    DatasetManager.print_dataset_info(dataset_name)
    num_classes = DatasetManager.get_num_classes(dataset_name)
    class_names = DatasetManager.get_class_names(dataset_name)
    best_ckpt_path, _ = DatasetManager.get_model_paths(dataset_name)

    load_ckpt_path = model_path if model_path else best_ckpt_path
    if data_path is None:
        data_path = DatasetManager.get_default_data_path(dataset_name)

    DatasetClass = DatasetManager.get_dataset_class(dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    net = DVS_SiamFC(num_gestures=num_classes, hu_type="attention").to(device)

    if os.path.exists(load_ckpt_path):
        state = safe_torch_load(load_ckpt_path)
        net.load_state_dict(state)
        print(f"加载模型: {load_ckpt_path}")
    else:
        print(f"警告: 未找到模型检查点 {load_ckpt_path}")
        return 0.0

    test_dataset = DatasetClass(data_path, train=False, T=10, apply_augmentation=False)
    test_data = DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=False)

    net.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for step, data in tqdm(enumerate(test_data), total=len(test_data), desc="测试进度"):
            aps, dvs, aps_loc, dvs_loc, true_labels = [d.to(device) for d in data]
            net_out = net(aps, dvs, aps_loc, dvs_loc, training=False)
            pred_gesture = net_out["pred_gesture"]

            total += 1
            is_correct = (pred_gesture == true_labels).sum().item()
            correct += is_correct

            all_predictions.append(pred_gesture.item())
            all_true_labels.append(true_labels.item())

            if step < 5:
                print(f"样本 {step + 1}:")
                print(f"  预测: {class_names[pred_gesture.item()]} (类别{pred_gesture.item()})")
                print(f"  真实: {class_names[true_labels.item()]} (类别{true_labels.item()})")

    accuracy = 100.0 * correct / max(1, total)
    print("\n测试结果报告:")
    print("  =========================================")
    print(f"  测试集总样本数: {total}")
    print(f"  正确预测数: {correct}")
    print(f"  错误预测数: {total - correct}")
    print(f"  测试准确率: {accuracy:.2f}%")
    print("  =========================================")

    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{dataset_name} Test Confusion Matrix\nAccuracy: {accuracy:.2f}%")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    out_png = f"test_confusion_matrix_{dataset_name}.png"
    plt.savefig(out_png)
    print(f"混淆矩阵已保存: {out_png}")
    plt.close()

    print("\n详细分类报告:")
    print(classification_report(all_true_labels, all_predictions, target_names=class_names, digits=4, zero_division=0))

    print("\n各类别性能分析:")
    for i in range(num_classes):
        if i < len(cm):
            tp = cm[i, i]
            total_class = cm[i, :].sum()
            if total_class > 0:
                class_acc = 100.0 * tp / total_class
                print(f"  类别{i} ({class_names[i]}): {tp}/{total_class} = {class_acc:.1f}%")

    return accuracy


def main():
    set_seed(42)

    if len(sys.argv) > 1:
        print("检测到命令行参数，使用参数解析模式...")
        args = parse_args()
    else:
        print("未提供命令行参数，进入交互式菜单模式...")
        args = interactive_menu()

    print_args(args)

    if args.mode == "train":
        train_siamfc(dataset_name=args.dataset, data_path=args.data_path, model_path=args.model_path)
    elif args.mode == "test":
        test_siamfc(dataset_name=args.dataset, data_path=args.data_path, model_path=args.model_path)
    elif args.mode == "both":
        _, dataset_name, data_path = train_siamfc(
            dataset_name=args.dataset,
            data_path=args.data_path,
            model_path=args.model_path,
        )
        test_siamfc(dataset_name=dataset_name, data_path=data_path, model_path=args.model_path)


if __name__ == "__main__":
    main()