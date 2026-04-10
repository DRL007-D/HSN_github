import argparse
import sys
from dataset_manager import DatasetManager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练CNN-SNN混合神经网络')

    # 数据集选择参数
    available_datasets = DatasetManager.get_available_datasets()
    parser.add_argument('--dataset', type=str, default='n_mnist',
                        # 可选择以下四种数据集dvs_gesture128、cifar10_dvs、n_mnist、dvs_behavior
                        choices=available_datasets,
                        help=f'选择数据集: {available_datasets}')

    # 数据路径参数
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据集路径，默认为对应数据集的默认路径')

    # 模式选择
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'both'],
                        help='运行模式: train(训练), test(测试), both(训练+测试)')

    # 模型检查点路径
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型检查点路径，测试模式时使用')

    return parser.parse_args()


def print_args(args):
    """打印命令行参数"""
    print("\n命令行参数配置:")
    print(f"数据集: {args.dataset}")
    print(f"模式: {args.mode}")
    print(f"数据路径: {args.data_path if args.data_path else '使用默认路径'}")
    if args.model_path:
        print(f"模型路径: {args.model_path}")
    print()