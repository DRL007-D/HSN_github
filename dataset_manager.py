# dataset_manager.py
import os
from typing import Dict, Any
import importlib

# 获取项目根目录
# 假设 dataset_manager.py 在项目根目录下
project_root = os.path.dirname(os.path.abspath(__file__))


class DatasetManager:
    """数据集管理器，统一管理不同DVS数据集的配置"""

    # 数据集配置字典
    _dataset_configs = {
        'dvs_gesture128': {
            'module': 'aedat4_dataset',
            'class_name': 'DVSGestureDataset',
            'num_classes': 11,
            'class_names': [
                'hand_clapping', 'right_hand_wave', 'left_hand_wave',
                'right_arm_clockwise', 'right_arm_counter_clockwise',
                'left_arm_clockwise', 'left_arm_counter_clockwise',
                'arm_roll', 'air_drums', 'air_guitar', 'other'
            ],
            'default_data_path': os.path.join(project_root, 'aedat4_data'),
            'description': 'DVS128 Gesture手势数据集',
            'model_prefix': 'DVSGESTURE128'
        },
        'n_mnist': {
            'module': 'n_mnist_dataset',
            'class_name': 'NMNISTDataset',
            'num_classes': 10,
            'class_names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'default_data_path': os.path.join(project_root, 'n_mnist_data'),
            'description': 'N-MNIST手写数字数据集',
            'model_prefix': 'NMNIST'
        },
        'dvs_behavior': {
            'module': 'dvs_behavior_dataset',
            'class_name': 'DVSBehaviorDataset',
            'num_classes': 7,
            'class_names': ['cigar', 'drinking', 'laser_pointer', 'pen', 'phone', 'remote', 'sugar'],
            'default_data_path': os.path.join(project_root, 'dvs_behavior_data'),
            'description': 'DVS Behavior行为识别数据集',
            'model_prefix': 'DVSBehavior'
        },
        # 在dataset_manager.py的_dataset_configs字典中修正
        'asldvs': {
            'module': 'asldvs_dataset',
            'class_name': 'ASLDVSDataset',
            'num_classes': 24,  # 正确的数量：A-Y，排除J
            'class_names': [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) != 'J'],
            'default_data_path': os.path.join(project_root, 'asldvs_data'),
            'description': 'ASL-DVS手语字母数据集（24个字母，A-Y排除J）',
            'model_prefix': 'ASLDVS'
        },
        'cifar10_dvs': {
            'module': 'cifar10_dvs_dataset',  # 对应的模块文件名（无.py后缀）
            'class_name': 'CIFAR10DVSDataset',  # 数据集类名
            'num_classes': 10,  # CIFAR10 有10个类别
            'class_names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'default_data_path': os.path.join(project_root, 'cifar10_dvs_data'),
            'description': 'CIFAR10-DVS 事件视觉数据集（10类物体识别）',
            'model_prefix': 'CIFAR10DVS'
        },
        'cifar10_dvs_sj': {
            'module': 'cifar10_dvs_spikingjelly',  # 你的新文件名（不含 .py）
            'class_name': 'CIFAR10DVSSJDataset',  # 你的新类名
            'num_classes': 10,
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck'],
            'default_data_path': os.path.join(project_root, 'cifar10_dvs_sj_data'),
            'description': 'CIFAR10-DVS 事件视觉数据集 (SpikingJelly实现)',
            'model_prefix': 'CIFAR10DVSSJ'
        },
        'n_caltech101': {
            'module': 'n_caltech101_dataset',
            'class_name': 'NCaltech101Dataset',
            'num_classes': 101,
            'class_names': None,  # 类别较多，一般不手动列出
            'default_data_path': os.path.join(project_root, 'n_caltech101_data'),
            'description': 'N-Caltech101事件视觉数据集（101类目标识别）',
            'model_prefix': 'NCaltech101'
        }
    }


    @classmethod
    def get_available_datasets(cls) -> list:
        """获取可用的数据集列表"""
        return list(cls._dataset_configs.keys())

    @classmethod
    def get_dataset_config(cls, dataset_name: str) -> Dict[str, Any]:
        """获取指定数据集的配置"""
        if dataset_name not in cls._dataset_configs:
            raise ValueError(f"未知的数据集: {dataset_name}。可选数据集: {cls.get_available_datasets()}")
        return cls._dataset_configs[dataset_name]

    @classmethod
    def get_dataset_class(cls, dataset_name: str):
        """获取数据集类"""
        config = cls.get_dataset_config(dataset_name)
        module_name = config['module']
        class_name = config['class_name']

        # 动态导入模块
        try:
            module = importlib.import_module(module_name)
            dataset_class = getattr(module, class_name)
            return dataset_class
        except (ImportError, AttributeError) as e:
            raise ImportError(f"无法导入数据集类 {class_name} 从模块 {module_name}: {e}")

    @classmethod
    def get_num_classes(cls, dataset_name: str) -> int:
        """获取数据集的类别数"""
        return cls.get_dataset_config(dataset_name)['num_classes']

    @classmethod
    def get_class_names(cls, dataset_name: str) -> list:
        """获取数据集的类别名称"""
        return cls.get_dataset_config(dataset_name)['class_names']

    @classmethod
    def get_default_data_path(cls, dataset_name: str) -> str:
        """获取数据集的默认路径"""
        return cls.get_dataset_config(dataset_name)['default_data_path']

    @classmethod
    def get_model_paths(cls, dataset_name: str, base_dir: str = "ckpt",
                        use_model_prefix: bool = True) -> tuple:
        """获取模型的保存路径"""
        if use_model_prefix:
            prefix = cls.get_dataset_config(dataset_name)['model_prefix']
            best_path = os.path.join(base_dir, f"{prefix}_SiamFC_best.ckpt")
            final_path = os.path.join(base_dir, f"{prefix}_SiamFC.ckpt")
        else:
            best_path = os.path.join(base_dir, f"{dataset_name}_SiamFC_best.ckpt")
            final_path = os.path.join(base_dir, f"{dataset_name}_SiamFC.ckpt")

        return best_path, final_path

    @classmethod
    def print_dataset_info(cls, dataset_name: str):
        """打印数据集信息"""
        config = cls.get_dataset_config(dataset_name)
        print(f"数据集: {dataset_name}")
        print(f"描述: {config['description']}")
        print(f"类别数: {config['num_classes']}")
        print(f"类别: {config['class_names']}")
        print(f"默认路径: {config['default_data_path']}")