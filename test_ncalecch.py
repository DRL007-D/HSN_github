import os
import numpy as np

def get_sensor_resolution(bin_file_path):
    """读取单个 .bin 文件，返回 (max_x, max_y) 或 None"""
    x_coords, y_coords = [], []
    try:
        with open(bin_file_path, 'rb') as f:
            while True:
                raw_data = f.read(5)
                if len(raw_data) < 5:
                    break
                # 小端序解析 40 位数据
                data = int.from_bytes(raw_data, byteorder='little')
                x = (data >> 32) & 0xFF
                y = (data >> 24) & 0xFF
                x_coords.append(x)
                y_coords.append(y)
        if not x_coords:
            return None
        max_x = max(x_coords)
        max_y = max(y_coords)
        return max_x, max_y
    except Exception as e:
        print(f"读取失败 {bin_file_path}: {e}")
        return None

def scan_dataset(root_dir):
    """遍历 root_dir 下所有 .bin 文件，输出传感器尺寸"""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 跳过 annotations 目录（如果存在）
        if 'annotations' in dirpath.lower():
            continue
        for fname in filenames:
            if fname.endswith('.bin'):
                full_path = os.path.join(dirpath, fname)
                res = get_sensor_resolution(full_path)
                if res:
                    max_x, max_y = res
                    print(f"{full_path}: {max_x+1} x {max_y+1}")
                else:
                    print(f"{full_path}: 无事件数据或读取失败")

if __name__ == "__main__":
    # 请根据实际数据集路径修改
    data_root = "./n_caltech101_data/Caltech101"
    scan_dataset(data_root)