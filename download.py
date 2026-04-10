from spikingjelly.datasets import ASLDVS

# 指定保存目录
root_dir = './data/asldvs_data'
# 首次运行会自动下载
dataset = ASLDVS(root=root_dir, train=True, use_frame=True)