import os
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader


# 定义 62 个通道名称的列表
SEED_CHANNEL_LIST = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]
# 定义 9×9 网格上各通道的位置映射
SEED_LOCATION_LIST = [
    ['-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-'],
    ['-', '-', '-', 'AF3',  '-', 'AF4', '-', '-', '-'],
    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    ['FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8'],
    ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
    ['TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8'],
    ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
    ['-', 'PO7','PO5','PO3','POZ','PO4','PO6','PO8','-'],
    ['-', '-', 'CB1','O1','OZ','O2','CB2','-','-']
]

# 生成通道到 (row, col) 坐标的映射字典
def format_channel_location_dict(channel_list, location_list):
    loc_arr = np.array(location_list)
    mapping = {}
    for ch in channel_list:
        pos = np.argwhere(loc_arr == ch)
        mapping[ch] = pos[0].tolist() if pos.size else None
    return mapping

SEED_CHANNEL_LOCATION_DICT = format_channel_location_dict(SEED_CHANNEL_LIST, SEED_LOCATION_LIST)


def map_trial_to_grid(trial: np.ndarray) -> np.ndarray:
    n_chan, n_t, n_bands = trial.shape
    grid = np.zeros((n_bands, n_t, 9, 9), dtype=np.float32)
    for idx, ch in enumerate(SEED_CHANNEL_LIST):
        loc = SEED_CHANNEL_LOCATION_DICT.get(ch)
        if loc is None:
            continue
        y, x = loc
        band_ts = trial[idx].T  # Shape: (n_bands, n_t)
        for b in range(n_bands):
            grid[b, :, y, x] = band_ts[b]
    return grid

if __name__ == "__main__":
    # 模拟一个trial
    trial = np.random.randn(62, 185, 5)
    grid = map_trial_to_grid(trial)
    print(grid.shape)
    # 可视化
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 展示一个band的grid
    plt.figure(figsize=(10, 10))
    sns.heatmap(grid[0, 0, :, :], cmap='viridis')
    plt.savefig('grid.png')
    plt.show()
