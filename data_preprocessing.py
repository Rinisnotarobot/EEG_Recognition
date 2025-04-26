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

# 定义文件排序的 key 函数，根据文件名前缀和日期排序
def custom_sort_key(filename):
    prefix, date_part = filename.split('_')  # '1_20210101.mat'
    year_month_day = int(date_part.split('.')[0])
    return int(prefix), year_month_day

# 获取指定路径下所有 .mat 文件（排除 label.mat），并按自定义顺序排序
def sorted_filelist(root_path: str):
    return sorted(
        [f for f in os.listdir(root_path) if f.endswith('.mat') and f != 'label.mat'],
        key=custom_sort_key
    )

# 加载单个 .mat 文件，返回一个 trial 列表，每个 trial 形状为 (62, 180, 5)
def load_mat(file_path: str) -> list[np.ndarray]:
    last_timestep = 180  # 保留最后 180 个时间点
    data = sio.loadmat(file_path)
    # 假设数据包含 de_movingAve1 到 de_movingAve15 共 15 个键
    keys = [f"de_movingAve{idx}" for idx in range(1, 16)]
    trials = []
    for key in keys:
        arr = data[key]
        trial = arr[:, -last_timestep:, :].astype(np.float32)  # 62×180×5
        trials.append(trial)
    return trials  # list of (62,180,5)

# 加载标签文件，返回标签列表，标签从 1 开始
def load_label(label_path: str = './ExtractedFeatures/label.mat') -> np.ndarray:
    data = sio.loadmat(label_path)
    label = np.squeeze(data['label'])  # 压缩成一维
    return (label.astype(int) + 1)  # ndarray, shape (num_trials,)

# 核心函数：将单个 trial (62,180,5) 映射为 (5,180,9,9)
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
    
    # Min-max normalization for each 9x9 matrix
    mins = np.amin(grid, axis=(-2, -1), keepdims=True)
    maxs = np.amax(grid, axis=(-2, -1), keepdims=True)
    denominator = maxs - mins
    denominator[denominator == 0] = 1.0  # Avoid division by zero
    grid = (grid - mins) / denominator
    return grid  # Shape: (n_bands, n_t, 9, 9)

# 将所有 trials 转成 CNN 输入格式: (num_trials, 5, 180, 9, 9)
def prepare_data(f_path: str):
    grids = []
    trials = load_mat(f_path)  # list of (62,180,5)
    for trial in trials:
        grids.append(map_trial_to_grid(trial))
    return grids