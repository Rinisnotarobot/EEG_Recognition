import os
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from to_gird import map_trial_to_grid

root_dir = '/hy-tmp/extracted'
label_file = os.path.join(root_dir, 'label.mat')
label_data = loadmat(label_file)['label'][0]  # [ 1  0 -1 -1  0  1 -1  0  1  1  0 -1  0  1 -1]

session_list = ['session1', 'session2', 'session3']
sub_ids = [str(i) for i in range(1, 16)]

class SEEDDataset(Dataset):
    def __init__(self, seq_len, root_dir=root_dir,session=None, sub=None):
        self.root_dir = root_dir
        self.data = []
        self.label = []

        self.session = [session] if isinstance(session, str) else session or session_list
        self.sub = [sub] if isinstance(sub, str) else sub or sub_ids

        for session_name in self.session:
            session_path = os.path.join(self.root_dir, session_name)
            file_list = os.listdir(session_path)

            for sub_id in self.sub:
                sub_files = [f for f in file_list if f.startswith(f'{sub_id}_')]
                for file in sub_files:
                    file_path = os.path.join(session_path, file)
                    all_data = loadmat(file_path)

                    for i in range(1, 16):  # 遍历 trial
                        trial_key = f"de_LDS{i}"
                        if trial_key not in all_data:
                            continue
                        trial_data = all_data[trial_key][:, :, 1:5]  # (62,185,)
                        grid_data = map_trial_to_grid(trial_data)     # (4,180,9,9)
                        num_seg = grid_data.shape[1] // seq_len
                        # 划分为seq_len大小的小段
                        for j in range(num_seg):
                            data = grid_data[:, j*seq_len:(j+1)*seq_len, :, :]
                            self.data.append(data)
                            self.label.append(label_data[i - 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]            # (4,seq_len,9,9)
        # 转置为(seq_len,4,9,9)以匹配CNN_GRU模型的输入要求
        data = torch.tensor(data, dtype=torch.float32).permute(1, 0, 2, 3)
        # 确保标签是Long类型
        label = torch.tensor(self.label[index] + 1, dtype=torch.long)  # 将label从[-1,0,1]调整为[0,1,2]
        return data, label

if __name__ == "__main__":
    dataset = SEEDDataset(seq_len=20, root_dir=root_dir, session=None, sub="1",)
    print(f"数据集大小: {len(dataset)}")
    data, label = dataset[4]
    print(f"数据形状: {data.shape}")
    print(f"标签: {label}")
    print(f"标签类型: {label.dtype}")
    # 可视化
    # import matplotlib.pyplot as plt
    # import seaborn as sns
   
    # # 五张子图在同一张图
    # fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    # for i in range(4):
    #     sns.heatmap(data[i, 0, :, :], ax=axes[i], cmap='viridis')
    # plt.show()
    # # save
    # plt.savefig('grid.png')
    