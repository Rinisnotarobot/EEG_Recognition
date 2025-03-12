import scipy.io as sio
import os
import torch
from torch.utils.data import Dataset

class MultiChannelEEGDataset(Dataset):
    def __init__(self, root_path, label_file):
        self.root_path = root_path
        self.label_file = label_file

        self.samples_label = []
        self.labels = sio.loadmat(str(os.path.join(self.root_path, self.label_file)))['label'][0]

        for file in os.listdir(self.root_path):
            if file.endswith(".mat") and file != self.label_file:
                file_path = os.path.join(self.root_path, file)
                feature = sio.loadmat(file_path)
                for i in range(1, 16):
                    key = f'de_movingAve{i}'
                    if key in feature:
                        data = feature[key]
                        data = torch.from_numpy(data).float()
                        # Permute to (T, 62, 5) - keep original structure
                        data = data.permute(1, 0, 2)
                        # Don't reshape - we want to preserve the channels
                        self.samples_label.append([data, self.labels[i-1]])

        self._normalize()

    def _normalize(self):
        # Normalize each channel separately
        for i in range(len(self.samples_label)):
            data = self.samples_label[i][0]  # Shape (T, 62, 5)
            for channel in range(data.shape[2]):
                channel_data = data[:, :, channel]
                min_val = channel_data.min()
                max_val = channel_data.max()
                if max_val > min_val:  # Avoid division by zero
                    data[:, :, channel] = (channel_data - min_val) / (max_val - min_val)
            self.samples_label[i][0] = data

    def __len__(self):
        return len(self.samples_label)

    def __getitem__(self, idx):
        data, label = self.samples_label[idx]
        if label == -1:
            label = 2
        return data, label