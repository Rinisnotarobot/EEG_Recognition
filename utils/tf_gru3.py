import os
import re
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import scipy.io as sio
import numpy as np

def custom_sort_key(filename):
    # 提取文件名中的数字部分
    prefix, date_part = filename.split('_')
    year_month_day = int(date_part.split('.')[0])
    return int(prefix), year_month_day

def sorted_filelist(root_path: str):
    return sorted([f for f in os.listdir(root_path) if f.endswith(".mat") and f != 'label.mat'], key=custom_sort_key)


def load_mat(file_path) -> list[np.ndarray]:
    last_timestep = 180
    experiments = []
    data = sio.loadmat(file_path)
    keys = [f"de_movingAve{idx}" for idx in range(1,16)]
    for key in keys:
        exp_data = data[key][:, -last_timestep:, :]
        experiments.append(exp_data.astype(np.float32))
    return experiments


def load_label(label_path: str='./ExtractedFeatures/label.mat') -> list[int]:
    data = sio.loadmat(label_path)
    label = np.squeeze(data["label"])
    return [x+1 for x in label]


def reshape_data(experiments: list[np.ndarray]):
    return [exp.transpose(0, 2, 1).reshape(310, 180) for exp in experiments]


def normalize_data(experiments: list[np.ndarray]) -> list[np.ndarray]:
    def zscore_entire_matrix(data: np.ndarray) -> np.ndarray:
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    return [zscore_entire_matrix(data) for data in experiments]


class singel_dataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.data = normalize_data(reshape_data(load_mat(file_path=file_path)))
        self.label = load_label()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]


class combined_dataset(Dataset):
    def __init__(self, index_list):
        super().__init__()
        self.root_path = './ExtractedFeatures'
        self.files = [sorted_filelist(self.root_path)[x] for x in index_list]
        self.datasets = []
        for path in self.files:
            full_path = full_path = os.path.join(self.root_path, path)
            self.datasets.append(singel_dataset(full_path))
        self.concat_dataset = ConcatDataset(self.datasets)

    def __len__(self):
        return len(self.concat_dataset)
    
    def __getitem__(self, index):
        return self.concat_dataset[index]


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size * 2, 1)

    def forward(self, gru_outputs):
        scores = self.attention_weights(gru_outputs)
        scores = torch.tanh(scores)
        attention_weights = F.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights * gru_outputs, dim=1)
        return context_vector, attention_weights


class TFAtBiGRU(nn.Module):
    def __init__(self,
                 input_size,
                 nhead,
                 tf_layers,
                 dim_ffn,
                 gru_hidden_size,
                 gru_layers,
                 num_classes,
                 dropout
                 ):
        super(TFAtBiGRU, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=nhead,
            dim_feedforward=dim_ffn,
            batch_first=True,
            dropout=dropout
        )
        self.num_gru_layers = gru_layers
        self.hidden_size = gru_hidden_size

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)

        self.gru = nn.GRU(input_size, gru_hidden_size, gru_layers,
                          batch_first=True, bidirectional=True, dropout=dropout if gru_layers > 1 else 0)

        self.attention = Attention(gru_hidden_size)

        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)

    def forward(self, x):
        """
        :param x: 输入序列，形状为 (batch_size, seq_len, input_size)
        :return: 模型的输出和注意力权重
        """
        # Step 1: Transformer Encoder 处理输入
        transformer_output = self.transformer_encoder(x)  # (batch_size, seq_len, input_size)

        # Step 2: 双向 GRU 处理 Transformer 的输出
        h0 = torch.zeros(self.num_gru_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        gru_outputs, _ = self.gru(transformer_output, h0)  # (batch_size, seq_len, hidden_size * 2)

        # Step 3: 应用注意力机制
        context_vector, attention_weights = self.attention(gru_outputs)  # (batch_size, hidden_size * 2)

        # Step 4: 全连接层生成最终输出
        output = self.fc(context_vector)  # (batch_size, num_classes)

        return output, attention_weights

if __name__ == '__main__':
    index = [i for i in range(0, 45, 3)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_index = [index[-1]]
    train_index = index[:14]
    train_dataset = combined_dataset(train_index)
    test_dataset = combined_dataset(test_index)
    input_size = 180
    nhead = 4
    tf_layers = 4
    dim_ffn = 256
    gru_hidden_size = 128
    gru_layers = 2
    num_classes = 3
    dropout = 0.3

    num_epo = 700
    model = TFAtBiGRU(input_size = 180, 
                      nhead = 18,
                      tf_layers = 1,
                      dim_ffn = 256,
                      gru_hidden_size = 256,
                      gru_layers = 2,
                      num_classes = 3,
                      dropout = 0.3).to(device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    model.train()
    for epoch in range(num_epo):
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.8f}")   

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs[0], 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Accuracy : {accuracy:.2f}%")