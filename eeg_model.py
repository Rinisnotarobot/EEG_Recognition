import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=3, input_channels=5):
        super(CNNLSTM, self).__init__()

        # CNN部分 - 单个时间步处理
        self.cnn_block = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x4x4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x2x2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 最后输出128x1x1
        )

        # CNN输出大小
        cnn_output_size = 128  # 经过自适应池化后就是128

        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # 分类器
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        输入: x [batch_size, n_bands, timesteps, H, W]
        """

        batch_size, n_bands, timesteps, H, W = x.shape

        # 把时间步展开，合并batch和timesteps，统一送进CNN
        x = x.permute(0, 2, 1, 3, 4)  # -> [batch_size, timesteps, channels, H, W]
        x = x.reshape(batch_size * timesteps, n_bands, H, W)  # -> [batch_size * timesteps, channels, H, W]

        # CNN提取特征
        x = self.cnn_block(x)  # -> [batch_size * timesteps, 128, 1, 1]
        x = x.view(batch_size, timesteps, -1)  # -> [batch_size, timesteps, 128]

        # LSTM处理
        lstm_out, _ = self.lstm(x)  # -> [batch_size, timesteps, hidden_size]

        # 取最后时间步输出
        out = lstm_out[:, -1, :]  # -> [batch_size, hidden_size]

        # 全连接层
        out = self.fc(out)  # -> [batch_size, num_classes]

        return out
