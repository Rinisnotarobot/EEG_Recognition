import torch
import torch.nn as nn

class CNN_GRU(nn.Module):
    def __init__(self, num_class=3, num_ch=4, seq_len=3, dropout_rate=0.1):
        super(CNN_GRU, self).__init__()  # Proper inheritance
        
        self.CNN = nn.Sequential(
            nn.Conv2d(num_ch, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),  # Add dropout after second conv block
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3)
        )
        # 添加展平层，将卷积输出转换为一维向量
        self.flatten = nn.Flatten()
        
        # Add dropout before GRU layer
        self.dropout_cnn = nn.Dropout(dropout_rate)
        
        # GRU layer with specified dropout for recurrent connections
        self.GRU = nn.GRU(
            input_size=576, 
            hidden_size=128, 
            batch_first=True,
            num_layers=2,
            dropout=0.1,  # Only applicable with more than 1 layer
        )
        
        # Add dropout before the classifier
        self.dropout_gru = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(128, num_class)

    def forward(self, x):
        # 输入 x 的形状为 [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 重塑输入以便按时间序列顺序处理
        x = x.reshape(B * T, C, H, W)  # [B*T, C, H, W]
        
        # 通过CNN处理每个时间步
        cnn_out = self.CNN(x)  # [B*T, 64, 3, 3]
        
        # 展平
        cnn_out = self.flatten(cnn_out)  # [B*T, 64*3*3]
        
        # Apply dropout after CNN
        cnn_out = self.dropout_cnn(cnn_out)

        # 恢复序列
        cnn_out = cnn_out.view(B, T, -1)  # [B, T, 64*3*3]

        # 将序列输入到GRU
        gru_out, _ = self.GRU(cnn_out)  # [B, T, 128]
        
        # 获取最后一个时间步的输出
        last_output = gru_out[:, -1, :]  # [B, 128]
        
        # Apply dropout before classification
        last_output = self.dropout_gru(last_output)
        
        # 分类
        output = self.classifier(last_output)  # [B, num_class]
        
        return output
        

# 主函数测试代码
if __name__ == "__main__":
    # 创建一个随机输入张量，形状为 [批次大小, 时间步, 通道数, 高度, 宽度]
    batch_size = 4
    seq_len = 15
    channels = 4
    height = 9
    width = 9
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, channels, height, width)
    
    # 初始化模型
    model = CNN_GRU(num_class=3, num_ch=channels, seq_len=seq_len)
    
    # 前向传播
    output = model(x)
    
    # 打印输出形状
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 检查模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params}")



