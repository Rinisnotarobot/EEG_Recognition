import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from eeg_dataset import Subject
from torch.utils.data import ConcatDataset, random_split, DataLoader
from eeg_model import CNNLSTM
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time
from tqdm import tqdm

# 设置随机种子以确保可重复性
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数设置
num_epochs = 1000
learning_rate = 0.001
batch_size = 8
num_classes = 3  # 假设是二分类任务

# 设置早停参数
patience = 200
min_delta = 0.0001

root_path = '.\\ExtractedFeatures\\session1'
file_list = [os.path.join(root_path, file_name) for file_name in os.listdir(root_path)]
train_test_filename_split = [(item, file_list[:i] + file_list[i+1:]) for i, item in enumerate(file_list)]

# 保存每个被试的结果
results = []

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(dataloader, desc="训练中"):
        inputs = inputs.to(device)
        # 将labels转换为LongTensor类型
        labels = labels.long().to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 记录统计数据
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="验证中"):
            inputs = inputs.to(device)
            # 将labels转换为LongTensor类型
            labels = labels.long().to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 记录统计数据
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1

def test(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="测试中"):
            inputs = inputs.to(device)
            # 将labels转换为LongTensor类型
            labels = labels.long().to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    return test_acc, test_f1, conf_mat

# 记录总体结果
all_subject_results = {}

for i, train_test in enumerate(train_test_filename_split):
    print(f"\n===== 正在处理被试 {i+1}/{len(train_test_filename_split)} =====")
    subject_name = os.path.basename(train_test[0])
    print(f"测试文件: {subject_name}")
    
    # 准备数据
    test_dataset = Subject(train_test[0])
    train_val_dataset = ConcatDataset([Subject(f_path) for f_path in train_test[1]])
    
    # 计算训练集和验证集大小
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    
    # 使用带有生成器的random_split以确保可重复性
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 获取输入的形状
    sample_data, _ = next(iter(train_loader))
    print(f"输入数据形状: {sample_data.shape}")
    
    # 初始化模型
    model = CNNLSTM(num_classes=num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证阶段
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, F1分数: {train_f1:.4f}")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, F1分数: {val_f1:.4f}")
        
        # 检查早停条件
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"model/best_model_subject_{i+1}.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"早停触发！训练在epoch {epoch+1}停止")
                break
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(f"model/best_model_subject_{i+1}.pth"))
    
    # 测试阶段
    test_acc, test_f1, conf_mat = test(model, test_loader, criterion, device)
    print(f"\n测试结果 - 准确率: {test_acc:.4f}, F1分数: {test_f1:.4f}")
    print(f"混淆矩阵:\n{conf_mat}")
    
    # 保存结果
    all_subject_results[subject_name] = {
        'test_acc': test_acc,
        'test_f1': test_f1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'confusion_matrix': conf_mat
    }
    
    # 绘制学习曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title(f'被试 {i+1} 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.title(f'被试 {i+1} 准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"fig/subject_{i+1}_learning_curves.png")
    plt.close()

# 计算总体性能
avg_test_acc = np.mean([result['test_acc'] for result in all_subject_results.values()])
avg_test_f1 = np.mean([result['test_f1'] for result in all_subject_results.values()])

print("\n===== 总体结果 =====")
print(f"平均测试准确率: {avg_test_acc:.4f}")
print(f"平均测试F1分数: {avg_test_f1:.4f}")

# 保存结果
import pickle
with open('eeg_recognition_results.pkl', 'wb') as f:
    pickle.dump(all_subject_results, f)

print("所有结果已保存到 eeg_recognition_results.pkl")