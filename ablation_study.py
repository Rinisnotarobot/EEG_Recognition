import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import threading
import seaborn as sns
from dataset import SEEDDataset
from model import CNN_GRU

# 定义各种消融实验模型
class CNN_Only(nn.Module):
    def __init__(self, num_class=3, num_ch=4, seq_len=3, dropout_rate=0.1):
        super(CNN_Only, self).__init__()
        
        self.CNN = nn.Sequential(
            nn.Conv2d(num_ch, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3)
        )
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        
        # 计算特征的大小
        self.fc = nn.Sequential(
            nn.Linear(576 * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        # 输入 x 的形状为 [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 重塑输入以便按时间序列顺序处理
        x = x.reshape(B * T, C, H, W)
        
        # 通过CNN处理每个时间步
        cnn_out = self.CNN(x)
        
        # 展平
        cnn_out = self.flatten(cnn_out)
        
        # 恢复序列并合并时间维度
        cnn_out = cnn_out.view(B, T, -1)
        cnn_out = cnn_out.reshape(B, -1)  # 合并所有时间步的特征
        
        # 应用dropout和分类
        cnn_out = self.dropout(cnn_out)
        output = self.fc(cnn_out)
        
        return output

class GRU_Only(nn.Module):
    def __init__(self, num_class=3, num_ch=4, seq_len=3, dropout_rate=0.1):
        super(GRU_Only, self).__init__()
        
        # 简化的特征提取，没有复杂的CNN
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_ch * 9 * 9, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # GRU层
        self.GRU = nn.GRU(
            input_size=128, 
            hidden_size=128, 
            batch_first=True,
            num_layers=2,
            dropout=0.1
        )
        
        self.dropout_gru = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128, num_class)

    def forward(self, x):
        # 输入 x 的形状为 [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 重塑输入并展平
        x = x.reshape(B * T, C, H, W)
        x = x.reshape(B * T, -1)  # 展平空间维度
        
        # 简单特征提取
        features = self.feature_extractor(x)
        
        # 恢复序列维度
        features = features.view(B, T, -1)
        
        # GRU处理
        gru_out, _ = self.GRU(features)
        
        # 获取最后一个时间步的输出
        last_output = gru_out[:, -1, :]
        
        # Dropout和分类
        last_output = self.dropout_gru(last_output)
        output = self.classifier(last_output)
        
        return output

class Simple_CNN_GRU(nn.Module):
    def __init__(self, num_class=3, num_ch=4, seq_len=3, dropout_rate=0.1):
        super(Simple_CNN_GRU, self).__init__()
        
        # 简化的CNN，只有两层
        self.CNN = nn.Sequential(
            nn.Conv2d(num_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3)
        )
        
        self.flatten = nn.Flatten()
        self.dropout_cnn = nn.Dropout(dropout_rate)
        
        # 单层GRU
        self.GRU = nn.GRU(
            input_size=576, 
            hidden_size=64, 
            batch_first=True,
            num_layers=1
        )
        
        self.dropout_gru = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(64, num_class)

    def forward(self, x):
        # 输入 x 的形状为 [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 重塑输入
        x = x.reshape(B * T, C, H, W)
        
        # 通过CNN处理
        cnn_out = self.CNN(x)
        cnn_out = self.flatten(cnn_out)
        cnn_out = self.dropout_cnn(cnn_out)
        
        # 恢复序列
        cnn_out = cnn_out.view(B, T, -1)
        
        # GRU处理
        gru_out, _ = self.GRU(cnn_out)
        last_output = gru_out[:, -1, :]
        
        # 分类
        last_output = self.dropout_gru(last_output)
        output = self.classifier(last_output)
        
        return output

class CNN_GRU_NoDropout(nn.Module):
    def __init__(self, num_class=3, num_ch=4, seq_len=3):
        super(CNN_GRU_NoDropout, self).__init__()
        
        # 与原始模型相同的CNN结构，但没有dropout
        self.CNN = nn.Sequential(
            nn.Conv2d(num_ch, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3)
        )
        
        self.flatten = nn.Flatten()
        
        # 没有dropout的GRU
        self.GRU = nn.GRU(
            input_size=576, 
            hidden_size=128, 
            batch_first=True,
            num_layers=2,
            dropout=0  # 移除层间dropout
        )
        
        self.classifier = nn.Linear(128, num_class)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        cnn_out = self.CNN(x)
        cnn_out = self.flatten(cnn_out)
        cnn_out = cnn_out.view(B, T, -1)
        gru_out, _ = self.GRU(cnn_out)
        last_output = gru_out[:, -1, :]
        output = self.classifier(last_output)
        return output

# 定义单个实验的函数
def run_single_experiment(experiment_config, sub="1", device_id=0):
    exp_name = experiment_config["name"]
    model_class = experiment_config["model_class"]
    params = experiment_config["params"]
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    
    print(f"\n开始实验: {exp_name} (设备: {device})")
    
    # 设置基本参数
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.0001
    
    # 获取序列长度参数
    seq_len = params.get("seq_len", 12)
    
    # 加载数据集
    dataset = SEEDDataset(sub=sub, seq_len=seq_len)
    print(f"数据集大小: {len(dataset)}")
    
    # 5折交叉验证
    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    indices = list(range(len(dataset)))
    fold_accuracies = []
    fold_f1_scores = []
    all_cms = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        print(f"\n实验 {exp_name} - 折 {fold+1} 开始训练")
        
        # 创建数据加载器
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=train_subsampler,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=val_subsampler,
            num_workers=2,
            pin_memory=True
        )
        
        # 初始化模型
        model = model_class(num_class=3, num_ch=4, seq_len=seq_len, **params).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = train_correct / train_total
            
            # 每10轮打印结果
            if (epoch + 1) % 10 == 0:
                print(f"实验 {exp_name} - 轮次 [{epoch+1}/{num_epochs}] - 训练损失: {train_loss/len(train_loader):.4f}, 训练准确率: {train_acc:.4f}")
        
        # 最终验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 收集所有预测和标签用于计算F1和混淆矩阵
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        final_val_acc = val_correct / val_total
        
        # 计算F1分数 (macro平均所有类别)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        all_cms.append(cm)
        
        print(f"实验 {exp_name} - 折 {fold+1} 最终验证准确率: {final_val_acc:.4f}, F1分数: {f1:.4f}")
        print(f"混淆矩阵:\n{cm}")
        
        fold_accuracies.append(final_val_acc)
        fold_f1_scores.append(f1)
    
    # 计算并存储平均准确率和F1分数
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    
    # 计算总体混淆矩阵
    total_cm = np.sum(all_cms, axis=0)
    
    result = {
        'name': exp_name,
        'fold_accuracies': fold_accuracies,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'fold_f1_scores': fold_f1_scores,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'confusion_matrix': total_cm
    }
    
    print(f"\n5折交叉验证结果 - 实验 {exp_name}:")
    print(f"平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"平均F1分数: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"各折准确率: {fold_accuracies}")
    print(f"各折F1分数: {fold_f1_scores}")
    print(f"总体混淆矩阵:\n{total_cm}")
    
    return result

# 主函数修改为支持同时在多个GPU上运行
def run_ablation_study():
    # 设置设备信息
    num_gpus = torch.cuda.device_count()
    print(f"可用GPU数量: {num_gpus}")
    
    # 实验配置
    experiments = [
        # {"name": "原始CNN-GRU", "model_class": CNN_GRU, "params": {}},  # 移除原始模型
        {"name": "仅CNN", "model_class": CNN_Only, "params": {}},
        {"name": "仅GRU", "model_class": GRU_Only, "params": {}},
        {"name": "简化CNN-GRU", "model_class": Simple_CNN_GRU, "params": {}},
        {"name": "无Dropout", "model_class": CNN_GRU_NoDropout, "params": {}},
    ]
    
    # 在不同的GPU上运行不同的实验（不使用多进程）
    results_list = []
    for i, exp in enumerate(experiments):
        # 选择GPU设备
        device_id = i % max(1, num_gpus)  # 如果没有GPU，使用CPU
        result = run_single_experiment(exp, "1", device_id)
        results_list.append(result)
    
    # 转换结果为字典格式
    results = {result['name']: {'fold_accuracies': result['fold_accuracies'], 
                               'mean_acc': result['mean_acc'], 
                               'std_acc': result['std_acc'],
                               'fold_f1_scores': result['fold_f1_scores'],
                               'mean_f1': result['mean_f1'],
                               'std_f1': result['std_f1'],
                               'confusion_matrix': result['confusion_matrix']} 
              for result in results_list}
    
    # 创建结果文件夹
    os.makedirs('ablation_results', exist_ok=True)
    
    # 绘制所有实验结果对比图
    plt.figure(figsize=(14, 8))
    exp_names = list(results.keys())
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 准确率图表
    mean_accs = [results[name]['mean_acc'] for name in exp_names]
    std_accs = [results[name]['std_acc'] for name in exp_names]
    bars1 = ax1.bar(exp_names, mean_accs, yerr=std_accs, capsize=5, alpha=0.7)
    ax1.set_title('平均准确率对比', fontsize=16)
    ax1.set_xlabel('实验', fontsize=14)
    ax1.set_ylabel('准确率', fontsize=14)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, acc in zip(bars1, mean_accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # F1分数图表
    mean_f1s = [results[name]['mean_f1'] for name in exp_names]
    std_f1s = [results[name]['std_f1'] for name in exp_names]
    bars2 = ax2.bar(exp_names, mean_f1s, yerr=std_f1s, capsize=5, alpha=0.7)
    ax2.set_title('平均F1分数对比', fontsize=16)
    ax2.set_xlabel('实验', fontsize=14)
    ax2.set_ylabel('F1分数', fontsize=14)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, f1 in zip(bars2, mean_f1s):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('ablation_results/ablation_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('ablation_results/ablation_metrics_comparison.pdf', bbox_inches='tight')
    print("指标对比图已保存到 ablation_results/ablation_metrics_comparison.png 和 .pdf")
    
    # 保存结果到CSV
    results_df = pd.DataFrame({
        'Experiment': exp_names,
        'Mean_Accuracy': [results[name]['mean_acc'] for name in exp_names],
        'Std_Accuracy': [results[name]['std_acc'] for name in exp_names],
        'Mean_F1': [results[name]['mean_f1'] for name in exp_names],
        'Std_F1': [results[name]['std_f1'] for name in exp_names]
    })
    results_df.to_csv('ablation_results/ablation_metrics.csv', index=False)
    print("消融实验指标数据已保存到 ablation_results/ablation_metrics.csv")
    
    # 绘制每个实验的混淆矩阵
    for name in exp_names:
        plt.figure(figsize=(8, 6))
        cm = results[name]['confusion_matrix']
        
        # 计算归一化的混淆矩阵（转为百分比）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 使用seaborn绘制更美观的混淆矩阵（百分比形式）
        sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=['负面', '中性', '正面'], 
                   yticklabels=['负面', '中性', '正面'])
        
        plt.title(f'{name} - 混淆矩阵 (%)', fontsize=14)
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.tight_layout()
        
        # 保存混淆矩阵图
        plt.savefig(f'ablation_results/confusion_matrix_{name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'ablation_results/confusion_matrix_{name}.pdf', bbox_inches='tight')
    
    print("各实验的混淆矩阵已保存到 ablation_results/ 目录")

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_ablation_study() 