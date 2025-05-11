import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dataset import SEEDDataset
from model import CNN_GRU

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Training parameters
batch_size = 64
num_epochs = 150
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 5-fold cross validation for each subject
subs = [str(i) for i in range(1, 16)]
seq_len = 12
num_folds = 5

# Dictionary to store results
results = {}

for sub in subs:
    print(f"\nStarting training for subject {sub}")
    
    # Load dataset for current subject
    dataset = SEEDDataset(sub=sub, seq_len=seq_len)
    print(f"Dataset size for subject {sub}: {len(dataset)}")
    
    # Print data shape for debugging
    sample_data, sample_label = dataset[0]
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample label: {sample_label}")
    
    # Initialize 5-fold cross validation
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Store accuracies for each fold
    fold_accuracies = []
    fold_f1_scores = []
    all_cms = []
    
    # Get all indices
    indices = list(range(len(dataset)))
    
    # Train and validate for each fold
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        print(f"\nSubject {sub} - Fold {fold+1} training started")
        
        # Create data loaders for current fold
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
        
        # Initialize model, loss function and optimizer
        model = CNN_GRU(num_class=3, num_ch=4, seq_len=seq_len).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop - only training, no intermediate validation
        for epoch in range(num_epochs):
            # Training mode
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = train_correct / train_total
            
            # Print results every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Train loss: {train_loss/len(train_loader):.4f}, Train accuracy: {train_acc:.4f}")
        
        # Save final model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/sub_{sub}_fold_{fold+1}_final.pth')
        print(f"Subject {sub} - Fold {fold+1} training completed, final model saved")
        
        # Complete validation evaluation after training
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
                
                # 收集预测和标签用于计算F1和混淆矩阵
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        final_val_acc = val_correct / val_total
        
        # 计算F1分数 (macro平均)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        all_cms.append(cm)
        
        print(f"Subject {sub} - Fold {fold+1} final validation accuracy: {final_val_acc:.4f}, F1 score: {f1:.4f}")
        print(f"Confusion matrix:\n{cm}")
        
        fold_accuracies.append(final_val_acc)
        fold_f1_scores.append(f1)
    
    # 计算总体混淆矩阵
    total_cm = np.sum(all_cms, axis=0)
    
    # Calculate and store average accuracy for this subject
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    
    results[sub] = {
        'fold_accuracies': fold_accuracies,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'fold_f1_scores': fold_f1_scores,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'confusion_matrix': total_cm
    }
    
    print(f"\n5-fold cross-validation results for subject {sub}:")
    print(f"Average accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Average F1 score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Fold accuracies: {fold_accuracies}")
    print(f"Fold F1 scores: {fold_f1_scores}")
    print(f"Overall confusion matrix:\n{total_cm}")
    
    # 绘制并保存每个受试者的混淆矩阵
    plt.figure(figsize=(8, 6))
    cm_normalized = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
               xticklabels=['Negative', 'Neutral', 'Positive'],
               yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title(f'Subject {sub} - Confusion Matrix (%)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # 创建结果文件夹
    os.makedirs('results/confusion_matrices', exist_ok=True)
    plt.savefig(f'results/confusion_matrices/cm_subject_{sub}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/confusion_matrices/cm_subject_{sub}.pdf', bbox_inches='tight')

# Print results for all subjects
print("\nAverage accuracy for all subjects:")
all_accs = [results[sub]['mean_acc'] for sub in subs]
all_f1s = [results[sub]['mean_f1'] for sub in subs]
print(f"Overall average accuracy: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")
print(f"Overall average F1 score: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")

# Create results folder
os.makedirs('results', exist_ok=True)

# 绘制平均混淆矩阵
all_cms = np.array([results[sub]['confusion_matrix'] for sub in subs])
avg_cm = np.mean(all_cms, axis=0).astype(int)

plt.figure(figsize=(10, 8))
avg_cm_normalized = avg_cm.astype('float') / avg_cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(avg_cm_normalized, annot=True, fmt='.1f', cmap='Blues',
           xticklabels=['Negative', 'Neutral', 'Positive'],
           yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Average Confusion Matrix Across All Subjects (%)', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.tight_layout()
plt.savefig('results/average_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('results/average_confusion_matrix.pdf', bbox_inches='tight')
print("Average confusion matrix saved to results/average_confusion_matrix.png and .pdf")

# Plot bar chart of average accuracies for all subjects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 准确率图表
subjects = [f'S{sub}' for sub in subs]
mean_accs = [results[sub]['mean_acc'] for sub in subs]
std_accs = [results[sub]['std_acc'] for sub in subs]

# Create bar chart for accuracy
bars1 = ax1.bar(subjects, mean_accs, yerr=std_accs, capsize=5, alpha=0.7)

# Add value labels to each bar
for bar, acc in zip(bars1, mean_accs):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

# Add horizontal average line
ax1.axhline(y=np.mean(all_accs), color='r', linestyle='--', alpha=0.7)
ax1.text(0, np.mean(all_accs) + 0.01, f'Average: {np.mean(all_accs):.3f}', 
         color='r', ha='left', va='bottom')

# Add title and labels
ax1.set_title('Average Accuracy by Subject', fontsize=16)
ax1.set_xlabel('Subject', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# F1分数图表
mean_f1s = [results[sub]['mean_f1'] for sub in subs]
std_f1s = [results[sub]['std_f1'] for sub in subs]

# Create bar chart for F1
bars2 = ax2.bar(subjects, mean_f1s, yerr=std_f1s, capsize=5, alpha=0.7)

# Add value labels to each bar
for bar, f1 in zip(bars2, mean_f1s):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{f1:.3f}', ha='center', va='bottom', fontsize=9)

# Add horizontal average line
ax2.axhline(y=np.mean(all_f1s), color='r', linestyle='--', alpha=0.7)
ax2.text(0, np.mean(all_f1s) + 0.01, f'Average: {np.mean(all_f1s):.3f}', 
         color='r', ha='left', va='bottom')

# Add title and labels
ax2.set_title('Average F1 Score by Subject', fontsize=16)
ax2.set_xlabel('Subject', fontsize=14)
ax2.set_ylabel('F1 Score', fontsize=14)
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# Save figures
plt.savefig('results/metrics_by_subject.png', dpi=300, bbox_inches='tight')
plt.savefig('results/metrics_by_subject.pdf', bbox_inches='tight')
print("Metrics bar chart saved to results/metrics_by_subject.png and .pdf")

# Save results to CSV file
results_df = pd.DataFrame({
    'Subject': subs,
    'Mean_Accuracy': [results[sub]['mean_acc'] for sub in subs],
    'Std_Accuracy': [results[sub]['std_acc'] for sub in subs],
    'Mean_F1': [results[sub]['mean_f1'] for sub in subs],
    'Std_F1': [results[sub]['std_f1'] for sub in subs]
})
results_df.to_csv('results/subject_metrics.csv', index=False)
print("Metrics data saved to results/subject_metrics.csv")

