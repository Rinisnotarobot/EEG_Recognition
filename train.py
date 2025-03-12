# Parameters
import torch
from torch import nn
from torch.utils.data import DataLoader

from MC_LSTM import MultiChannelLSTMClassifier
from dataset import MultiChannelEEGDataset
from padding import multi_channel_collate_fn

input_size_per_channel = 62  # Each channel has 62 features
hidden_size = 256
num_layers = 4
num_classes = 3
num_channels = 5
num_epochs = 400
batch_size = 32
learning_rate = 0.001

# Create model
model = MultiChannelLSTMClassifier(
    input_size_per_channel=input_size_per_channel,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    num_channels=num_channels,
    bidirectional=True
)

# Create dataset and dataloader
dataset = MultiChannelEEGDataset(root_path='/content/drive/MyDrive/ExtractedFeatures', label_file='label.mat')
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multi_channel_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=multi_channel_collate_fn)

# Training loop would follow...
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for padded_sequences, lengths, labels in train_dataloader:
        padded_sequences = padded_sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(padded_sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.8f}")


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for padded_sequences, lengths, labels in test_dataloader:
        padded_sequences = padded_sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        outputs = model(padded_sequences, lengths)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {correct / total * 100:.2f}%")


