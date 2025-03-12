# Parameters
import torch
from torch import nn
from torch.utils.data import DataLoader

from MultiChannal.MC_LSTM import MultiChannelLSTMClassifier
from MultiChannal.dataset import MultiChannelEEGDataset
from MultiChannal.padding import multi_channel_collate_fn

input_size_per_channel = 62  # Each channel has 62 features
hidden_size = 128
num_layers = 2
num_classes = 3
num_channels = 5
num_epochs = 100
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
dataset = MultiChannelEEGDataset(root_path='../ExtractedFeatures', label_file='label.mat')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=multi_channel_collate_fn)

# Training loop would follow...
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for padded_sequences, lengths, labels in dataloader:
        padded_sequences = padded_sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(padded_sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")