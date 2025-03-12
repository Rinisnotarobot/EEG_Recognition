import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class MultiChannelLSTMClassifier(nn.Module):
    def __init__(self, input_size_per_channel, hidden_size, num_layers, num_classes, num_channels=5,
                 bidirectional=False):
        super(MultiChannelLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_channels = num_channels

        # Create separate LSTM for each channel
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size_per_channel, hidden_size, num_layers,
                    batch_first=True, bidirectional=bidirectional)
            for _ in range(num_channels)
        ])

        direction = 2 if bidirectional else 1
        # FC layer takes combined outputs from all LSTMs
        self.fc1 = nn.Linear(hidden_size * direction * num_channels, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x, lengths):
        # x shape: (batch, max_seq_len, 62, 5)
        batch_size = x.size(0)
        channel_outputs = []

        # Process each channel with its dedicated LSTM
        for channel_idx in range(self.num_channels):
            # Extract this channel's data: (batch, seq_len, 62)
            channel_data = x[:, :, :, channel_idx]

            # Pack sequence for variable length processing
            packed_input = pack_padded_sequence(
                channel_data, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            # Process with channel-specific LSTM
            _, (h_n, _) = self.lstms[channel_idx](packed_input)

            # Get final hidden state
            if self.bidirectional:
                h_forward = h_n[-2, :, :]
                h_backward = h_n[-1, :, :]
                h = torch.cat((h_forward, h_backward), dim=1)
            else:
                h = h_n[-1, :, :]

            channel_outputs.append(h)

        # Combine all channel outputs
        combined = torch.cat(channel_outputs, dim=1)

        # Final classification
        self.fc1(combined)
        out = torch.relu(self.fc1(combined))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out