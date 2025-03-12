import torch
from torch.nn.utils.rnn import pad_sequence


def multi_channel_collate_fn(batch):
    # batch is a list of (data, label) where data has shape (T, 62, 5)
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)

    # Pad sequences along the time dimension
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, lengths, labels