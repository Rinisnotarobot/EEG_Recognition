import os
from torch.utils.data import Dataset

from data_preprocessing import prepare_data, load_label


class Subject(Dataset):
    def __init__(self,f_path) -> None:
        super().__init__()
        self.data = prepare_data(f_path)
        self.label = load_label()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]