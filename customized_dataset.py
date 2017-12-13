from torch.utils.data import Dataset
from IOUtil import get_glove_nips
import numpy as np
import json
import torch


class TextDataset(Dataset):
    def __init__(self, glove_path, text_reader=None, text_data=None):
        if not text_reader and not text_data:
            raise ValueError("At least one of the text path and text data should not be not empty")

        if text_reader and text_data:
            raise ValueError("text path is mutually exclusive with text data")

        self.glove_dict = get_glove_nips(glove_path)
        if text_reader:
            self.data_x = text_reader.read_text()
        else:
            self.data_x = text_data

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        sentence = self.data_x[index]
        sentence = sentence.split(" ")
        gloves = [self.glove_dict[word] for word in sentence]
        glove_array = np.array(gloves)
        return torch.FloatTensor(glove_array)
