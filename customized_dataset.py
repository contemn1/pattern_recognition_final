from torch.utils.data import Dataset
from IOUtil import read_file
from IOUtil import split_string
import numpy as np
import json
import torch

class TextDataset(Dataset):
    def __init__(self, glove_path, embedding_path, text_reader, text_data):
        if not text_reader and not text_data:
            raise ValueError("At least one of the text path and text data should not be not empty")

        if text_reader and text_data:
            raise ValueError("text path is mutually exclusive with text data")

        glove_list = read_file(glove_path, split_string)
        self.glove_dict = {ele[0]: np.fromstring(ele[1], sep=" ") for ele in glove_list}
        if text_reader:
            self.data_x = text_reader.read_text()
        else:
            self.data_x = text_data
        self.dimension = None
        self.sentence_embeddings = np.load(embedding_path)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        index, words = self.data_x[index]
        sentence_embedding = self.sentence_embeddings[index]
        new_words = [get_glove_array(word, self.glove_dict) for word in words if word]
        word_embeddings = np.hstack(new_words)
        all_embeddings = np.hstack((sentence_embedding, word_embeddings))
        if not self.dimension:
            self.dimension = all_embeddings.size
        x_tensor = torch.FloatTensor(all_embeddings)
        return x_tensor, self.data_y[index]

