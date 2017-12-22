import collections
import multiprocessing
import re

import numpy as np
import torch
from src.customized_dataset import TextDataset
from src.customized_dataset import TextIndexDataset
from torch.utils.data import DataLoader

from src.IOUtil import get_glove_nips
from src.IOUtil import read_file

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def padding_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        batch_length = sorted([tensor.size()[0] for tensor in batch], reverse=True)
        return torch.stack(padding_batch(batch), 0, out=out), batch_length
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))

    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)

    elif isinstance(batch[0], collections.Mapping):
        return {key: padding_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [padding_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def padding_index_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        sorted_batch = sorted(batch, key=lambda x: -x.size()[0])
        batch_length = [len(ele) for ele in sorted_batch]
        return torch.stack(padding_index_batch(sorted_batch), 0, out=out), batch_length
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))

    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)

    elif isinstance(batch[0], collections.Mapping):
        return {key: padding_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [padding_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def padding_single(current_tensor, max_length):
    length_difference = max_length - len(current_tensor)
    if length_difference <= 0:
        return current_tensor

    padding_tensor = torch.zeros_like(torch.FloatTensor(length_difference, current_tensor.size(1)))
    return torch.cat((current_tensor, padding_tensor), 0)


def padding_index_batch(batch):
    def padding_index_single(current_tensor, max_length):
        length_difference = max_length - len(current_tensor)
        if length_difference <= 0:
            return current_tensor

        padding_tensor = torch.LongTensor(length_difference).zero_()
        result = torch.cat((current_tensor, padding_tensor))
        return result

    max_length = len(batch[0])
    return [padding_index_single(tensor, max_length) for tensor in batch]


def padding_batch(batch):
    batch_length = [len(ele) for ele in batch]
    max_length = max(batch_length)
    return [padding_single(tensor, max_length) for tensor in batch]


def create_data_loader(text_path, glove_path, batch_size):
    text_data = read_file(text_path, lambda x: x.strip())
    text_data_set = TextDataset(glove_path=glove_path, text_data=text_data)
    num_words = len(text_data_set.glove_dict)
    loader = DataLoader(dataset=text_data_set,
                        collate_fn=padding_collate,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=multiprocessing.cpu_count())
    return loader, num_words


def create_new_data_loader(args, path=''):
    if not path:
        path = args.train_path

    text_data = read_file(path, lambda x: x.strip())
    glove_dict = get_glove_nips(args.glove_path)
    num_words = len(glove_dict) + 1
    embedding_matrix = np.zeros((num_words, args.embedding_dimension))
    index = 1
    word_to_idx = {}
    for word, embedding in glove_dict.items():
        word_to_idx[word] = index
        embedding_matrix[index] = embedding
        index += 1

    index_dataset = TextIndexDataset(word_to_index=word_to_idx, text_data=text_data)
    loader = DataLoader(dataset=index_dataset,
                        collate_fn=padding_index_collate,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=multiprocessing.cpu_count(),
                        pin_memory=args.use_cuda)
    return loader, num_words, embedding_matrix