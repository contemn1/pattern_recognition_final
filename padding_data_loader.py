import torch
from customized_dataset import TextDataset
from IOUtil import read_file
from torch.utils.data import DataLoader
import multiprocessing
import numpy as np
import collections
import re

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
        return torch.stack(padding_batch(batch), 0, out=out)
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


def padding_batch(batch):
    max_length = len(batch[0])
    return [padding_single(tensor, max_length) for tensor in batch]


def create_data_loader(text_path, glove_path, batch_size):
    text_data = read_file(text_path, lambda x: x.strip())
    text_data_set = TextDataset(glove_path=glove_path, text_data=text_data)
    loader = DataLoader(dataset=text_data_set,
                        collate_fn=padding_collate,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=multiprocessing.cpu_count())
    return loader
