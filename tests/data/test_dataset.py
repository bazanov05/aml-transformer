import pytest
import torch
from src.data.dataset import AMLDataset


@pytest.fixture
def dummy_ids():
    # create a simple tensor of 10 sequential integers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return torch.arange(10, dtype=torch.int64)


@pytest.fixture
def context_length():
    # define a short context window for testing
    return 3


def test_dataset_length(dummy_ids, context_length):
    dataset = AMLDataset(dummy_ids, context_length)
    
    # length should be total tokens minus context length (10 - 3 = 7)
    assert len(dataset) == 7


def test_dataset_getitem_types_and_shapes(dummy_ids, context_length):
    dataset = AMLDataset(dummy_ids, context_length)
    chunk, target = dataset[0]
    
    # check if outputs are actually pytorch tensors
    assert isinstance(chunk, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    
    # check if their shapes perfectly match the context length
    assert chunk.shape == (context_length,)
    assert target.shape == (context_length,)


def test_dataset_getitem_shifting(dummy_ids, context_length):
    dataset = AMLDataset(dummy_ids, context_length)
    
    # grab the very first window
    chunk, target = dataset[0]
    
    # chunk should be exactly [0, 1, 2]
    assert chunk.tolist() == [0, 1, 2]
    
    # target should be shifted exactly by 1: [1, 2, 3]
    assert target.tolist() == [1, 2, 3]
    
    # grab the very last valid window (index 6, since len is 7)
    last_chunk, last_target = dataset[6]
    
    # chunk should be [6, 7, 8]
    assert last_chunk.tolist() == [6, 7, 8]
    
    # target should be [7, 8, 9]
    assert last_target.tolist() == [7, 8, 9]
    