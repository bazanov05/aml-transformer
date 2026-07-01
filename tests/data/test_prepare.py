import pytest
import torch
from unittest.mock import MagicMock
from src.data.prepare import train_tokenizer, prepare_dataset


@pytest.fixture
def mock_tokenizer():
    # create a fake tokenizer object to bypass c++ execution
    tokenizer = MagicMock()
    # force the encode method to return a known list of ids
    tokenizer.encode.return_value = [12, 45, 7]
    return tokenizer


def test_train_tokenizer(mock_tokenizer):
    test_text = "some raw financial text"
    test_path = "dummy_tokenizer.json"

    # run the function with our fake tokenizer
    train_tokenizer(mock_tokenizer, test_text, test_path)

    # verify it called the underlying c++ methods exactly once with correct args
    mock_tokenizer.train.assert_called_once_with(test_text)
    mock_tokenizer.save.assert_called_once_with(test_path)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires apple silicon")
def test_prepare_dataset(mock_tokenizer):
    test_text = "hello world"

    # run the function
    result_tensor = prepare_dataset(mock_tokenizer, test_text)

    # verify the encode method was triggered
    mock_tokenizer.encode.assert_called_once_with(test_text)

    # check if the output is actually a pytorch tensor
    assert isinstance(result_tensor, torch.Tensor)
    
    # check if values match what the mock returned
    assert result_tensor.tolist() == [12, 45, 7]
    
    # check required properties for embedding layers and m4 memory
    assert result_tensor.dtype == torch.int64
    assert result_tensor.device.type == "mps"
    