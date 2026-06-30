import torch
from aml_tokenizer import Tokenizer


def train_tokenizer(tokenizer: Tokenizer, training_data: str, path: str = "tokenizer.json"):
    """
    Trains the BPE tokenizer on raw text to build vocabulary and merge rules,
    then saves the resulting state to a JSON file.
    """
    tokenizer.train(training_data)  # find the most frequent pairs and assign new token IDs
    tokenizer.save(path)            # save rules to disk so we can load them during text generation


def prepare_dataset(tokenizer: Tokenizer, text: str) -> torch.Tensor:
    """
    Encodes raw text into a 1D tensor of token IDs.
    """
    encoded_ids = tokenizer.encode(text)
    return torch.tensor(encoded_ids, dtype=torch.int64, device="mps")