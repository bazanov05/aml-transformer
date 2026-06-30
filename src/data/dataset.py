import torch
from torch.utils.data import Dataset


class AMLDataset(Dataset):
    def __init__(self, ids: torch.Tensor, context_length: int):
        """
        Initializes the dataset with a single sequence of token IDs.
        The sequence acts as both the input and the target simultaneously.
        """
        self._ids = ids
        self._context_length = context_length
    
    def __len__(self) -> int:
        # Total number of valid starting windows. Prevents the target slice 
        # from overshooting the edge of the array.
        return self._ids.shape[-1] - self._context_length
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract identical window sizes shifted by exactly 1 token
        chunks = self._ids[index : index + self._context_length]
        targets = self._ids[index + 1 : index + 1 + self._context_length]
        return chunks, targets
    