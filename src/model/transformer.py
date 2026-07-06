import torch
import torch.nn as nn 


class FeedForward(nn.Module):
    """
    Implements a position-wise Feed-Forward Network (FFN).
    
    Applies a two-layer linear transformation with a GELU non-linearity 
    independently and identically to each individual token position. 
    It expands the feature dimension to 4 * d_model and then projects 
    it back to the original d_model size.
    """
    def __init__(self, d_model: int):
        """
        Initializes the Feed-Forward network.
        
        Args:
            d_model: The hidden dimension size of the input and output tensors.
        """
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=4*d_model),
            nn.GELU(),
            nn.Linear(in_features=4*d_model, out_features=d_model)
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass on the input tensor.
        
        Args:
            X: Input tensor containing token features, shape (batch_size, seq_len, d_model).
               
        Returns:
            The transformed tensor of identical shape (batch_size, seq_len, d_model).
        """
        return self._model(X)
