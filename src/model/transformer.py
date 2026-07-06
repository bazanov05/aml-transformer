import torch
import torch.nn as nn 
from src.model.attention import MultiHeadAttention


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


class TransformerBlock(nn.Module):
    """
    A single Transformer block implementing Pre-Layer Normalization.
    """
    def __init__(self, d_model: int, num_of_heads: int):
        """
        Initializes the Transformer block.
        
        Args:
            d_model: The hidden dimension size of the input and output tensors.
            num_of_heads: The number of parallel attention heads.
        """
        super().__init__()

        # create Attention and FFN blocks 
        self._multi_head_attention = MultiHeadAttention(d_model, num_of_heads)
        self._ffn = FeedForward(d_model)
        
        # initialize the stateful LayerNorm modules 
        self._ln1 = nn.LayerNorm(d_model)
        self._ln2 = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass applying Pre-Layer Normalization, 
        Multi-Head Attention, and a Feed-Forward Network.
        
        Args:
            X: Input tensor containing token features, shape (batch_size, seq_len, d_model).
               
        Returns:
            The transformed tensor of identical shape (batch_size, seq_len, d_model).
        """
        # pass the tensor through the initialized layers

        # updated_token = old_token + context
        X = X + self._multi_head_attention(self._ln1(X))

        # token after conclusions = token with context + conclusion 
        X = X + self._ffn(self._ln2(X))

        return X
