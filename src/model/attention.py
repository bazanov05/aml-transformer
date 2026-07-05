import torch
import torch.nn as nn
from torch.nn.functional import softmax 


class SelfAttention(nn.Module):
    """
    Computes scaled dot-product attention for a single head.
    Allows each token to gather context from previous tokens
    via learned Query, Key, and Value projections.
    """
    def __init__(self, d_model: int, d_k: int, d_v: int):
        """
        Constructor - initializes 3 layers: Q(query), K(key), V(value).
        d_model: input embedding dimension
        d_k: dimension of query and key projections
        d_v: dimension of value projection and output (almost always equal to d_k)
        """
        super().__init__()
        self._W_K = nn.Linear(
            in_features=d_model, 
            out_features=d_k, 
            bias=False, 
        )

        self._W_Q = nn.Linear(
            in_features=d_model, 
            out_features=d_k, 
            bias=False, 
        )

        self._W_V = nn.Linear(
            in_features=d_model, 
            out_features=d_v, 
            bias=False, 
        )

        self._d_k = d_k
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass to compute vectors' shifts
        in models high-dimensional space.
        """
        # project input into query/key/value spaces
        Q = self._W_Q(X)    
        K = self._W_K(X)    
        V = self._W_V(X)    

        _, seq_len, _ = Q.shape 

        # create triangle causal matrix
        # diagonal and triangle under it are filled with 0, and above it with -inf
        # prev tokens are not allowed to attend to future ones during training
        # lower triangular mask — token at position i cannot attend to position j > i
        causal_mask = torch.zeros(seq_len, seq_len, device=X.device)
        causal_mask = causal_mask.masked_fill(torch.tril(torch.ones(seq_len, seq_len, device=X.device)) == 0, float('-inf'))

        # Q * K^T / sqrt(d_k)
        weights = (Q @ K.transpose(-2, -1)) / float(self._d_k ** 0.5)
        weights = weights + causal_mask # add mask to prevent learning from future tokens
        weights = softmax(weights, dim=-1)  # apply softmax to get probabilities

        # weighted sum of value vectors — each token absorbs context from attended positions
        # then we will add this new context to original Embedding to shift token 
        return weights @ V  
    