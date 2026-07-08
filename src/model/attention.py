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
        self._dropout = nn.Dropout(p=0.1)
    
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

        # kill some connections between tokens, so model can learn context deeply
        weights = self._dropout(weights)

        # weighted sum of value vectors — each token absorbs context from attended positions
        # then we will add this new context to original Embedding to shift token 
        return weights @ V  


class MultiHeadAttention(nn.Module):
    """
    Runs multiple self-attention heads in parallel, each learning
    different attention patterns. Concatenates their outputs and
    projects back to d_model via a learned output projection W_O.
    """
    def __init__(self, d_model: int, num_of_heads: int):
        """
        d_model: input and output embedding dimension
        num_of_heads: number of parallel attention heads
        Note: d_model must be divisible by num_of_heads
        """
        if not d_model % num_of_heads == 0:
            raise ValueError("d_model must be divisible by num_of_heads")
        
        super().__init__()

        # last layer of attention - collects results from diff heads
        # and translates them to one unit answer
        self._W_O = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False
        )

        self._d_k = d_model // num_of_heads
        self._heads = nn.ModuleList()   # use ModuleList so torch can track parameters
        self._dropout = nn.Dropout(p=0.1)
        
        for _ in range(num_of_heads):
            self._heads.append(SelfAttention(d_model=d_model, d_k=self._d_k, d_v=self._d_k))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Runs all attention heads on the same input, concatenates
        their outputs and projects through W_O.
        
        X: [batch, seq_len, d_model]
        returns: [batch, seq_len, d_model]
        """
        results = []

        # collect results from all heads
        for head in self._heads:
            result_from_single_head = head(X)
            results.append(result_from_single_head)
        
        # each single result has [batch, seq_len, d_v]
        # d_v = d_model // num_heads, so to recreate d_model dim we just cat all the results
        # as we have exactly num_heads results, d_v size each
        results = torch.cat(results, dim=-1)

        results = self._W_O(results)
        results = self._dropout(results)   # kill some features

        return results
    