import torch
import torch.nn as nn 
from torch.nn.functional import cross_entropy


class BigramLanguageModel(nn.Module):
    """
    A foundational autoregressive language model where the probability 
    of the next token depends only on the single current token.
    """

    def __init__(self, vocab_size: int):
        """
        Initializes the bigram model. 
        
        The embedding table acts as a direct lookup matrix where each row 
        represents the current token, and the columns represent the raw scores 
        (logits) for what the next token will be.
        """
        super().__init__()
        self._vocab_size = vocab_size
        
        # in Bigram model Embedding has dims of (vocab_size, vocab_size)
        # each row is current token - each col is the probability 
        # of this token comming after the current one 
        self._embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, X: torch.Tensor, Y: torch.Tensor = None):
        """
        Executes the forward pass to compute token logits and calculate loss.
        
        Args:
            X: Input tensor of token IDs, shape (batch_size, context_length).
            Y: Target tensor of shifted token IDs, shape (batch_size, context_length). 
               Defaults to None during text generation.
               
        Returns:
            A tuple of (logits, loss). If Y is provided, tensors are flattened 
            to calculate Cross-Entropy loss.
        """
        logits = self._embedding(X)

        if Y is not None:
            # b = batch size, t = time (context length), c = channels (vocab size)
            b, t, c = logits.shape
            
            # flatten tensors for cross entropy
            logits = logits.view(b * t, c)
            Y = Y.view(b * t)
            
            # cross_entropy in pytorch does not require softmax before
            # it has in under the hood
            loss = cross_entropy(input=logits, target=Y)
        else:
            loss = None

        return logits, loss
    