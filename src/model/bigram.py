import torch
import torch.nn as nn 
from torch.nn.functional import cross_entropy, softmax


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
            # it has it under the hood
            loss = cross_entropy(input=logits, target=Y)
        else:
            loss = None

        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens  starting from a seed context.
        
        Args:
            idx: Input tensor of token IDs, shape (batch_size, context_length).
            max_new_tokens: Integer count of how many new tokens to append.
               
        Returns:
            The extended tensor containing both the original context and all 
            newly sampled tokens, shape (batch_size, context_length + max_new_tokens).
        """
        while max_new_tokens > 0:
            # use only last tokens, in Bigram model only last token makes context 
            last_tokens = idx[:, -1]    # last token from every batch, shape becomes [B]
            last_tokens = torch.unsqueeze(last_tokens, dim=1)  # shape becomes [B, 1]
            logits, _ = self(last_tokens)   # we do not need loss for now 

            b, t, c = logits.shape
            # we need prob distribution over the vocab_size dim, which is last dim
            probabilities = softmax(logits.view(b * t, c), dim=-1)

            # use multinomial instead of argmax to make text more original
            # multinomial samples token based on it's prob
            # prob = 30% - token is sampled in 30% cases
            new_token = torch.multinomial(probabilities, num_samples=1)

            # add new_token to context dim
            idx = torch.cat(tensors=[idx, new_token], dim=1)

            max_new_tokens -= 1
        
        return idx
