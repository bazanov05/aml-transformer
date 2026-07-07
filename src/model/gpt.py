from src.model.transformer import TransformerBlock
import torch.nn as nn
import torch
from torch.nn.functional import cross_entropy, softmax


class GPT(nn.Module):
    """
    Decoder-only GPT-style language model.
    
    Combines token and positional embeddings, passes them through N stacked 
    TransformerBlocks, applies a final LayerNorm, and projects to vocab_size 
    logits for next-token prediction.
    
    Architecture:
        token embedding + positional embedding
        → N x TransformerBlock (MultiHeadAttention + FeedForward + residuals)
        → LayerNorm
        → Linear head (d_model → vocab_size)
        → logits
    """
    def __init__(self, vocab_size: int, d_model: int, num_of_heads: int, num_of_blocks: int, max_seq_len: int):
        super().__init__()

        self._max_seq_len = max_seq_len

        # create token and position embeddings 
        self._token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self._position_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=d_model)

        # ModuleList guarantees all parameters, gradients, and device placements are tracked perfectly
        self._transformer_blocks = nn.ModuleList()

        for _ in range(num_of_blocks):
            self._transformer_blocks.append(TransformerBlock(d_model, num_of_heads))

        self._last_layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self._linear_head = nn.Linear(in_features=d_model, out_features=vocab_size)
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Executes the forward pass through the full GPT architecture.
        
        Args:
            X: Input tensor of token IDs, shape (batch_size, seq_len).
            Y: Target tensor of shifted token IDs, shape (batch_size, seq_len).
               If provided, cross-entropy loss is computed and returned.
               Defaults to None during text generation.
        
        Returns:
            Tuple of (logits, loss).
            logits: raw scores over vocabulary, shape (batch_size, seq_len, vocab_size).
            loss: scalar cross-entropy loss if Y provided, otherwise None.
        """
        # create a list of positions based on seq_len size 
        positions = torch.arange(start=0, end=X.shape[-1], device=X.device)
        # take embedding meaning of all positions
        positions = self._position_embedding(positions)
        tokens = self._token_embedding(X)

        # enrich tokens' embeddings with their positions in sequence
        tokens = tokens + positions

        # go through every transformer block
        # and enrich each token with context from multi-head attetnion
        # and conslusion from FFN 
        for transformer_block in self._transformer_blocks:
            tokens = transformer_block(tokens)
        
        # normalize results
        tokens = self._last_layer_norm(tokens)
        # create logitc with math projection: d_model -> vocab_size
        # thanks to dot product we can see relations between tokens
        # now every row represtens similarity between curr token and other tokens in Model's vocab
        logits = self._linear_head(tokens)

        loss = None
        
        # if Y was provided - forward is used during the training
        # calculate the loss with cross entropy 
        if Y is not None:
            b, t, c = logits.shape
            logits_flat = logits.view(b * t, c)
            Y_flat = Y.view(b * t)  # Y contains right ids which should be predicted 
            loss = cross_entropy(logits_flat, Y_flat)
        
        return logits, loss 
    
    def generate(self, X: torch.Tensor, context_length: int) -> torch.Tensor:
        """
        Autoregressively generates new tokens based on the provided context.
        
        Args:
            X: Input tensor of context token IDs, shape (batch_size, seq_len).
            context_length: The number of new tokens to generate.
        
        Returns:
            Tensor containing the original context and the newly generated tokens, 
            shape (batch_size, seq_len + context_length).
        """
        for _ in range(context_length):
            _, c = X.shape

            # check if the current token is longer that max allowed length
            # if it is longer - we need to slice X input tensor 
            if c >= self._max_seq_len:
                start_index = c - self._max_seq_len
                logits, _ = self.forward(X[:, start_index:start_index + self._max_seq_len])
            else:
                logits, _ = self.forward(X)

            # logits has size [batch, seq_len, vocab_size]
            # so logits has prediction for every token in the sequence 
            # we want only the last one to generate new word 
            logits_last_step = logits[:, -1, :] # size: [batch, vocab_size]
            probabilities = softmax(logits_last_step, dim=-1)

            # sample - do not always choose token with the highest prediction
            new_token = torch.multinomial(input=probabilities, num_samples=1)

            # add new token to an input tensor 
            X = torch.cat(tensors=[X, new_token], dim=-1)
        
        return X
    