import pytest
import torch
from src.model.gpt import GPT


@pytest.fixture
def gpt_model():
    """Provides a tiny GPT model for fast unit testing."""
    return GPT(
        vocab_size=100,
        d_model=16,
        num_of_heads=2,
        num_of_blocks=2,
        max_seq_len=32
    )


def test_forward_inference_mode(gpt_model):
    batch_size, seq_len = 4, 10
    X = torch.randint(0, 100, (batch_size, seq_len))
    
    logits, loss = gpt_model(X)
    
    assert logits.shape == (batch_size, seq_len, 100)
    assert loss is None


def test_forward_training_mode(gpt_model):
    batch_size, seq_len = 4, 10
    X = torch.randint(0, 100, (batch_size, seq_len))
    Y = torch.randint(0, 100, (batch_size, seq_len))
    
    logits, loss = gpt_model(X, Y)
    
    assert logits.shape == (batch_size, seq_len, 100)
    assert loss is not None
    assert loss.item() > 0


def test_generate_standard(gpt_model):
    batch_size, seq_len = 2, 5
    context_length = 3
    X = torch.randint(0, 100, (batch_size, seq_len))
    
    generated_X = gpt_model.generate(X, context_length)
    
    assert generated_X.shape == (batch_size, seq_len + context_length)


def test_generate_exceeds_max_seq_len(gpt_model):
    batch_size, seq_len = 2, 30
    context_length = 5
    # The generation will cross the max_seq_len of 32
    X = torch.randint(0, 100, (batch_size, seq_len))
    
    generated_X = gpt_model.generate(X, context_length)
    
    # The output tensor should still grow by exactly context_length
    assert generated_X.shape == (batch_size, seq_len + context_length)
    