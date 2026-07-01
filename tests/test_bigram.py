import pytest
import torch
from src.model.bigram import BigramLanguageModel


@pytest.fixture
def vocab_size():
    # standard small vocab for testing
    return 128


@pytest.fixture
def model(vocab_size):
    return BigramLanguageModel(vocab_size)


def test_forward_without_targets(model, vocab_size):
    # inference mode - batch size 4, context length 10
    x = torch.randint(0, vocab_size, (4, 10))
    
    logits, loss = model(x)
    
    # loss must be none when targets are missing
    assert loss is None
    
    # logits shape must be (b, t, c)
    assert logits.shape == (4, 10, vocab_size)


def test_forward_with_targets(model, vocab_size):
    # training mode - batch size 4, context length 10
    x = torch.randint(0, vocab_size, (4, 10))
    y = torch.randint(0, vocab_size, (4, 10))
    
    logits, loss = model(x, y)
    
    # loss must be a valid scalar tensor
    assert loss is not None
    assert isinstance(loss.item(), float)
    assert loss.item() > 0.0
    
    # logits must be flattened to (b * t, c) for cross entropy
    assert logits.shape == (40, vocab_size)


def test_forward_edge_case_minimal_dimensions(model, vocab_size):
    # edge case - smallest possible inputs (batch 1, context 1)
    x = torch.randint(0, vocab_size, (1, 1))
    y = torch.randint(0, vocab_size, (1, 1))
    
    logits, loss = model(x, y)
    
    # ensure no dimension collapsing or reshaping errors happen
    assert loss is not None
    assert logits.shape == (1, vocab_size)


def test_forward_edge_case_boundary_tokens(model, vocab_size):
    # edge case - using the absolute highest and lowest valid token ids
    max_id = vocab_size - 1
    x = torch.tensor([[0, max_id]])
    
    # should process without index out of bounds error from embedding layer
    logits, loss = model(x)
    
    assert logits.shape == (1, 2, vocab_size)


def test_generate_base_case(model, vocab_size):
    # standard generation - batch size 2, starting context 5
    idx = torch.randint(0, vocab_size, (2, 5))
    max_new_tokens = 4
    
    out = model.generate(idx, max_new_tokens)
    
    # shape must grow by exactly max_new_tokens (5 + 4 = 9)
    assert out.shape == (2, 9)
    
    # all generated tokens must be within valid vocabulary range
    assert torch.all(out >= 0)
    assert torch.all(out < vocab_size)


def test_generate_edge_case_single_token(model, vocab_size):
    # edge case - starting with the absolute minimum context of 1 token
    idx = torch.randint(0, vocab_size, (4, 1))
    
    out = model.generate(idx, max_new_tokens=3)
    
    # model must handle 1d slicing correctly and return 4 tokens total
    assert out.shape == (4, 4)


def test_generate_edge_case_zero_tokens(model, vocab_size):
    # edge case - asking for 0 new tokens to be generated
    idx = torch.randint(0, vocab_size, (3, 7))
    
    out = model.generate(idx, max_new_tokens=0)
    
    # should bypass the loop entirely and return the identical tensor
    assert out.shape == (3, 7)
    assert torch.equal(out, idx)
