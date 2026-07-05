import pytest
import torch
from src.model.attention import SelfAttention, MultiHeadAttention


@pytest.fixture
def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.fixture
def input_tensor(device):
    return torch.rand([2, 5, 7], dtype=torch.float32, device=device)


@pytest.fixture
def head(device):
    # Tests different d_k and d_v shapes intrinsically
    return SelfAttention(d_model=7, d_k=4, d_v=5).to(device)


def test_shape_correctness(input_tensor, head):
    output = head(input_tensor)
    b, seq_len, d_v = output.shape

    assert isinstance(output, torch.Tensor)
    assert b == 2
    assert seq_len == 5
    assert d_v == 5


def test_causal_mask_prevents_future_attention(head, input_tensor):
    out_original = head(input_tensor)

    # Modify only the last token in the sequence
    input_modified = input_tensor.clone()
    input_modified[0, -1, :] = torch.rand(7)

    out_modified = head(input_modified)

    # Outputs for previous tokens must remain exactly identical
    assert torch.allclose(out_original[0, :-1, :], out_modified[0, :-1, :], atol=1e-6)
    # Output for the modified token must change
    assert not torch.allclose(out_original[0, -1, :], out_modified[0, -1, :], atol=1e-6)


def test_device_consistency():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
        
    for dev in devices:
        test_head = SelfAttention(d_model=7, d_k=4, d_v=5).to(dev)
        test_input = torch.rand([2, 5, 7], device=dev)
        
        output = test_head(test_input)
        assert output.device.type == dev


def test_gradients_flow(head, input_tensor):
    output = head(input_tensor)
    loss = output.sum()
    loss.backward()

    # Verify gradients are registered and computed for Q, K, V matrices
    assert head._W_Q.weight.grad is not None
    assert head._W_K.weight.grad is not None
    assert head._W_V.weight.grad is not None


def test_batch_independence(head, input_tensor):
    # Compute output for the entire batch
    batch_output = head(input_tensor)

    # Compute output for just the first sequence in isolation
    single_input = input_tensor[0:1, :, :] 
    single_output = head(single_input)

    # The result must be completely independent of other batch items
    assert torch.allclose(batch_output[0:1], single_output, atol=1e-6)


@pytest.fixture
def mha_input(device):
    # d_model must be divisible by num_heads, so we use 16 for d_model
    return torch.rand([2, 5, 16], dtype=torch.float32, device=device)


@pytest.fixture
def mha_head(device):
    # 16 / 4 = 4 (d_k and d_v will be 4)
    return MultiHeadAttention(d_model=16, num_of_heads=4).to(device)


def test_mha_shape_correctness(mha_input, mha_head):
    # Base case: Output shape must exactly match input shape [batch, seq_len, d_model]
    output = mha_head(mha_input)
    b, seq_len, d_model = output.shape

    assert isinstance(output, torch.Tensor)
    assert b == 2
    assert seq_len == 5
    assert d_model == 16  # Verifies the concatenation and _W_O projection worked

def test_mha_indivisible_d_model_raises_error():
    # Edge case: d_model not cleanly divisible by num_of_heads
    with pytest.raises(ValueError, match="d_model must be divisible by num_of_heads"):
        MultiHeadAttention(d_model=10, num_of_heads=3)


def test_mha_gradients_flow(mha_input, mha_head):
    output = mha_head(mha_input)
    loss = output.sum()
    loss.backward()

    # Verify gradients flow back through the final linear projection layer
    assert mha_head._W_O.weight.grad is not None
    
    # Verify gradients flow all the way back into the individual SelfAttention heads
    for head in mha_head._heads:
        assert head._W_Q.weight.grad is not None
        assert head._W_K.weight.grad is not None
        assert head._W_V.weight.grad is not None


def test_mha_batch_independence(mha_input, mha_head):
    # Compute output for the entire batch
    batch_output = mha_head(mha_input)

    # Compute output for just the first sequence in isolation
    single_input = mha_input[0:1, :, :] 
    single_output = mha_head(single_input)

    # Output of sequence 0 should not be affected by sequence 1 existing in the batch
    assert torch.allclose(batch_output[0:1], single_output, atol=1e-6)


def test_mha_edge_case_single_token(mha_head, device):
    # Edge case: Passing a sequence of exactly 1 token
    single_token_input = torch.rand([3, 1, 16], dtype=torch.float32, device=device)
    
    output = mha_head(single_token_input)
    
    # Should process without indexing errors and maintain shapes
    assert output.shape == (3, 1, 16)
