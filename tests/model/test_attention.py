import pytest
import torch
from src.model.attention import SelfAttention


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
    