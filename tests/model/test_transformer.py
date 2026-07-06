import pytest
import torch
from src.model.transformer import FeedForward


@pytest.fixture
def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.fixture
def ff_input(device):
    # Standard input shape: [batch, seq_len, d_model]
    return torch.rand([2, 5, 16], dtype=torch.float32, device=device)


@pytest.fixture
def ff_layer(device):
    return FeedForward(d_model=16).to(device)


def test_ff_shape_correctness(ff_input, ff_layer):
    output = ff_layer(ff_input)
    
    # Shape must remain unchanged [batch, seq_len, d_model]
    assert output.shape == ff_input.shape
    assert isinstance(output, torch.Tensor)


def test_ff_gradients_flow(ff_input, ff_layer):
    output = ff_layer(ff_input)
    loss = output.sum()
    loss.backward()

    # Verify that both linear layers have registered gradients
    for layer in ff_layer._model:
        if isinstance(layer, torch.nn.Linear):
            assert layer.weight.grad is not None
            if layer.bias is not None:
                assert layer.bias.grad is not None


def test_ff_gelu_activation(ff_input, ff_layer):
    # Verify the layer is producing non-linear transformations 
    # (output should not be a simple linear scaling of input)
    output = ff_layer(ff_input)
    
    # We test this by checking if the output is different from a linear mapping
    # A simple way is to ensure output is not just scalar multiplied input
    assert not torch.allclose(output, ff_input * 4.0, atol=1e-1)


def test_ff_batch_independence(ff_input, ff_layer):
    # Each token is processed independently. Modifying one token 
    # should not change the output of others.
    output_orig = ff_layer(ff_input)
    
    ff_input_modified = ff_input.clone()
    ff_input_modified[0, 0, :] = torch.rand(16)
    
    output_modified = ff_layer(ff_input_modified)
    
    # Only the modified token output should change
    assert torch.allclose(output_orig[0, 1:, :], output_modified[0, 1:, :], atol=1e-6)
    assert not torch.allclose(output_orig[0, 0, :], output_modified[0, 0, :], atol=1e-6)