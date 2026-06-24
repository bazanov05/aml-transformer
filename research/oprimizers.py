import torch 
import torch.nn as nn
from research.dataset import CustomDataset
from torch.utils.data import DataLoader
import copy


def sgd_with_mini_batches(model: nn.Sequential, dataloader: DataLoader, epochs: int, learning_rate: float) -> float:
    for _ in range(epochs):
        for batch_X, batch_Y in dataloader:
            calculated_ans = model(batch_X)
            loss = ((calculated_ans - batch_Y) ** 2).mean()

            loss.backward()

            with torch.no_grad():
                for layer in model.parameters():
                    layer -= layer.grad * learning_rate
            
            model.zero_grad()
        
    return loss.item()


def sgd_with_momentum(model: nn.Sequential, dataloader: DataLoader, epochs: int, learning_rate: float) -> float:
    """
    SGD with momenrum alows us to make more bigger steps, unlike naive SGD.
    Velocity allows us to skip local minima, and even if we overshoot
    velovity will change the direction due to gradient and losing 10% from prev velocity.
    """
    beta = 0.9  # momentum coefficient - almost always set to 0.9, which means the loss of 10% in kinematic energy 

    # create a dict of velocities since every weight in every layer has its own velocity
    # start with velocity = 0
    velocities = {param: torch.zeros_like(param) for param in model.parameters()}

    for _ in range(epochs):
        for batch_X, batch_Y in dataloader:
            calculated_ans = model(batch_X)
            loss = ((calculated_ans - batch_Y) ** 2).mean()

            loss.backward()

            with torch.no_grad():
                for layer in model.parameters():
                    v_old = velocities[layer]
                    v_new = beta * v_old + layer.grad
                    layer -= v_new * learning_rate
                    velocities[layer] = v_new   # update velocities 
            
            model.zero_grad()
        
    return loss.item()


def adagrad(model: nn.Sequential, dataloader: DataLoader, epochs: int, learning_rate: float) -> float:
    """
    The idea is to make lr unique for every weight,
    as some weights are more sensitive and have bigger impact on Loss.
    The drawback is the sum of grad^2 always growning, so in the end
    model stops learning, cause we divide by very big num and adaptive_lr becomes 0.
    """
    squared_grads = {layer: torch.zeros_like(layer) for layer in model.parameters()}
    epsilon = 1e-9

    for _ in range(epochs):
        for batch_X, batch_Y in dataloader:
            calculated_ans = model(batch_X)
            loss = ((calculated_ans - batch_Y) ** 2).mean()

            loss.backward()

            with torch.no_grad():
                for layer in model.parameters():
                    squared_grads[layer] += layer.grad ** 2
                    # add small epsilon not to divide by 0 when the grad = 0
                    adaptive_lr = learning_rate / (squared_grads[layer] ** 0.5 + epsilon)
                    layer -= adaptive_lr * layer.grad
            
            model.zero_grad()
    
    return loss.item()


def compare_optimizers():
    input_data = torch.randn([6, 5, 5], device="mps")
    output_data = torch.rand([6, 5, 1], device="mps")

    base_model = nn.Sequential(
        nn.Linear(in_features=5, out_features=5),
        nn.ReLU(),
        nn.Linear(in_features=5, out_features=1),
    ).to(device="mps")

    model_sgd = copy.deepcopy(base_model)
    model_momentum = copy.deepcopy(base_model)
    model_adagrad = copy.deepcopy(base_model)

    epochs = 100
    learning_rate = 0.05

    dataset = CustomDataset(input_data, output_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    loss_from_sgd = sgd_with_mini_batches(model_sgd, dataloader, epochs, learning_rate)
    loss_from_sgd_with_momentum = sgd_with_momentum(model_momentum, dataloader, epochs, learning_rate)
    loss_from_adagrad = adagrad(model_adagrad, dataloader, epochs, learning_rate=1.5)

    print(f"loss without momentum: {loss_from_sgd}\n")
    print(f"loss with momentum: {loss_from_sgd_with_momentum}")
    print(f"loss from adagrad: {loss_from_adagrad}\n")



compare_optimizers()