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


def adam(model: nn.Sequential, dataloader: DataLoader, epochs: int, learning_rate: float) -> float:
    """
    My implementation of Adam optimizer(Adaptive Moment Estimation).
    It solves both problems of AdaGrad and SGD with Momentum:
    takes velocity from momentum and adaptive lr from adagrad,
    but takes tiny part of squarred grad so it will not explode.
    What is more - at very first steps velocity and adaptive lr are very small,
    that is why Adam biases them by dividing by (1 - beta ** t), so at very steps
    beta ** t is bigger, but later it is close to 0.
    """
    momentum_beta = 0.9
    adaptive_lr_beta = 0.999
    time = 1    # we do not reset it every epoch!
    epsilon = 1e-8

    velocities = {param: torch.zeros_like(param) for param in model.parameters()}
    adaptive_lrs = {param: torch.zeros_like(param) for param in model.parameters()}

    for _ in range(epochs):
        for batch_X, batch_Y in dataloader:
            calculated_ans = model(batch_X)
            loss = ((calculated_ans - batch_Y) ** 2).mean()

            loss.backward()

            with torch.no_grad():
                for parameter in model.parameters():
                    # v_new = beta1 * v_old + (1 - beta1) * grad
                    new_velocity = momentum_beta * velocities[parameter] + (1 - momentum_beta) * parameter.grad
                    # new_lr = beta2 * old_lr + (1 - beta2) * grad^2
                    new_adaptive_lr = adaptive_lr_beta * adaptive_lrs[parameter] + (1 - adaptive_lr_beta) * (parameter.grad ** 2)

                    velocities[parameter] = new_velocity
                    adaptive_lrs[parameter] = new_adaptive_lr

                    # bias_v = new_v / (1 - beta1 ^ t)
                    bias_velocity = new_velocity / (1 - momentum_beta ** time)
                    # bias_lr = new_lr / (1 - beta2 ^ t)
                    bias_adaptive_lr = new_adaptive_lr / (1 - adaptive_lr_beta ** time)

                    # new_weight = old_weight - lr * bias_v / (sqrt(bias_lr) + epsilon)
                    parameter -= learning_rate * bias_velocity / (bias_adaptive_lr ** 0.5 + epsilon)
            
            model.zero_grad()
            time += 1
    
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
    N = 2000

    X = torch.randn(N, 20, device="mps")
    Y = (
        torch.sin(X[:, 0:1])
        + torch.sin(3 * X[:, 1:2])
        + X[:, 2:3]**3
        + 0.1 * torch.randn_like(X[:, :1])
    ).to(device="mps")

    base_model = nn.Sequential(
    nn.Linear(20, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    ).to(device="mps")

    model_sgd = copy.deepcopy(base_model)
    model_momentum = copy.deepcopy(base_model)
    model_adagrad = copy.deepcopy(base_model)
    model_adam = copy.deepcopy(base_model)

    epochs = 20
    learning_rate = 0.001

    dataset = CustomDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    loss_from_sgd = sgd_with_mini_batches(model_sgd, dataloader, epochs, learning_rate)
    loss_from_sgd_with_momentum = sgd_with_momentum(model_momentum, dataloader, epochs, learning_rate)
    loss_from_adagrad = adagrad(model_adagrad, dataloader, epochs, learning_rate)
    loss_from_adam = adam(model_adam, dataloader, epochs, learning_rate)

    print(f"loss without momentum: {loss_from_sgd}")
    print(f"loss with momentum: {loss_from_sgd_with_momentum}")
    print(f"loss from adagrad: {loss_from_adagrad}")
    print(f"loss from adam: {loss_from_adam}\n")



compare_optimizers()