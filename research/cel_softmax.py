import torch
import torch.nn as nn


def softmax(x_tensor: torch.tensor) -> torch.tensor:
    x_tensor = x_tensor - x_tensor.max()    # prevent from exploding during pow(e)
    x_tensor = torch.exp(x_tensor)  # every component becomes positive
    x_tensor_exp = x_tensor / x_tensor.sum()      # every component becomes in range [0, 1]
    return x_tensor_exp

def cel():
    input = torch.randn([3], device="mps")
    expected_probabilites = torch.tensor([1.0, 0.0, 0.0], device="mps")

    model = nn.Sequential(
        nn.Linear(3, 3),
        nn.Linear(3, 3)
    ).to(device="mps")

    epochs = 10
    learning_rate = 0.5

    for epoch in range(epochs):
        calculated_answers = model(input)   # produce answer due to hidden layers
        calculated_probabilites = softmax(calculated_answers) # calculate probability 

        # calculate the CEL : -ln(z)
        loss = -1 * torch.log(calculated_probabilites[0])
        print(f"Epoch {epoch}: Loss = {loss.item()}")
        loss.backward()

        # update weights and biases
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= parameter.grad * learning_rate
        
        model.zero_grad()   # reset grads

    print(calculated_answers)
    print(calculated_probabilites)
    print(expected_probabilites)


cel()
