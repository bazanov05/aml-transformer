import torch
import torch.nn as nn


def training_with_modules():
    Input = torch.randn([2, 3, 5], device="mps")
    Expected_answers = torch.rand([2, 3, 1], device="mps")

    model = nn.Sequential(
        nn.Linear(in_features=5, out_features=9),   # 1st linear layer - 5 dims of token in, 9 features out
        nn.ReLU(),
        nn.Linear(in_features=9, out_features=1),   # 2ns linear layer - 9 features from prev layer in, 1 out
        nn.ReLU()    
        ).to(device="mps")

    learning_rate = 0.01
    loss_value = float("inf")
    iterations_made = 0

    while loss_value > 0.01 and iterations_made < 2000:
        calculated_answers = model(Input)

        # calculate the loss(MSE)
        loss = ((calculated_answers - Expected_answers) ** 2).sum() / (Expected_answers.shape[0] * Expected_answers.shape[1])

        loss.backward() # calculate the gradients

        # update the weigths for all the neurons in all layers
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= parameter.grad * learning_rate
        
        model.zero_grad()   # reset all the gradients
        iterations_made += 1

        loss_value = loss.item()

    print(f"Final Loss: {loss}\n")
    print(f"Iterations made: {iterations_made}")
    print(f"Expected answers: {Expected_answers}\n")
    print(f"Calculated answers: {model(Input)}\n")

training_with_modules()
