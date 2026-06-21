import torch


def one_neuron_multiple_tokens():
    Input = torch.randn([2, 3, 5], device="mps")    # tensor with 2 sentences, 3 tokens 5 dims each
    Weights = torch.randn([5, 1], device="mps", requires_grad=True) # 5 in_features, 1 out_feature
    Bias = torch.zeros([1], device="mps", requires_grad=True) # bias

    Expected_answers = torch.rand([2, 3, 1], device="mps")   # each token at the end recieves one feature 

    loss_value = float("inf")
    learning_rate = 0.01
    iterations_made = 0

    # I will be satified if the loss is close to 0.05
    while(loss_value > 0.05 and iterations_made < 1000):
        linear_output = Input @ Weights + Bias  # z = XW + b

        # use ReLU as an activation function
        activation_output = torch.relu(linear_output)
        
        # use MSE as loss function - (y - true) ^ 2
        # sum loss for every token and find the average loss across all of them
        loss = ((activation_output - Expected_answers) ** 2).sum() / (Expected_answers.shape[1] * Expected_answers.shape[0])
        
        loss.backward()     # compute the gradients for Weights and Bias 

        # updating the weights without grad so the Computational Graph is not messed
        with torch.no_grad():
            # new_weight = old_weight - learn.rate * grad
            Weights -= learning_rate * Weights.grad
            Bias -= learning_rate * Bias.grad

        iterations_made += 1
        loss_value = loss.item()

        # reset the grads so they will not sum up in the next iterations - prevent from exploding
        Bias.grad.zero_()
        Weights.grad.zero_()
    
    print(f"Final Loss: {loss}\n")
    print(f"Final Weights: {Weights}\n")
    print(f"Final Bias: {Bias}\n")
    print(f"Iterations made: {iterations_made}")
    print(f"Expected answers: {Expected_answers}\n")
    print(f"Calculated answers: {torch.relu(Input @ Weights + Bias)}\n")


one_neuron_multiple_tokens()
