import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset


class SinDataset(Dataset):
    """
    Custom dataset class for sin data
    """
    def __init__(self, input: torch.tensor, output: torch.tensor):
        self._input = input
        self._output = output

    def __len__(self) -> int:
        return self._input.shape[0]
    
    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        return (self._input[index], self._output[index])
    

def adamw(model: nn.Sequential, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int, learning_rate: float) -> tuple[list, list, torch.tensor]:
    """
    AdamW optimizer function, which for each epoch calculates 
    the loss for training data and validation data.
    Unlike Adam it also does weight decay which prevents weigths from exploding.
    Early stopping mechanism is implemented to prevent overfitting:
    when the validation loss is not dropping for 5 or more epochs - stop training. 
    Return the tuple of two lists : training and validation losses for each epoch,
    to track when the model starts to overfit.
    """
    momentum_beta = 0.9
    adaptive_lr_beta = 0.999
    time = 1    # we do not reset it every epoch!
    epsilon = 1e-8
    lamda = 1e-4

    velocities = {param: torch.zeros_like(param) for param in model.parameters()}
    adaptive_lrs = {param: torch.zeros_like(param) for param in model.parameters()}

    val_losses = []
    train_losses = []

    # logic for early stop - when the val loss starts to grow despite train loss drop
    best_val_loss = float("inf")
    patience_counter = 0    # increase it when after backprop val loss becomes worse
    limit = 5   # if val loss is not dropping for 5 iterations - stop training 
    best_weights = {param: torch.zeros_like(param) for param in model.parameters()}

    for _ in range(epochs):
        # reset losses at the begining of every epoch 
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        for batch_X, batch_Y in train_dataloader:
            calculated_ans = model(batch_X)
            train_loss = ((calculated_ans - batch_Y) ** 2).mean()
            epoch_train_loss += train_loss.item()
            

            train_loss.backward()

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

                    # weight decay in adamw
                    # adamw does this outside grad, not like L2
                    # new_weight = new_weight - lr * lamda * old_weight
                    parameter -= parameter * lamda * learning_rate
            
            model.zero_grad()
            time += 1
        
        epoch_train_loss /= len(train_dataloader) # mean loss

        # operation for validation data do not require gradient
        # cause we do not want to update weights for validation data
        # we want just to see how good does our model works on the data it has not seen before
        with torch.no_grad():
            for batch_X, batch_Y in val_dataloader:
                calculated_ans = model(batch_X)
                val_loss = ((calculated_ans - batch_Y) ** 2).mean()
                epoch_val_loss += val_loss.item()

            epoch_val_loss /= len(val_dataloader)
        
        # if val loss still dropping - update the best loss and best weights
        # also reset patience counter
        if best_val_loss > epoch_val_loss:
            patience_counter = 0
            best_val_loss = epoch_val_loss

            for parameter in model.parameters():
                # allocate new memory for tensor and cut of the computational graph
                # cause we need just weights, not the grads
                best_weights[parameter] = parameter.clone().detach()
        
        # if the val loss becomes worse - increase patience counter 
        else:
            patience_counter += 1
        
        val_losses.append(epoch_val_loss)
        train_losses.append(epoch_train_loss)

        # if the limit is broken - stop learning 
        if patience_counter >= limit:
            break

    return train_losses, val_losses, best_weights


def benchmark():
    X = torch.linspace(start=0, end=6.28, steps=200, device="mps")    # create 200 inputs from range [0, 2 * pi]
    Y_clean = torch.sin(X).to(device="mps")

    Y_noisy = torch.randn(X.shape, device="mps")    # create noisy data 
    Y_noisy *= 0.1
    Y_noisy += Y_clean

    X = X.unsqueeze(-1) # [200] to [200, 1]
    Y_noisy = Y_noisy.unsqueeze(-1) # unsqueeze does not modify in place! 

    # create random permutaion of indecies to mix data 
    indices = torch.randperm(len(X), device="mps")

    
    model = nn.Sequential(
        nn.Linear(in_features=1, out_features=15),
        nn.ReLU(),
        nn.Linear(in_features=15, out_features=10),
        nn.ReLU(),
        nn.Linear(in_features=10, out_features=1)
    ).to(device="mps")

    # to train my model i will feed to it only 150 points, the rest of them is for validation
    train_dataset = SinDataset(input=X[indices[:150]], output=Y_noisy[indices[:150]])
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    val_dataset = SinDataset(input=X[indices[150:]], output=Y_noisy[indices[150:]])
    val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=True)

    epochs = 100
    learning_rate = 0.001


    train_losses, val_losses, best_weights = adamw(model, train_dataloader, val_dataloader, epochs, learning_rate)
    
    for epoch in range(len(val_losses)):
        print(f"epoch {epoch + 1}: train_loss {train_losses[epoch]}, val_loss {val_losses[epoch]}")

    with torch.no_grad():
        for p in model.parameters():
            p.copy_(best_weights[p])

benchmark()
