import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 

class CustomDataset(Dataset):
    def __init__(self, input_dataset, output_dataset):
        # store the data in memory so DataLoader can have an access to it 
        super().__init__()
        self._input = input_dataset
        self._output = output_dataset
    
    def __len__(self) -> int:
        # tell DataLoader the size of my batch so it can plan the batching of samples
        return self._input.shape[0]
    
    def __getitem__(self, index) -> tuple[torch.tensor]:
        # provide the item based on index - DataLoader will need this in a loop
        return (self._input[index], self._output[index])

def training_dataset():
    input_batch = torch.randn([6, 3, 5], device="mps")
    output_batch = torch.rand([6, 3, 1], device="mps")

    model = nn.Sequential(
        nn.Linear(in_features=5, out_features=2),
        nn.ReLU(),
        nn.Linear(in_features=2, out_features=1),
        nn.ReLU()
    ).to(device="mps")
    dataset = CustomDataset(input_batch, output_batch)

    # process 2 sentences in a parallel and shuffle all data before every Epoch
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    
    number_of_epochs = 10
    learning_rate = 0.01

    for _ in range(number_of_epochs):
        for batch_X, batch_Y in dataloader:
            calculated_answers = model(batch_X)
            loss = ((calculated_answers - batch_Y) ** 2).mean()

            loss.backward()

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= parameter.grad * learning_rate
            
            model.zero_grad()
    
    print(f"Loss value: {loss.item()}\n")

training_dataset()