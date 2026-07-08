from src.data.dataset import AMLDataset
from src.data.prepare import prepare_dataset, train_tokenizer
from torch.utils.data import DataLoader, Subset
from aml_tokenizer import Tokenizer
import os 
from src.model.gpt import GPT
from torch.optim import AdamW
import torch


RAW_TEXT_PATH      = "src/data/raw/input.txt"
TOKENIZER_PATH     = "src/data/tokenizer.json"
MODEL_STATS_PATH   = "src/checkpoints/best_model.pt"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    target_vocab_size = 512  # target vocab size for tokenizer vocab
    context_length = 8     # for AMLDataset

    tokenizer = Tokenizer(target_vocab_size)    # create tokenizer 

    # open txt file with training data and load it to str
    with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
        training_data = f.read()

    # if merge rules and vocab exist - just them from json file
    # there is no need to train tokenizer one more time
    if os.path.exists(TOKENIZER_PATH):
        tokenizer.load(TOKENIZER_PATH)
    # otherwise - train tokenizer on input data
    else:
        train_tokenizer(tokenizer, training_data, path=TOKENIZER_PATH)

    # create 1D tensor of ids 
    ids = prepare_dataset(tokenizer, text=training_data, device=device)
    dataset = AMLDataset(ids, context_length)   # create dataset 

    # use 90% of training data to train
    training_dataset = Subset(dataset=dataset, indices=range(0, int(len(dataset) * 0.9)))
    # 10% use to check how model performs on unseen data
    validation_dataset = Subset(dataset=dataset, indices=range(int(len(dataset) * 0.9), len(dataset)))

    training_dataloader = DataLoader(training_dataset, batch_size=128, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    d_model = 64
    num_of_heads = 4
    num_of_blocks = 3
    max_seq_len = context_length  

    model = GPT(
        vocab_size=target_vocab_size,
        d_model=d_model,
        num_of_heads=num_of_heads,
        num_of_blocks=num_of_blocks,
        max_seq_len=max_seq_len
    ).to(device=device)
    optimizer = AdamW(params=model.parameters(), lr=3e-4)
    epochs = 100

    patience_rate = 0
    max_patience_rate = 5
    best_loss = float("inf")

    for epoch in range(epochs):
        total_train_loss = 0

        # apply train mode - some neurons will be killed
        model.train()
        for X, Y in training_dataloader:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()   # resets gradients before calculations

            _, loss = model(X, Y)
            loss.backward()     # calculate gradients
            optimizer.step()    # update wieghts 
            total_train_loss += loss.item()

        mean_train_loss = total_train_loss / len(training_dataloader)
        
        total_val_loss = 0

        # apply val mode - all neurons will be alive
        model.eval()
        for X, Y in validation_dataloader:
            X, Y = X.to(device), Y.to(device)

            # we do not want to compute gradients for validation data 
            with torch.no_grad():
                _, loss = model(X, Y)
                total_val_loss += loss.item()
        
        # calculate mean loss across all batches
        mean_val_loss = total_val_loss / len(validation_dataloader)
        print(f"Epoch {epoch+1},  train_loss: {mean_train_loss}, val_loss: {mean_val_loss}")

        # improvement found — save checkpoint and reset patience
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            patience_rate = 0   # reset patience rate 
            torch.save(obj=model.state_dict(), f=MODEL_STATS_PATH) # save best weights
        else:
            # if model perfomance on validation data becomes worse
            # increase patience rate
            patience_rate += 1

            # if we hit the limit - we started overfitting
            # stop the training 
            if patience_rate >= max_patience_rate:
                model.load_state_dict(torch.load(MODEL_STATS_PATH, weights_only=True))
                break
    
    first_token = 0
    idx = torch.tensor([[first_token]], device=device, dtype=torch.int64)

    context_length = 25
    result = model.generate(idx, context_length)
    result = result.squeeze(dim=0)
    result = result.tolist()
    decoded_text = tokenizer.decode(result)
    print(decoded_text)


if __name__ == "__main__":
    main()
    