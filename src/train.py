from torch.utils.data import DataLoader
from src.model.gpt import GPT
from torch.optim import AdamW
import torch
from src.tracker.db.runs import create_run
from src.tracker.db.epochs import insert_epoch


def train(
        db, 
        training_dataloader: DataLoader, 
        validation_dataloader: DataLoader, 
        d_model: int, 
        lr: float,
        num_of_heads: int, 
        num_of_blocks: int,
        device: str,
        target_vocab_size: int,
        context_length: int):
    
    model = GPT(
        vocab_size=target_vocab_size,
        d_model=d_model,
        num_of_heads=num_of_heads,
        num_of_blocks=num_of_blocks,
        max_seq_len=context_length
    ).to(device=device)

    optimizer = AdamW(params=model.parameters(), lr=lr)
    epochs = 50
    patience_rate = 0
    max_patience_rate = 5
    best_loss = float("inf")

    run_id = create_run(db, d_model, lr, num_of_blocks, num_of_heads)

    for epoch in range(1, epochs + 1):
        total_train_loss = 0

        model.train()
        for X, Y in training_dataloader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            _, loss = model(X, Y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        mean_train_loss = total_train_loss / len(training_dataloader)
        total_val_loss = 0

        model.eval()
        for X, Y in validation_dataloader:
            X, Y = X.to(device), Y.to(device)
            with torch.no_grad():
                _, loss = model(X, Y)
                total_val_loss += loss.item()

        mean_val_loss = total_val_loss / len(validation_dataloader)
        print(f"Epoch {epoch}, train_loss: {mean_train_loss}, val_loss: {mean_val_loss}")

        insert_epoch(db, run_id, epoch, mean_train_loss, mean_val_loss)

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            patience_rate = 0
        else:
            patience_rate += 1
            if patience_rate >= max_patience_rate:
                break

    return run_id, best_loss
