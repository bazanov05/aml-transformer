from src.train import train
from src.tracker.db import connection
from torch.utils.data import Subset, DataLoader
from src.data.dataset import AMLDataset
from aml_tokenizer import Tokenizer
from psycopg.rows import dict_row
import torch
import json
import os
from src.visualization.plot import best_epoch_per_each_run, val_loss_per_config_run


PATH_TO_TRAINING_DATA = "aml_corpus.txt"
TOKENIZER_PATH = "src/data/tokenizer.json"
CONFIGS_PATH = "configs.json"


def open_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File with path: {path} was not found")
    except IsADirectoryError:
        raise IsADirectoryError("You are trying to open a directory")
    except PermissionError:
        raise PermissionError("You do not have permission to open that file")
    except Exception as e:
        raise e


def load_configs(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def main():
    connection.init_pool() # create pool of connects 

    # choose device based on it's avialability 
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # load training text
    training_data = open_file(PATH_TO_TRAINING_DATA)

    # train tokenizer once or load if already trained
    target_vocab_size = 2048
    context_length = 32 # for AMLDataset and for Positional Embedding
    tokenizer = Tokenizer(target_vocab_size)

    # load Tokenizer's dict from json file if it exists
    # train Tokenizer otherwise 
    if os.path.exists(TOKENIZER_PATH):
        tokenizer.load(TOKENIZER_PATH)
    else:
        tokenizer.train(training_data)
        tokenizer.save(TOKENIZER_PATH)

    # encode text to ids once
    ids = tokenizer.encode(training_data)
    ids = torch.tensor(data=ids, dtype=torch.int64, device=device)

    # create dataset and dataloaders once
    dataset = AMLDataset(ids=ids, context_length=context_length)

    subset_size = int(len(dataset) * 0.5)   # use 50% of created data to train the model 
    dataset = Subset(dataset, indices=range(subset_size))
    training_subset = Subset(dataset, indices=range(0, int(subset_size * 0.9)))
    validation_subset = Subset(dataset, indices=range(int(subset_size * 0.9), subset_size))

    training_dataloader = DataLoader(dataset=training_subset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_subset, batch_size=32, shuffle=True)

    # load hyperparameter configs
    configs = load_configs(CONFIGS_PATH)

    runs_counter = 1

    # run sweep, create separate conn for every invidual run
    for config in configs:
        with connection.pool.connection() as conn:
            conn.row_factory = dict_row     # db will return results as dict, not tuple
            print(f"RUN NR: {runs_counter}")
            print(f"\nStarting run: {config}")
            train(
                db=conn,
                training_dataloader=training_dataloader,
                validation_dataloader=validation_dataloader,
                d_model=config["d_model"],
                lr=config["lr"],
                num_of_heads=config["num_of_heads"],
                num_of_blocks=config["num_of_blocks"],
                device=device,
                target_vocab_size=target_vocab_size,
                context_length=context_length
            )
            runs_counter += 1

    connection.close_pool()

    best_epoch_per_each_run(file_for_plot="./src/visualization/best_epochs.png")
    val_loss_per_config_run(num_of_runs=6, file_for_plot="./src/visualization/val_losses.png")



if __name__ == "__main__":
    main()
