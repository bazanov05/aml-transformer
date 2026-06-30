from src.data.dataset import AMLDataset
from src.data.prepare import prepare_dataset, train_tokenizer
from torch.utils.data import DataLoader
from aml_tokenizer import Tokenizer
import os 


RAW_TEXT_PATH      = "data/raw/input.txt"
TOKENIZER_PATH     = "data/tokenizer.json"


def main():
    target_vocab_size = 10000  # target vocab size for tokenizer vocab
    context_length = 25     # for AMLDataset

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
        train_tokenizer(tokenizer, training_data)

    # create 1D tensor of ids 
    ids = prepare_dataset(tokenizer, text=training_data)
    dataset = AMLDataset(ids, context_length)   # create dataset 

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


if __name__ == "__main__":
    main()
    