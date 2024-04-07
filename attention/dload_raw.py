from datasets import load_dataset
from config import get_config
import torch
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path
from dataset import BilingualDataset
from torch.utils.data import random_split
import pandas as pd


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

if __name__ == '__main__':
    config=get_config()
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size], generator=torch.Generator().manual_seed(42))

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    #pd.Dataframe(train_ds.numpy().to_csv("data/train_ds.csv",index=False))
    #pd.Dataframe(val_ds.numpy().to_csv("data/val_ds.csv",index=False))    

    torch.save(train_ds, "data/train_ds.pt")
    torch.save(val_ds, "data/val_ds.pt")
    torch.save(tokenizer_src, "data/tokenizer_src.pt")
    torch.save(tokenizer_tgt, "data/tokenizer_tgt.pt")
    
    #train_ds.to_csv("data/train_ds.csv", index=False)
    #val_ds.to_csv("data/val_ds.csv", index=False)

