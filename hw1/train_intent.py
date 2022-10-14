import json
import pickle
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tqdm import trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

import os


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
MODEL_PATH = r'hw1_1_model/'
MODEL_NAME = 'LSTM'

def save_checkpoint(net, optimizer, path, epoch, loss, last_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': loss,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        },path)
    try:
        os.remove(last_path)
    except:
        pass


def main(args):

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    print(len(datasets[TRAIN]))

    # TODO create DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets[TRAIN], args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn)
    val_dataloader = DataLoader(datasets[DEV], args.batch_size, shuffle=False, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO init model and move model to target device(cpu / gpu)

    model = SeqClassifier(
        embeddings=embeddings,\
        hidden_size=args.hidden_size,\
        num_layers=args.num_layers,\
        dropout=args.dropout,\
        bidirectional=args.bidirectional,\
        num_class=datasets[TRAIN].num_classes,\
        ).to(args.device)
    print(model)
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss() #ignore index

    max_val_acc = 0.0
    last_model = ''
    epoch_pbar = [i for i in range(args.num_epoch)]
    for epoch in epoch_pbar:

        tr_loss, val_loss = 0.0, 0.0
        correct = 0.0

        # TODO Training loop - iterate over train dataloader and update model weights
        model.train()
        for i, (id, text, text_len, intent) in enumerate(train_dataloader):
            # Compute prediction and loss
            text, intent = torch.LongTensor(text), torch.Tensor(intent).long()
            text, intent = text.to(args.device), intent.to(args.device)
            output = model(text, text_len)
            loss = criterion(output, intent)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item()
        tr_l = tr_loss/train_dataloader.__len__()
        print(f'[train__{epoch + 1}]\tloss: {tr_l:.3f}')

        # TODO Evaluation loop - calculate accuracy and save model weights23
        model.eval()
        with torch.no_grad():
            for i, (id, text, text_len, intent) in enumerate(val_dataloader):
                text, intent = torch.LongTensor(text), torch.LongTensor(intent).long()
                text, intent = text.to(args.device), intent.to(args.device)
                pred = model(text, text_len)
                loss = criterion(pred, intent)

                _, max_idxs = torch.max(pred, dim=1)
                
                val_loss += loss.item()
                correct += sum([1 if max_idxs[i]==intent[i] else 0 for i in range(len(intent))])

        val_l = val_loss/val_dataloader.__len__()
        a = correct/len(list(datasets[DEV]))
        print(f'[val__\t{epoch + 1}]\tloss: {val_l:.3f}, acc.: {a:.4f} %')

        if a > max_val_acc:
            max_val_acc = a
            cp_name = f'{MODEL_NAME}_{epoch}_acc_{a:.3f}.pt'
            save_checkpoint(model, optimizer, MODEL_PATH+cp_name, epoch+1, tr_l, MODEL_PATH+last_model)
            last_model = cp_name
            print(f'save cp {cp_name}')

    torch.cuda.empty_cache()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    #seed
    parser.add_argument("--rand_seed", type=int, help="Random seed.", default=99)

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=1024) # 512
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--momentum", type=float, default=0.7)
    

    # data loader
    parser.add_argument("--batch_size", type=int, default=16) # 64

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=200)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
