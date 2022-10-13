import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
MODEL_PATH = r'hw1_2_model/'
MODEL_NAME = 'hw1_2'


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

    slot_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())
    print(tag2idx)
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    # data[TRAIN] = data[TRAIN][:100]

    # data: a dict with train and val dataset in list format
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    print('train_dataset_num:', len(datasets[TRAIN]))
    print('val_dataset_num:', len(datasets[DEV]))

    # TODO create DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets[TRAIN], args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn)
    val_dataloader = DataLoader(datasets[DEV], len(datasets[DEV]), shuffle=False, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO init model and move model to target device(cpu / gpu)

    model = SeqTagger(
        embeddings=embeddings,\
        hidden_size=args.hidden_size,\
        num_layers=args.num_layers,\
        dropout=args.dropout,\
        bidirectional=args.bidirectional,\
        num_class=datasets[TRAIN].num_classes,\
        max_len=args.max_len,
        ).to(args.device)
    # print(model)
    model.to(args.device)

    # TODO init optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0,\
    #     momentum=args.momentum, centered=False, foreach=None)
    criterion = nn.CrossEntropyLoss(ignore_index=-100) #ignore index


    y_loss = {}  # loss history
    y_loss['train'], y_loss['val'] = [], []
    y_acc = {}
    y_acc['train'], y_acc['val'] = [], []
    max_val_score = 0.0
    last_model = ''
    epoch_pbar = [i for i in range(args.num_epoch)]
    for epoch in epoch_pbar:

        tr_loss, val_loss = 0.0, 0.0
        tr_acc, val_acc = 0.0, 0.0
        epoch_loss = 0.0
        correct = 0.0

        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for i, (id, token, token_len, tag) in enumerate(train_dataloader):
            print(f'----------- Training  Epoch No. {epoch+1}   {(i+1)*100/len(train_dataloader) :.1f} %-----------best score: {max_val_score}', end='\r')
            # print('token', token)
            # print(token_len)
            # print('tag: ', tag)

            # Compute prediction and loss
            token, tag = torch.LongTensor(token), torch.LongTensor(tag)
            token, tag = token.to(args.device), tag.to(args.device)
            # print(type(token), type(tag))
            output = model(token, token_len)
            # print('token.shape: ', token.shape, '\ntag.shape: ', tag.shape, '\npred.shape: ', pred.shape)
            # print(output.shape, tag.shape)
            # print(output, tag)
        #FIXME:
            loss = criterion(output.to(args.device), tag)
            # print(output, tag)
            # print(loss)

            # _, max_idxs = torch.max(pred, dim=1)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item()
        # print(model.state_dict())
        tr_l = tr_loss/train_dataloader.__len__()
        y_loss['train'].append(tr_l)
        print(f'\n[train__{epoch + 1}]\tloss: {tr_l:.3f}')

        # TODO Evaluation loop - calculate accuracy and save model weights
        model.eval()
        val_pred, val_tag = [], []
        score = 0
        with torch.no_grad():
            for i, (id, token, token_len, tag) in enumerate(val_dataloader):
                token, tag = torch.LongTensor(token), torch.LongTensor(tag)
                # print(token.shape, tag.shape)

                token, tag = token.to(args.device), tag.to(args.device)
                pred = model(token, token_len)
                loss = criterion(pred.to(args.device), tag)

                _, max_idxs = torch.max(pred, dim=1)
                # print(loss)
                # print(tag, max_idxs)

                for i in range(len(token)):
                    p = [datasets[TRAIN].idx2tag(item) for item in max_idxs[i][:token_len[i]].tolist()]
                    t = [datasets[TRAIN].idx2tag(item) for item in tag[i][:token_len[i]].tolist()]
                    val_pred.append(p)
                    val_tag.append(t)
                    if p == t: score += 1


                val_loss += loss.item()
                # correct += sum([1 if max_idxs[i]==tag[i] else 0 for i in range(len(tag))])\
        val_l = val_loss/val_dataloader.__len__()
        print(f'[val__\t{epoch + 1}]\tloss: {val_l:.3f}')

        print('f1_scoer: ', f'\t\t{f1_score(val_tag, val_pred):.3f}')
        print('word-wise acc_scoer: ', f'\t{accuracy_score(val_tag, val_pred):.3f}')
        js = score/len(datasets[DEV])
        print('joint acc_scoer: ', f'\t{js:.3f}')
        # classification_report(val_tag, val_pred, scheme=IOB2, mode='strict')

        if js > max_val_score:
            max_val_score = js
            cp_name = f'{MODEL_NAME}_{epoch}_js_{js:.3f}.pt'
            save_checkpoint(model, optimizer, MODEL_PATH+cp_name, epoch+1, tr_l, MODEL_PATH+last_model)
            last_model = cp_name
            print(f'save cp {cp_name}')

        # epoch_pbar.set_postfix(train_loss=tr_l, val_loss=val_l, val_acc=a)

    torch.cuda.empty_cache()



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)