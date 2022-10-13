import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import csv

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import pandas as pd


from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


MODEL_PATH = r'hw1_2_model/'
MODEL_NAME = 'hw1_2'


def main(args):

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())
    print(tag2idx)
    test_data_paths = args.data_dir / f"test.json"
    test_data = json.loads(test_data_paths.read_text())

    # test_data: a dict with train and val dataset in list format
    test_datasets = SeqTaggingClsDataset(test_data, vocab, tag2idx, args.max_len)

    print('test_dataset_num:', len(test_datasets))

    # TODO create DataLoader for train / dev test_datasets
    test_dataloader = DataLoader(test_datasets, args.batch_size, shuffle=True, collate_fn=test_datasets.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO init model and move model to target device(cpu / gpu)

    model = SeqTagger(
        embeddings=embeddings,\
        hidden_size=args.hidden_size,\
        num_layers=args.num_layers,\
        dropout=args.dropout,\
        bidirectional=args.bidirectional,\
        num_class=test_datasets.num_classes,\
        max_len=args.max_len,
        ).to(args.device)
    # print(model)
    model.to(args.device)

    checkpoint = torch.load(args.ckpt_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
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

    
    # TODO Evaluation loop - calculate accuracy and save model weights
    model.eval()
    test_pred = {}
    score = 0
    with torch.no_grad():
        #TODO: return data id for output pred.csv
        for i, (id, token, token_len, tag) in enumerate(test_dataloader):
            token = torch.LongTensor(token)
            # print(token.shape, tag.shape)

            token = token.to(args.device)
            pred = model(token, token_len)

            _, max_idxs = torch.max(pred, dim=1)
            # print(loss)
            # print(tag, max_idxs)

            for i in range(len(token)):
                p = ' '.join([test_datasets.idx2tag(item) for item in max_idxs[i][:token_len[i]].tolist()])
                test_pred[id[i]] = p
    # print(test_pred)

    with open(args.pred_dir, 'w') as f:
        f.write('id,tags\n')
        for k in test_pred.keys():
            f.write(k + ',' + test_pred[k])
            f.write('\n')

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
    parser.add_argument(
        "--pred_dir",
        type=Path,
        help="Directory to save the pred file.",
        default="./pred/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # test_data loader
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)