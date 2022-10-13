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

import random
import os

import optuna


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
MODEL_PATH = r'hw1_1_model/'
MODEL_NAME = 'LSTM'

trail_num = 0

def save_checkpoint(net, optimizer, path, epoch, loss, last_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': loss,
        },path)
    try:
        os.remove(last_path)
    except:
        pass


# Define an objective function to be minimized.
def objective(trial):
    
    #TODO:
    hyperparam = {
    #     'img_size': 512,
    #     'batch_size': 16,#32
    #     'lr': 5e-4,
    #     'momentum': 0.7,
    #     'weight_decay': 0.01,
    #     'dropout': 0.7,
    #     'epoch': 50,

        'data_dir': r'./data/intent/',
        'cache_dir': r'./cache/intent/',
        'ckpt_dir': r'./ckpt/intent/',

        'max_len': 128,
        'dropout': 0.65,
        'bidirectional': True,
        'lr': 0.0015,
        'momentum': 0.5,
        'device': 'cuda',
        'num_epoch': 150,
        'early_stop': 30,

        'hidden_size': 1024,
        'num_layers': 1,
        'batch_size': 160,


        # 'max_len': 128,
        # 'dropout': trial.suggest_float('dropout', 0.55, 0.7),
        # 'bidirectional': True,
        # 'lr': trial.suggest_float('lr', 0.0005, 0.005),
        # 'momentum': trial.suggest_float('momentum', 0.6, 0.8),
        # 'device': 'cuda',
        # 'num_epoch': 150,
        # 'early_stop': 30,

        # 'hidden_size': trial.suggest_int('hidden_size', 256, 1024, log=True),
        # 'num_layers': 1,
        # 'batch_size': trial.suggest_int('batch_size', 128, 256, log=True),
    }


    # Invoke suggest methods of a Trial object to generate hyperparameters.
    # regressor_name = trial.suggest_categorical('regressor', ['SVR', 'RandomForest'])
    # svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)


    error = one_epoch(hyperparam)




    
    return error  # An objective value linked with the Trial object.



def one_epoch(hyp):
    global trail_num

    with open(hyp['cache_dir']+"vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = Path(hyp['cache_dir']+"intent2idx.json")
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: Path(hyp['data_dir']) / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, hyp['max_len'])
        for split, split_data in data.items()
    }

    print(len(datasets[TRAIN]))

    # TODO: create DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets[TRAIN], hyp['batch_size'], shuffle=True, collate_fn=datasets[TRAIN].collate_fn)
    val_dataloader = DataLoader(datasets[DEV], hyp['batch_size'], shuffle=False, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(Path(hyp['cache_dir']) / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)

    model = SeqClassifier(
        embeddings=embeddings,\
        hidden_size=hyp['hidden_size'],\
        num_layers=hyp['num_layers'],\
        dropout=hyp['dropout'],\
        bidirectional=hyp['bidirectional'],\
        num_class=datasets[TRAIN].num_classes,\
        ).to(hyp['device'])
    print(model)
    model.to(hyp['device'])

    # TODO: init optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr, momentum=hyp['momentum)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyp['lr'])
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=hyp['lr, alpha=0.99, eps=1e-08, weight_decay=0,\
    #     momentum=hyp['momentum, centered=False, foreach=None)
    criterion = nn.CrossEntropyLoss() #ignore index


    y_loss = {}  # loss history
    y_loss['train'], y_loss['val'] = [], []
    y_acc = {}
    y_acc['train'], y_acc['val'] = [], []
    max_val_acc = 0.0
    last_model = ''
    last_better_epoch = 0
    epoch_pbar = [i for i in range(hyp['num_epoch'])]
    for epoch in epoch_pbar:

        tr_loss, val_loss = 0.0, 0.0
        tr_acc, val_acc = 0.0, 0.0
        epoch_loss = 0.0
        correct = 0.0

        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for i, (id, text, text_len, intent) in enumerate(train_dataloader):
            # Compute prediction and loss
            text, intent = torch.LongTensor(text), torch.Tensor(intent).long()
            text, intent = text.to(hyp['device']), intent.to(hyp['device'])
            output = model(text, text_len)
            # print('text.shape: ', text.shape, '\nintent.shape: ', intent.shape, '\npred.shape: ', pred.shape)
            loss = criterion(output, intent)
            # print(pred, intent)

            # _, max_idxs = torch.max(pred, dim=1)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item()
        # print(model.state_dict())

        tr_l = tr_loss/train_dataloader.__len__()
        y_loss['train'].append(tr_l)
        print(f'[train__{epoch + 1}]\tloss: {tr_l:.3f}')

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            for i, (id, text, text_len, intent) in enumerate(val_dataloader):
                text, intent = torch.LongTensor(text), torch.Tensor(intent).long()
                # print(text.shape, intent.shape)

                text, intent = text.to(hyp['device']), intent.to(hyp['device'])
                pred = model(text, text_len)
                loss = criterion(pred, intent)

                _, max_idxs = torch.max(pred, dim=1)
                
                val_loss += loss.item()
                correct += sum([1 if max_idxs[i]==intent[i] else 0 for i in range(len(intent))])

        val_l = val_loss/val_dataloader.__len__()
        a = correct/len(list(datasets[DEV]))
        y_loss['val'].append(val_l)
        y_acc['val'].append(a)
        print(f'[val__\t{epoch + 1}]\tloss: {val_l:.3f}, acc.: {a:.4f} %')

        if a > max_val_acc:
            max_val_acc = a
            cp_name = "{mn}_t{tn}_e{e}_acc_{a:.3f}.pt".format(mn=MODEL_NAME, tn=trail_num, e=epoch, a=a)
            save_checkpoint(model, optimizer, MODEL_PATH+cp_name, epoch+1, tr_l, MODEL_PATH+last_model)
            last_model = cp_name
            print(f'save cp {cp_name}')
            last_better_epoch = 0

        if last_better_epoch > hyp['early_stop']:
            break
        last_better_epoch += 1

        if epoch > 20 and max_val_acc < 0.02:
            break

        # epoch_pbar.set_postfix(train_loss=tr_l, val_loss=val_l, val_acc=a)

    trail_num += 1
    torch.cuda.empty_cache()
    return -max_val_acc

    # TODO: Inference on test set


# def parse_argv) -> Namespace:
#     parser = ArgumentParser()
#     parser.add_argument(
#         "--data_dir",
#         type=Path,
#         help="Directory to the dataset.",
#         default="./data/intent/",
#     )
#     parser.add_argument(
#         "--cache_dir",
#         type=Path,
#         help="Directory to the preprocessed caches.",
#         default="./cache/intent/",
#     )
#     parser.add_argument(
#         "--ckpt_dir",
#         type=Path,
#         help="Directory to save the model file.",
#         default="./ckpt/intent/",
#     )

#     #seed
#     parser.add_argument("--rand_seed", type=int, help="Random seed.", default=99)

#     # data
#     parser.add_argument("--max_len", type=int, default=128)

#     # model
#     parser.add_argument("--hidden_size", type=int, default=1024) # 512
#     parser.add_argument("--num_layers", type=int, default=2)
#     parser.add_argument("--dropout", type=float, default=0.7)
#     parser.add_argument("--bidirectional", type=bool, default=True)

#     # optimizer
#     parser.add_argument("--lr", type=float, default=5e-1)
#     parser.add_argument("--momentum", type=float, default=0.9)
    

#     # data loader
#     parser.add_argument("--batch_size", type=int, default=256) # 64

#     # training
#     parser.add_argument(
#         "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
#     )
#     parser.add_argument("--num_epoch", type=int, default=120)

#     args = parser.parse_args()
#     return args

if __name__ == "__main__":
    # args = parse_args
    # args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    # main(args)

    study = optuna.create_study()  # Create a new study.
    study.optimize(objective, n_trials=1)  # Invoke optimization of the objective function.


    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
