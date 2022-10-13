import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import pandas as pd

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab



def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO crecate DataLoader for test dataset
    test_dataloader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    # net.eval()

    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    
    # TODO predict dataset
    pred = {}
    model.eval()
    with torch.no_grad():
        for i, (ids, text, text_len, _) in enumerate(test_dataloader):
            # print(text)
            text, text_len = torch.LongTensor(text), torch.Tensor(text_len)
            text = text.to(args.device)
            output = model(text, text_len)
            # print(pred)
            _, max_idxs = torch.max(output, dim=1)

            for id in ids:
                t = max_idxs[ids.index(id)]
                pred[id] = dataset.idx2label(t.item())
    
    # print(pred)

    # TODO write prediction to file (args.pred_file)
    df = pd.DataFrame.from_dict(pred, orient='index')
    print(df)
    df.reset_index(inplace=True)
    df.rename(columns={0: 'intent', 'index': 'id'}, inplace=True)
    print(df)
    df.to_csv(args.pred_file, index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
