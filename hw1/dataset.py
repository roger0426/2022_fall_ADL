from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab
import string


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent_label for intent_label, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        d = {}
        d['id'] = [sample['id'] for sample in samples]
        d['text'] = []
        d['text_len'] = []
        for sample in samples:
            sent_list = sample['text'].translate(str.maketrans('', '', string.punctuation)).split()
            d['text_len'].append(len(sent_list))
            d['text'].append(sent_list)
        d['text'] = self.vocab.encode_batch(d['text'], to_len=self.max_len)
        # print(len(d['text']))
        try:
            d['intent_label'] = [self.label2idx(sample['intent']) for sample in samples]
        except:
            d['intent_label'] = []
        # print(d['id'], '\n', d['text'], '\n', d['intent_label'])
        #text intent_label text_id(回傳資料)
        return d['id'], d['text'], d['text_len'], d['intent_label']

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        raise NotImplementedError
