from typing import Dict

import torch
import torch.nn.functional as F
from torch.nn import Embedding

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        # TODO: model architecture
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        
        self.LSTM1 = torch.nn.LSTM(input_size=len(embeddings[0]),\
                                    hidden_size=hidden_size,\
                                    num_layers = num_layers,\
                                    dropout=dropout,\
                                    bidirectional=bidirectional,
                                    batch_first=True,
                                    )
        self.GRU = torch.nn.GRU(input_size=len(embeddings[0]),\
                                    hidden_size=hidden_size,\
                                    num_layers = num_layers,\
                                    dropout=dropout,\
                                    bidirectional=bidirectional,
                                    batch_first=True,
                                    )
        
        self.fc1 = torch.nn.Linear(2*hidden_size, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, num_class)

        self.dp = torch.nn.Dropout(p=dropout)


    @property
    def encoder_output_size(self) -> int:
        # TODO calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch, x_len) -> Dict[str, torch.Tensor]:
        # TODO implement model forward
        # embedded = self.embed(torch.LongTensor(batch))
        embedded = self.embed(batch)
        x_packed = pack_padded_sequence(embedded, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.LSTM1(x_packed)
        outputs, output_lengths = pad_packed_sequence(output, batch_first=True)

        # output = outputs.mean(dim=1)
        # output = torch.sigmoid(self.fc1(output))

        max, _ = outputs.max(dim=1)
        output = self.dp(torch.relu(self.fc1(max)))
        output = self.dp(torch.relu(self.fc2(output)))
        output = self.dp(torch.relu(self.fc3(output)))
        output = self.fc4(output)
        return output


class SeqTagger(SeqClassifier):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        max_len: int
    ) -> None:
        # super(SeqTagger, self).__init__()
        super().__init__(embeddings, hidden_size, num_layers, dropout, bidirectional, num_class)
        self.max_len = max_len
        self.num_class = num_class

        self.fc1 = torch.nn.Linear(2*hidden_size, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, num_class)


    def forward(self, batch, x_len) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embedded = self.embed(batch)
        x_packed = pack_padded_sequence(embedded, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.GRU(x_packed)
        outputs, output_lengths = pad_packed_sequence(output, batch_first=True)
        
        output = self.dp(torch.relu(self.fc1(outputs)))
        output = self.dp(torch.relu(self.fc2(output)))
        output = self.fc3(output)


        # output = torch.relu(self.fc1(max))
        # output = torch.relu(self.fc2(output))
        # output = torch.relu(self.fc3(output))
        # output = self.fc4(output)
        output = self.pad_batch(output)
        
        # print(output.shape)


        # output = torch.relu(self.fc2(output))
        # output = self.fc3(output)
        return output

    def pad_batch(self, data):
        ignore_idx = -100
        target = torch.empty(len(data), self.max_len, self.num_class).fill_(ignore_idx)
        target[:, :len(data[0]), :] = data
        target = target.transpose(1, 2)
        # print(target.shape)
        return target