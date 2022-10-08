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
                                    # [batch_size, seq_len, num_directions * hidden_size]
        # self.rnn1 = torch.nn.RNN(input_size, hidden_layer, num_layer, bias=True, batch_first=False, dropout = 0, bidirectional = False)
        
        self.fc1 = torch.nn.Linear(2*hidden_size, 512)
        self.fc2 = torch.nn.Linear(512, num_class)


    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch, x_len) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # embedded = self.embed(torch.LongTensor(batch))
        embedded = self.embed(batch)
        x_packed = pack_padded_sequence(embedded, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.LSTM1(x_packed)
        outputs, output_lengths = pad_packed_sequence(output, batch_first=True)
        output = outputs.mean(dim=1)
        output = torch.sigmoid(self.fc1(output))
        output = self.fc2(output)
        return output


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
