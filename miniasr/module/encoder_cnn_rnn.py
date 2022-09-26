from audioop import bias
import torch
from torch import nn

class CNNRNNEncoder(nn.Module):

    def __init__(self, in_dim, hid_dim, n_layers, module='LSTM',
                 dropout=0.5, bidirectional=True):
        
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # CNN models
        self.conv1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=32,
            kernel_size=(11,41),
            strides=(2,2),
            bias=False
        )
        self.norm1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(11,21),
            strides=(1,2),
            bias=False
        )
        self.norm2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU()

        self.linear1 = nn.Linear(hid_dim,2*hid_dim)
        self.relu3 = nn.ReLU()
        self.do = nn.Dropout(p = dropout)


    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):

        if not self.training:
            #TODO
            pass

        x = feat.view((-1, self.in_dim, 1))
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))

        x = x.view((-1, list(x.shape)[-2]*list(x.shape[-1])))

        # RNN model
        self.rnn = nn.GRU(
            num_layers = self.n_layers,
            input_size = list(x.shape)[-1],
            hidden_size = self.hid_dim,
            bias = True,
            bidirectional = self.bidirectional,
            dropout = self.dropout
        )

        x, _ = self.rnn(x)

        out = self.do(self.relu3(self.linear1(x)))

        return out, feat_len




