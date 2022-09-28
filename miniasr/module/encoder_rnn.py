'''
    File      [ encoder_rnn.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ RNN-based encoder. ]
'''

import torch
from torch import nn


class RNNEncoder(nn.Module):
    '''
        RNN-based encoder.
        in_dim [int]: input feature dimension
        hid_dim [int]: hidden feature dimension
        n_layers [int]: number of layers
        module [str]: RNN model type
        dropout [float]: dropout rate
        bidirectional [bool]: bidirectional encoding
    '''

    def __init__(self, in_dim, hid_dim, n_layers, module='LSTM',
                 dropout=0, bidirectional=True):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, (11,41), stride=(2,2), padding=(5, 20), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)  
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 32, (11,21), stride=(1,2), padding=(5, 10), bias=False)
        self.batch_norm2 = nn.BatchNorm2d(32)  
        self.relu2 = nn.ReLU()


        # RNN model
        self.rnn = getattr(nn, module)(
            input_size=((in_dim//2+1)//2 + 1)*32,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True)

        # Output dimension
        # Bidirectional makes output size * 2
        self.out_dim = hid_dim * (2 if bidirectional else 1)

    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        '''
            Input:
                feat [float tensor]: acoustic feature sequence
                feat_len [long tensor]: feature lengths
            Output:
                out [float tensor]: encoded feature sequence
                out_len [long tensor]: encoded feature lengths
        '''

        feat_size = feat.size()
        
        feat = feat.reshape(feat_size[0], 1, -1, feat_size[2])
        
        x = self.conv1(feat)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        # print(x.size())
        
        x = x.permute((0,2,1,3))
        x = x.reshape(feat_size[0], -1, ((feat_size[2]//2+1)//2 + 1)*32)


        if not self.training:
            self.rnn.flatten_parameters()

        out, _ = self.rnn(x)

        feat_len = torch.round(torch.mul(feat_len, 0.5)).to(torch.int32)

        return out, feat_len
