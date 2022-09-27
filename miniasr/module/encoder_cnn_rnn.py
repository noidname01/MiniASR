'''
    File      [ encoder_rnn.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ RNN-based encoder. ]
'''

import torch
from torch import nn


class CNN_RNN_Encoder(nn.Module):
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

                                                                # B * 1 * T * 257
        self.conv1 = nn.Conv2d(1, 32, (11,41), stride=(2,2), padding=(5, 20))      # B * 32 * T * 128
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 32, (11,21), stride=(1,2), padding=(5, 10))   # B * 32 * T * 64
        self.relu2 = nn.ReLU()

        # RNN model
        self.rnn = getattr(nn, "GRU")(
            input_size=((in_dim//2+1)//2 + 1)*32,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
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
        batch_size = feat.size(dim=0)
        N = feat.size(dim=2)
        
        feat = torch.reshape(feat, (batch_size,1,-1,N))
        
    
        out = self.conv1(feat)
        # print(out.size())
        out = self.relu1(out)
        out = self.conv2(out)
        # print(out.size())
        out = self.relu2(out)
        # print(out.size())

        # out = self.conv3(out)
        # out = self.relu3(out)
        # out = self.conv4(out)
        # out = self.relu4(out)
        # out = self.maxpool2(out)
        
        out = out.permute((0,2,1,3))
        out = torch.reshape(out, (batch_size, -1, ((N//2+1)//2 + 1)*32))

        if not self.training:
            self.rnn.flatten_parameters()

        out, _ = self.rnn(out)

        # feat_len = torch.sub(feat_len, 10)
        feat_len = torch.mul(feat_len, 0.5)
        feat_len = torch.floor(feat_len)
        # feat_len = torch.sub(feat_len, 10)
        feat_len = feat_len.to(torch.int32)

        

        return out, feat_len
