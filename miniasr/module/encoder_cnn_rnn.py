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

                                                                # B * 1 * T * 240
        self.conv1 = nn.Conv2d(1, 32, (11,11), padding=(5,5))      # B * 32 * T * 240
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 32, (11,11), padding=(5,5) )   # B * 32 * T * 240
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,2))         # B * 32 * T * 120


        self.conv3 = nn.Conv2d(16, 32, (11,11),padding=(5,5))      # B * 32 * T//2 * 120
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(32, 32, (11,11), padding=(5,5) )   # B * 32 * T//2 * 120
        self.relu4 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,2))         # B * 32 * T//4 * 60
        
        # RNN model
        self.rnn = getattr(nn, "GRU")(
            input_size=1920,
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
        feat = torch.reshape(feat, (32,1,-1,240))
        
       
        out = self.conv1(feat)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool1(out)

        # out = self.conv3(out)
        # out = self.relu3(out)
        # out = self.conv4(out)
        # out = self.relu4(out)
        # out = self.maxpool2(out)
        print(out.shape)
        out = out.permute((0,2,3,1))
        out = torch.reshape(out, (32, -1, 1920))

        if not self.training:
            self.rnn.flatten_parameters()

        out, _ = self.rnn(out)

        return out, feat_len
