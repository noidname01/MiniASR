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

        # CNN part
        

        self.conv1 = nn.Conv2d(1, 32, (5,5), padding=(2,2))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, (5,5), padding=(2,2))
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,2))

        self.conv3= nn.Conv2d(32, 32, (1,5), padding=(0,2))
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4= nn.Conv2d(32, 32, (1,5), padding=(0,2))
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,4))

        self.conv5= nn.Conv2d(32, 64, (1,5), padding=(0,2))
        self.batch_norm5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6= nn.Conv2d(64, 64, (1,5), padding=(0,2))
        self.batch_norm6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1,2))

        # RNN model
        self.rnn = getattr(nn, module)(
            input_size=64*15,
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
        feat = torch.reshape(feat, (feat_size[0], 1, -1, feat_size[2]))

        x = self.conv1(feat)
        # x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        # x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.maxpool1(x)

        x = self.conv3(x)
        # x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        # x = self.batch_norm4(x)
        x = self.relu4(x)

        x = self.maxpool2(x)

        x = self.conv5(x)
        # x = self.batch_norm5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        # x = self.batch_norm6(x)
        x = self.relu6(x)

        x = self.maxpool3(x)

        x = x.permute((0,2,1,3))


        x = torch.reshape(x, (feat_size[0], -1, 64*15))

        feat_len = feat_len.mul(0.5).floor().to(torch.int32)

        if not self.training:
            self.rnn.flatten_parameters()

        out, _ = self.rnn(x)


        return out, feat_len
