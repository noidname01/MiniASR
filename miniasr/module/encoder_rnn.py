'''
    File      [ encoder_rnn.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ RNN-based encoder. ]
'''

import torch
from torch import nn

from conformer import Conformer


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

        # CNN
        self.conv1 = nn.Conv2d(1, 32, (11,41), stride=(1,2), padding=(5, 20))    
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 32, (11,21), stride=(1,2), padding=(5, 10)) 
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d((1,4))



        # Conformer model
        self.conformer = Conformer(
            input_dim = ((in_dim//2 + 1)//2 + 1)//4*32,
            # input_dim = in_dim,
            num_encoder_layers= n_layers,
            num_classes = hid_dim,
            conv_dropout_p = dropout,
        )

        # RNN model
        self.rnn = getattr(nn, "GRU")(
            # input_size=((in_dim//2+1)//2 + 1)//4*32,
            input_size=in_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True
        )

        # Output dimension
        # Bidirectional makes output size * 2
        self.out_dim = hid_dim 

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
        feat = torch.reshape(feat, (feat_size[0],1,-1,feat_size[2]))
        
        x = self.conv1(feat)
        # print(x.size())
        x = self.relu1(x)
        x = self.conv2(x)
        # print(x.size())
        x = self.relu2(x)

        x = self.maxpool1(x)

        


        x = x.permute((0,2,1,3))
        x = torch.reshape(x, (feat_size[0], -1, ((feat_size[2]//2+1)//2 + 1)//4*32))

        # feat_len = torch.mul(feat_len, 0.5)
        # feat_len = torch.floor(feat_len)
        # feat_len = feat_len.to(torch.int32)



        out, enc_len = self.conformer(x, feat_len)
        # print(feat_len)
        # print(enc_len)
        # out, _ = self.rnn(x)

        return out, enc_len
        # return out, enc_len
