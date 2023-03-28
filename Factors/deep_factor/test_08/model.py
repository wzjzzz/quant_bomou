import torch
from torch import nn


class lstm(nn.Module):
    def __init__(self, params, gru=False, attention=False):
        super(lstm, self).__init__()
        
        self.input_dim = params['input_dim']
        lstm_dim = params['lstm_dim']
        lstm_num = params['lstm_num']
        output_dim = params['output_dim']
        self.gru = params['gru'] if 'gru' in params else gru
        attention = params['attention'] if 'attention' in params else attention
        nhead = params['nhead'] if 'nhead' in params else 8
        attention_num = params['attention_num'] if 'attention_num' in params else 1

        if self.gru:
            self.l1 = nn.GRU(input_size=self.input_dim, hidden_size=lstm_dim, num_layers=lstm_num, batch_first=True, bidirectional=False)
        else:
            self.l1 = nn.LSTM(input_size=self.input_dim, hidden_size=lstm_dim, num_layers=lstm_num, batch_first=True, bidirectional=False)


        if attention:
            transformer_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=nhead, batch_first=True)
            self.attention_layer = nn.TransformerEncoder(transformer_layer, num_layers=attention_num)
        else:
            self.attention_layer = None
        
        self.output_layer = nn.Linear(lstm_dim, output_dim)
        self.scale_layer = nn.Tanh()

    


    def forward(self, x):
        if self.attention_layer:
            x = self.attention_layer(x)
        if not self.gru:
            x = self.l1(x)[1][0]
            x = x[-1]
        else:
            x = self.l1(x)[1]
            x = x[-1]
        x = self.output_layer(x)
        x = self.scale_layer(x)
        return x

    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)


    def freeze_layer(self):
        #
        if self.attention_layer:
            self.attention_layer.requires_grad = False
        self.l1.requires_grad = False





class mlp(nn.Module):
    def __init__(self, params):
        super(mlp, self).__init__()

        self.input_dim = params['input_dim']
        dims = params['dims']
        output_dim = params['output_dim']

        layers = []
        input_dim = self.input_dim
        for dim in dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())  
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(0.5))
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Tanh())

        self.layer = nn.Sequential(*layers)


    
    def forward(self, x):
        x = self.layer(x)
        return x

    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    
    
    def freeze_layer(self):
        #
        first_meet = False
        for i in range(len(self.layer)-1, -1, -1):
            if not first_meet:
                if isinstance(self.layer[i], nn.Linear):
                    first_meet = True
                continue
            self.layer[i].requires_grad = False

