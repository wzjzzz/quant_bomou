import torch
from torch import nn


class lstm(nn.Module):
    def __init__(self, params, gru=False, attention=False):
        super(lstm, self).__init__()
        
        input_dim = params['input_dim']
        lstm_dim = params['lstm_dim']
        lstm_num = params['lstm_num']
        output_dim = params['output_dim']
        self.gru = params['gru'] if 'gru' in params else gru
        attention = params['attention'] if 'attention' in params else attention
        nhead = params['nhead'] if 'nhead' in params else 8
        attention_num = params['attention_num'] if 'attention_num' in params else 1

        if self.gru:
            self.l1 = nn.GRU(input_size=input_dim, hidden_size=lstm_dim, num_layers=lstm_num, batch_first=True, bidirectional=False)
        else:
            self.l1 = nn.LSTM(input_size=input_dim, hidden_size=lstm_dim, num_layers=lstm_num, batch_first=True, bidirectional=False)


        if attention:
            transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True)
            self.attention_layer = nn.TransformerEncoder(transformer_layer, num_layers=attention_num)
        else:
            self.attention_layer = None
        
        self.output_layer = nn.Linear(lstm_dim, output_dim)
        self.scale_layer = nn.Tanh()

        self.revinlayer = RevIN1(num_features=input_dim)
        

    

    def forward(self, x):
        x = self.revinlayer(x, mode='norm')
        if self.attention_layer:
            x = self.attention_layer(x)
        x = self.revinlayer(x, mode='denorm')
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

        input_dim = params['input_dim']
        dims = params['dims']
        output_dim = params['output_dim']

        layers = []
        for dim in dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())  
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(0.5))
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Tanh())

        self.layer = nn.Sequential(*layers)
        self.revinlayer = RevIN1(num_features=input_dim)


    
    def forward(self, x):
        x = self.revinlayer(x, mode='norm')
        x = self.revinlayer(x, mode='denorm')
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



'''
https://github.com/zhangxjohn/Reversible-Instance-Normalization
paper: https://openreview.net/pdf?id=cGDAkQo1C0p
两种尝试：
1、(原论文)按特征归一化,即30*50获取(1*50)的mean,反归一化用50*1来训练    -- revin1
2、按batch对30*50归一化,(30*50)的mean,反归一化用30*50*1训练  -- revin2
'''
class RevIN1(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError('Only modes norm and denorm are supported.')
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class RevIN2(RevIN1):
    def __init__(self, sequance_len):
        super().__init__()
        self.sequence_len = sequance_len

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.sequence_len, self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.sequence_len, self.num_features))

    def _get_statistics(self, x):
        dim2reduce = 0
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
