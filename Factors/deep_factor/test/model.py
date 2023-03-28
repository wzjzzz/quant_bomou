import torch
from torch import nn
#https://zhuanlan.zhihu.com/p/537040171
#这里默认是一个batch内刚好是同一时刻的一池子股票
#attention机制改了一下


class FeatureExtractor(nn.Module):
    # input: batchsize * sequence_len * raw_feature_len
    def __init__(self, input_dim, proj_dim, embedding_dim, num_layers):
        super(FeatureExtractor, self).__init__()

        self.proj = nn.Linear(input_dim, proj_dim)
        self.leakyrulu = nn.LeakyReLU()
        self.gru = nn.GRU(input_size=proj_dim, hidden_size=embedding_dim, num_layers=num_layers, batch_first=True, bidirectional=False)

    def forward(self, x):
        x = self.proj(x)
        x = self.leakyrulu(x)
        x = self.gru(x)[1]
        x = x[-1]
        return x


class Post_FactorEncoder(nn.Module):
    # 输出得到后验z的mu和std
    # z我们所需要的因子
    def __init__(self, embedding_dim, portfolio_cnt, z_dim):
        super(Post_FactorEncoder, self).__init__()

        self.portfolio_layer = nn.Linear(embedding_dim, portfolio_cnt)
        self.portfolio_scale = nn.Softmax(dim=0)

        self.mu_layer = nn.Linear(portfolio_cnt, z_dim)
        self.std_layer = nn.Linear(portfolio_cnt, z_dim)
        self.std_scale = nn.Softplus()  #我的理解：高收益高风险。


    def forward(self, y, e):
        # y: y_true;     [batchsize, 1]
        # e: embedding   [batchsize, feature_len]

        weights = self.portfolio_scale(self.portfolio_layer(e))
        yp = (torch.mm(weights.T, y)).T

        mu_post = self.mu_layer(yp)
        std_post = self.std_scale(torch.abs(self.std_layer(yp)))
        #原来的激活函数特别容易出现很小的值，导致矩阵奇异。而且按直观理解，应该mu越小风险也越大，即做空风险也很大。所以change2：加一个abs
        
        return mu_post, std_post




class Prior_FactorEncoder(nn.Module):
    # 输出得到先验的z 的mu和std
    # attention，这里是共用一个q。逻辑是我们对所有的股票一样地查询？
    def __init__(self, embedding_dim, attention_dim, value_dim, nheads, z_dim):
        super(Prior_FactorEncoder, self).__init__()
        self.nheads = nheads
        self.k_layer = nn.Linear(embedding_dim, attention_dim*nheads)
        self.v_layer = nn.Linear(embedding_dim, value_dim*nheads)
        self.q = nn.Parameter(torch.ones(nheads, attention_dim))


        self.mu_layer = nn.Linear(nheads*value_dim, z_dim)
        self.std_layer = nn.Linear(nheads*value_dim, z_dim)
        self.std_scale = nn.Softplus()
    


    def forward(self, e):
        batchsize = e.shape[0]
        k = self.k_layer(e).reshape(batchsize, self.nheads, -1)
        v = self.v_layer(e).reshape(batchsize, self.nheads, -1)

        attention_score_numerator = torch.clip(torch.einsum('abc,bd -> ab', k, self.q), min=1e-6)  #batchsize*nheads
        norm_k = torch.norm(k, dim=2, p=2) * torch.norm(self.q, dim=1, p=2)
        norm_q = torch.norm(self.q, dim=1, p=2)
        attention_score_denominator = torch.clip(torch.einsum('ab, b -> ab', norm_k, norm_q), min=1e-6)  #batchsize*nheads
        attention_score = attention_score_numerator / attention_score_denominator

        h_attention = torch.einsum('ab, abc -> bc', attention_score, v)   #nheads*value_dim
        h_attention_concat = h_attention.reshape(1, -1)

        mu_prior = self.mu_layer(h_attention_concat)
        std_prior = self.std_scale(torch.abs(self.std_layer(h_attention_concat)))
        print(std_prior)

        return mu_prior, std_prior


        

class AlphaLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(AlphaLayer, self).__init__()

        self.proj = nn.Linear(embedding_dim, hidden_dim)
        self.leakyrelu = nn.LeakyReLU()

        self.mu_layer = nn.Linear(hidden_dim, 1)
        self.std_layer = nn.Linear(hidden_dim, 1)
        self.std_scale = nn.Softplus()


    def forward(self, e):
        # e: embedding         [batchsize, feature_len]
        hidden_e = self.proj(e)
        hidden_e = self.leakyrelu(hidden_e)

        mu_alpha = self.mu_layer(hidden_e)
        std_alpha = self.std_scale(torch.abs(self.std_layer(hidden_e)))

        return mu_alpha, std_alpha



class FactorDecoder(nn.Module):
    # 输出先验或后验的z预测yp的mu和std
    def __init__(self, embedding_dim, hidden_dim, z_dim):
        super(FactorDecoder, self).__init__()

        self.alpha_layer = AlphaLayer(embedding_dim, hidden_dim)
        self.beta_layer = nn.Linear(embedding_dim, z_dim)


    def forward(self, z, e):
        # z: output from encoder.  [1*z_dim, 1*z_dim]
        # e: embedding   [batchsize, feature_len]
        beta = self.beta_layer(e)   

        mu_z, std_z = z[0], z[1]
        mu_alpha, std_alpha = self.alpha_layer(e)

        mu_y = mu_alpha + torch.mm(beta, mu_z.T)
        std_y = torch.sqrt(std_alpha**2 + torch.mm(beta, std_z.T)**2)

        return mu_y, std_y


class FactorVAE(nn.Module):
    #最终的模型
    def __init__(self, params):
        super(FactorVAE, self).__init__()

        input_dim = params['input_dim']
        proj_dim = params['proj_dim']
        embedding_dim = params['embedding_dim']
        num_layers = params['num_layers']
        portfolio_cnt = params['portfolio_cnt']
        z_dim = params['z_dim']
        attention_dim = params['attention_dim']
        value_dim = params['value_dim']
        nheads = params['nheads']
        hidden_dim = params['hidden_dim']

        self.feature_extractor = FeatureExtractor(input_dim, proj_dim, embedding_dim, num_layers)
        self.post_encoder = Post_FactorEncoder(embedding_dim, portfolio_cnt, z_dim)
        self.prior_encoder = Prior_FactorEncoder(embedding_dim, attention_dim, value_dim, nheads, z_dim)
        self.factor_decoder = FactorDecoder(embedding_dim, hidden_dim, z_dim)
    
    def forward(self, x, y, training=True):
        e = self.feature_extractor(x)
        z_prior = self.prior_encoder(e)
        y_pred = self.factor_decoder(z_prior, e)[0]   #输出均值当y_pred

        if not training:
            return y_pred
        
        z_post = self.post_encoder(y, e)
        y_rec = self.factor_decoder(z_post, e)
        return z_post, z_prior, y_rec, y_pred
    

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)



class FactorVAE_Loss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma

    def forward(self, z_post, z_prior, y, y_rec):
        loss1 = self.logloss(y, y_rec)
        loss2 = self.KLloss(z_post, z_prior)
        loss = loss1 + self.gamma * loss2
        return loss
    
    def logloss(self, y, y_rec):
        # batchsize*1
        mu, sigma = y_rec
        probs = torch.exp(-torch.square(y-mu)/(2*torch.square(sigma))) / torch.sqrt(2*sigma*torch.pi)
        log_probs = torch.log(probs)
        loss = -torch.mean(log_probs)
        return loss
    


    def KLloss(self, z_post, z_prior):
        # 1*z_dim
        mu_1, sigma_1 = z_post
        mu_2, sigma_2 = z_prior
        sigma_1 = torch.clip(sigma_1, min=1e-6)
        sigma_2 = torch.clip(sigma_2, min=1e-6)
        #防止矩阵不可逆，KL散度对扰动不敏感吧？

        Sigma_1 = torch.diag(sigma_1.reshape(-1))
        Sigma_1_det = Sigma_1.det()
        Sigma_2 = torch.diag(sigma_2.reshape(-1))
        Sigma_2_det = Sigma_2.det()
        Sigma_2_inv = Sigma_2.inverse()

        loss = (torch.mm(Sigma_2_inv, Sigma_1)).trace() + torch.mm(torch.mm((mu_1-mu_2), Sigma_2_inv), (mu_1-mu_2).T) - mu_1.shape[1] + torch.log(Sigma_2_det/Sigma_1_det)
        return 0.5 * loss


