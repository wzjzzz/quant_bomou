import torch
from torch import nn 

class Loss(nn.Module):
    def __init__(self, alpha=0, model=None):
        super(Loss, self).__init__()
        self.alpha = alpha
        self.model = model
        self.cos = nn.CosineSimilarity(dim=0)


    def forward(self, y_pred, rtn, type='mse', mean=False, regular=False):
        if type == 'mse':
            loss = torch.mean((y_pred - rtn)**2)
        
        if type == 'ic':
            loss = -self.corrcoef(y_pred, rtn, mean=mean)

        if type == 'rank_ic':
            loss = self.rank_ic(y_pred, rtn, mean=mean)
        
        if type == 'ccc':
            loss = 1
        
        if regular:
            regular_loss = 0
            for param in self.model.parameters():
                regular_loss += (self.alpha * torch.norm(param, 1) )
            loss += regular_loss

        return loss


    def corrcoef(self, x, y, mean):
        x = x - torch.mean(x, dim=0)
        y = y - torch.mean(y, dim=0)
        ic = self.cos(x, y)
        if mean:
            return ic.mean()
        else:
            return ic


    def rank_ic(self, x, y, mean):
        rank_x = x.argsort(dim=0).argsort(dim=0).float()
        rank_y = y.argsort(dim=0).argsort(dim=0).float()
        return self.corrcoef(rank_x, rank_y, mean)