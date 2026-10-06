import random
import numpy as np
import torch
import torch.nn as nn

def reset_random(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

reset_random()

class ResidualBlock(nn.Module):
    def __init__(self,features,dropout=0.2):
        super().__init__()
        self.block=nn.Sequential(
            nn.Linear(features,features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features,features)
        )
        self.activation=nn.ReLU()

    def forward(self,x):
        return self.activation(x+self.block(x))

class MultiOutputResidualSurrogate(nn.Module):
    def __init__(self,input_dim=5,hidden_dim=128,blocks=3,dropout=0.2):
        super().__init__()
        self.shared=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.residual_blocks=nn.Sequential(*[ResidualBlock(hidden_dim,dropout) for _ in range(blocks)])
        self.t_out_head=nn.Linear(hidden_dim,1)
        self.bw_head=nn.Linear(hidden_dim,1)
        self.p_peak_head=nn.Linear(hidden_dim,1)

    def forward(self,x):
        x=self.shared(x)
        x=self.residual_blocks(x)
        t_out=self.t_out_head(x)
        bw=self.bw_head(x)
        p_peak=self.p_peak_head(x)
        return torch.cat([t_out,bw,p_peak],dim=1)

class MultiOutputMSELoss(nn.Module):
    def __init__(self,weights=None):
        super().__init__()
        self.weights=torch.tensor(weights if weights is not None else [1.0,1.0,1.0],dtype=torch.float32)

    def forward(self,pred,target):
        weights=self.weights.to(pred.device)
        loss=(pred-target)**2
        return (loss*weights).mean()

def create_model(input_dim=5,hidden_dim=128,blocks=3,dropout=0.2):
    return MultiOutputResidualSurrogate(input_dim,hidden_dim,blocks,dropout)