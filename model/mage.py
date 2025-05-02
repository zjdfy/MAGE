import numpy as np
import torch
import torch.nn as nn
from model.networks import DiffMLP


class MAGE(nn.Module):
    def __init__(
        self,
        arch,
        nfeats,
        latent_dim=256,
        num_layers=8,
        dropout=0.1,
        dataset="amass",
        sparse_dim=54,
        **kargs,
    ):
        super().__init__()

        self.arch = DiffMLP
        self.dataset = dataset

        self.input_feats = nfeats
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sparse_dim = sparse_dim

        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.input_process = nn.Linear(self.input_feats, self.latent_dim)

        self.mlp = self.arch(
            self.latent_dim, seq=kargs.get("input_motion_length"), num_layers=num_layers,first=True,later=False,core=True
        )
        self.mlp_later = self.arch(
            self.latent_dim, seq=kargs.get("input_motion_length"), num_layers=num_layers,first=True,later=True
        )
        self.mlp_later2 = self.arch(
            self.latent_dim, seq=kargs.get("input_motion_length"), num_layers=num_layers,first=True,later=True
        )
        self.embed_timestep = TimestepEmbeding(self.latent_dim)
        self.sparse_process = nn.Linear(self.sparse_dim, self.latent_dim)
        self.output_process = nn.Linear(self.latent_dim, self.input_feats)
        #####change
        self.first_process = nn.Linear(self.latent_dim, 36)
        self.first_recover = nn.Linear(36, self.latent_dim)
        self.second_process = nn.Linear(self.latent_dim, 66)
        self.second_recover = nn.Linear(66, self.latent_dim) 
        self.small = nn.Linear(self.latent_dim*2, self.latent_dim)
    def mask_cond_sparse(self, cond, force_mask=True):
        bs, n, c = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                bs, 1, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def forward(self, x, timesteps, sparse_emb, force_mask=False):
        """
        x: [batch_size, nframes, nfeats], denoted x_t in the paper
        sparse: [batch_size, nframes, sparse_dim], the sparse features
        timesteps: [batch_size] (int)
        """
        flag=True
        '''if flag:
            print('x',x.shape)
            print('sparse',sparse_emb.shape)
            print('timesteps',timesteps.shape)
            flag=False
            '''
        emb = self.embed_timestep(timesteps)  # time step embedding : [bs, 1, d]
        # Pass the sparse signal to a FC
        sparse_emb = self.sparse_process(
            self.mask_cond_sparse(sparse_emb, force_mask=force_mask)
        )

        # Pass the input to a FC
        x = self.input_process(x)
        # Concat the sparse feature with input
        x_latent = torch.cat((sparse_emb, x), axis=-1)
        output = self.mlp(x_latent, emb)
        output1 = self.first_process(output)
        output1_recover = self.first_recover(output1)
        x_latent1 = torch.cat(( output,sparse_emb,output1_recover), axis=-1)
        output = self.mlp_later(x_latent1,emb)
        output2 = self.second_process(output)
        output2_recover = self.second_recover(output2)
        x_latent2 = torch.cat((output,sparse_emb,output2_recover), axis=-1)
        output = self.mlp_later2(x_latent2,emb)
        output = self.output_process(output)
        if self.training:
            return output1, output2, output  
        else:
            return output  

class TimestepEmbeding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)# (max_len * 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)# (max_len * 1 * d_model, can just use unsqueeze(1))
        self.register_buffer("pe", pe)

    def forward(self, timesteps):
        return self.pe[timesteps]
