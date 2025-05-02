import torch.nn as nn
###############################
############ Layers ###########
###############################

class MLPblock(nn.Module):
    def __init__(self, dim, seq0, seq1, first=False, w_embed=True,later = False,core = False):
        super().__init__()

        self.w_embed = w_embed
        self.fc0 = nn.Conv1d(seq0, seq1, 1)

        if self.w_embed:
            if later:
                self.conct = nn.Linear(dim * 3, dim)
            elif first:
                self.conct = nn.Linear(dim * 2, dim)
            else:
                self.conct = nn.Identity()
            self.emb_fc = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim)
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, inputs):

        if self.w_embed:
            x = inputs[0]
            embed = inputs[1]
            x = self.conct(x) + self.emb_fc(self.act(embed))
        else:
            x = inputs

        x_ = self.norm0(x)
        x_ = self.fc0(x_)
        x_ = self.act(x_)
        x = x + x_

        x_ = self.norm1(x)
        x_ = self.fc1(x_)
        x_ = self.act(x_)

        x = x + x_

        if self.w_embed:
            return x, embed
        else:
            return x



class BaseMLP(nn.Module):
    def __init__(self, dim, seq, num_layers, w_embed=True, first=True, later=False,core = False):
        super().__init__()
        layers = []
        if core:
            amp = 2
        else:
            amp = 1
        for i in range(num_layers//3):#########change
            layers.append(
                MLPblock(dim, seq, seq, first and w_embed and i==0, w_embed, later and w_embed and i==0,core)
            )

        self.mlps = nn.Sequential(*layers)
        print(num_layers//3,'Denoiser layers in each stage')
    def forward(self, x):
        x = self.mlps(x)
        return x


###############################
########### Networks ##########
###############################


class DiffMLP(nn.Module):
    def __init__(self, latent_dim=512, seq=98, num_layers=12,first=True,later = False,core = False):
        super(DiffMLP, self).__init__()

        self.motion_mlp = BaseMLP(dim=latent_dim, seq=seq, num_layers=num_layers, w_embed=True, first=first,later=later,core=core)

    def forward(self, motion_input, embed):

        motion_feats = self.motion_mlp([motion_input, embed])[0]

        return motion_feats

