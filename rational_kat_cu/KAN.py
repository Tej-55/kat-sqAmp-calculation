from rational_kat_cu.kat_rational import KAT_Group
import torch.nn as nn

class KAN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            out_features=None,
            act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
            bias=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, out_features, bias=bias)
        self.act1 = KAT_Group(mode = act_cfg['act_init'][1])


    def forward(self, x):
        x = self.act1(x)
        x = self.fc1(x)
        return x