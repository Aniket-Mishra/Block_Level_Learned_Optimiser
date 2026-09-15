"""Generic support-input encoder used by the learned optimizer."""

import torch
import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class TaskEncoderGeneric(nn.Module):
    """Architecture-agnostic task encoder.

    Flattens the support inputs and maps them through an MLP to a single task
    embedding of size config_params['embedding_size'], averaging over the
    batch. Works for any input shaped [B, ...].
    """

    def __init__(self, config_params):
        super().__init__()
        self.config_params = config_params
        emb_dim = config_params["embedding_size"]

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            LambdaLayer(lambda x: torch.mean(x, dim=0)),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return self.net(x)
