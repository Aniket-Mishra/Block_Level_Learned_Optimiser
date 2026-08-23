"""Task encoders for the L2L baseline
base repo: Reference: https://github.com/annaVettoruzzo/L2L_with_attention
file: models/task_encoder_models.py
"""

import torch
import torch.nn as nn
import torchvision.models as models


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def cnn_block(in_dim, features_dim, kernel_size=3, padding="same"):
    return nn.Sequential(
        nn.Conv2d(in_dim, features_dim, kernel_size, padding=padding),
        nn.BatchNorm2d(features_dim, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))


class PretrainedTaskEncoder(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = cnn_block(1, 32)
        self.flat = nn.Flatten()
        self.net = nn.Sequential(nn.Linear(6272, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 100),
                                 nn.ReLU())
        self.cls = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.flat(x)
        x = self.net(x)
        return self.cls(x)


class TaskEncoderMNIST(torch.nn.Module):
    def __init__(self, pretrained_model_path, config_params):
        super().__init__()
        self.config_params = config_params

        pretrained_task_encoder = PretrainedTaskEncoder()
        pretrained_task_encoder.load_state_dict(torch.load(pretrained_model_path))

        self.pretrained_task_encoder = nn.Sequential(*list(pretrained_task_encoder.children())[:-1])
        # Freeze the parameters of the pretrained encoder
        for param in self.pretrained_task_encoder.parameters():
            param.requires_grad = False

        self.lmbd_layer = LambdaLayer(lambda x: torch.mean(x, dim=0))

        self.net2 = nn.Sequential(nn.Linear(100, 100),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(100, config_params['embedding_size']))

    def forward(self, x):
        x = self.pretrained_task_encoder(x)
        x = self.lmbd_layer(x)
        x = self.net2(x)
        return x


class TaskEncoderCIFAR(torch.nn.Module):
    def __init__(self, config_params):
        super().__init__()
        self.config_params = config_params

        pretrained_task_encoder = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        self.pretrained_task_encoder = nn.Sequential(*list(pretrained_task_encoder.children())[0])
        # Freeze the parameters of the pretrained encoder
        for param in self.pretrained_task_encoder.parameters():
            param.requires_grad = False

        self.net2 = nn.Sequential(nn.Flatten(),
                                  nn.Linear(512, 100),
                                  nn.ReLU(),
                                  LambdaLayer(lambda x: torch.mean(x, dim=0)),
                                  nn.Linear(100, 100),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(100, config_params['embedding_size']))

    def forward(self, x):
        x = self.pretrained_task_encoder(x)
        x = self.net2(x)
        return x


class TaskEncoderPMNIST(torch.nn.Module):
    def __init__(self, config_params):
        super().__init__()
        self.config_params = config_params

        self.net = nn.Sequential(nn.Linear(784, 25),
                                 nn.ReLU(),
                                 LambdaLayer(lambda x: torch.mean(x, dim=0)),
                                 nn.Linear(25, 25),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(25, config_params['embedding_size']))

    def forward(self, x):
        x = self.net(x)
        return x
