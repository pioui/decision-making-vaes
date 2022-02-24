import logging

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as db

class EncoderB0(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        # TODO: describe architecture and choice for people 
        super().__init__()
        logging.info("Using MF encoder with convolutions")
        # FOR TRENTO
        self.fc_layer = nn.Linear(in_features=65, out_features=28*28)
        
        self.encoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=64, out_channels=n_hidden, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=0.1),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)


        # FOR TRENTO
        x = x.view(n_batch, 65)
        x = self.fc_layer(x)

        x_reshape = x.view(n_batch, 1, 28, 28)

        q = self.encoder_cv(x_reshape)
        q = q.view(n_batch, -1)

        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()
        variational_dist = db.Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=variational_dist, sum_last=True
        )

class EncoderB1(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        # TODO: describe architecture and choice for people 
        super().__init__()
        logging.info("Using MF encoder with convolutions")
        # FOR TRENTO
        self.fc_layer = nn.Linear(in_features=65, out_features=28*28)
        
        self.encoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=256, out_channels=n_hidden, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=0.1),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)


        # FOR TRENTO
        x = x.view(n_batch, 65)
        x = self.fc_layer(x)

        x_reshape = x.view(n_batch, 1, 28, 28)

        q = self.encoder_cv(x_reshape)
        q = q.view(n_batch, -1)

        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()
        variational_dist = db.Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=variational_dist, sum_last=True
        )

class EncoderB2(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        # TODO: describe architecture and choice for people 
        super().__init__()
        logging.info("Using MF encoder with convolutions")
        # FOR TRENTO
        self.fc_layer = nn.Linear(in_features=65, out_features=120*120)
        
        self.encoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=256, out_channels=n_hidden, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=0.1),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)


        # FOR TRENTO
        x = x.view(n_batch, 65)
        x = self.fc_layer(x)

        x_reshape = x.view(n_batch, 1, 120, 120)

        q = self.encoder_cv(x_reshape)
        q = q.view(n_batch, -1)

        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()
        variational_dist = db.Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=variational_dist, sum_last=True
        )


class EncoderB3(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Linear(in_features=65, out_features=512),            
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=256),            
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=256, out_features=128),            
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.1),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=0.1),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)


        # FOR TRENTO
        x = x.view(n_batch, 65)
        x = self.fc_layer(x)

        x_reshape = x.view(n_batch, 1, 28, 28)

        q = self.encoder_cv(x_reshape)
        q = q.view(n_batch, -1)

        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()
        variational_dist = db.Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=variational_dist, sum_last=True
        )