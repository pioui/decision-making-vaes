import logging

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as db

class EncoderB0(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        65 -FC-> 28x28 -3x2D Conv 32-128->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()
        # FOR TRENTO
        self.fc_layer = nn.Linear(in_features=65, out_features=28*28)
        
        self.encoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=n_hidden, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
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
        """  
        65 -FC-> 28x28 -3x2D Conv 128-512->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()
        # FOR TRENTO
        self.fc_layer = nn.Linear(in_features=65, out_features=28*28)
        
        self.encoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=n_hidden, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
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
        """  
        65 -FC-> 120*120 -5x2D Conv 32-512->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()
        # FOR TRENTO
        self.fc_layer = nn.Linear(in_features=65, out_features=120*120)
        
        self.encoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=n_hidden, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
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
        """  
        65 - 3x FC 512-128 ->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Linear(in_features=65, out_features=512),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=512, out_features=256),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=256, out_features=n_hidden),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)

        x_reshape = x.view(n_batch, -1)

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

class EncoderB4(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        65 - 5x FC 2048-128 ->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Linear(in_features=65, out_features=2048),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=2048, out_features=1048),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=1048, out_features=512),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=512, out_features=256),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=256, out_features=n_hidden),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)

        x_reshape = x.view(n_batch, -1)

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

class EncoderB5(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        65 - 3x Conv 1D 32-128 ->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=64, out_channels=n_hidden, kernel_size=9),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)

        x_reshape = x.view(n_batch, 1, -1)

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

class EncoderB6(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        65 - 5x Conv 1D 32-512 ->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=256, out_channels=n_hidden, kernel_size=1),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
        )


        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)

        x_reshape = x.view(n_batch, 1, -1)

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

class EncoderB7(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        65 -FC-> 28x28 -2x2D Conv 128-256->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=128, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=128, out_channels=n_hidden, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)
        print(x.shape)

        q = self.encoder_cv(x)
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

class FCLayersA(nn.Module):
    def __init__(
        self, n_input, n_output, n_middle=None, dropout_rate=0.1, do_batch_norm=False
    ):
        super().__init__()
        n_middle = n_output if n_middle is None else n_middle
        self.to_hidden = nn.Linear(in_features=n_input, out_features=n_middle)
        self.do_batch_norm = do_batch_norm
        if do_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=n_middle)

        self.to_out = nn.Linear(in_features=n_middle, out_features=n_output)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = nn.SELU()
        # self.activation = nn.ReLU()

    def forward(self, x):
        res = self.to_hidden(x)
        if self.do_batch_norm:
            if res.ndim == 4:
                n1, n2, n3, n4 = res.shape
                res = self.batch_norm(res.view(n1 * n2 * n3, n4))
                res = res.view(n1, n2, n3, n4)
            elif res.ndim == 3:
                n1, n2, n3 = res.shape
                res = self.batch_norm(res.view(n1 * n2, n3))
                res = res.view(n1, n2, n3)
            elif res.ndim == 2:
                res = self.batch_norm(res)
            else:
                raise ValueError("{} ndim not handled.".format(res.ndim))
        res = self.activation(res)
        res = self.dropout(res)
        res = self.to_out(res)
        return res
    
class BernoulliDecoderA7(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        dropout_rate: float = 0.0,
        do_batch_norm: bool = True,
        patch_size: int = 11
    ):
        super().__init__()
        self.loc = FCLayersA(
            n_input, n_hidden, dropout_rate=dropout_rate, do_batch_norm=do_batch_norm
        )
        self.decoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=n_hidden, out_channels=128, kernel_size=1),
            nn.SELU(),
            nn.Upsample((1,2,1,1)),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=128, out_channels=n_output, kernel_size=3),
            nn.SELU(),
            nn.Upsample((1,2,1,1)),
            nn.Dropout(p=dropout_rate),
        )
        self.n_hidden = n_hidden

    def forward(self, x):
        n_batch = len(x)

        x_1d = self.loc(x)
        x_3d = x_1d.view(n_batch, self.n_hidden, 1, 1)
        means = self.decoder_cv(x_3d)

        means = nn.Sigmoid()(means)
        return means