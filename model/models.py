import numpy as np
from numpy import log
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------
# --------------------------- PixelCNN ----------------------------------
# -----------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, block):
        super(ResBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.exclusive = kwargs.pop('exclusive')
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        _, _, kh, kw = self.weight.shape
        self.register_buffer('mask', torch.ones([kh, kw]))
        self.mask[kh // 2, kw // 2 + (not self.exclusive):] = 0
        self.mask[kh // 2 + 1:] = 0
        self.weight.data *= self.mask

        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.conv2d(x, self.mask * self.weight, self.bias,
                                    self.stride, self.padding, self.dilation,
                                    self.groups)

    def extra_repr(self):
        return (super(MaskedConv2d, self).extra_repr() +
                ', exclusive={exclusive}'.format(**self.__dict__))

    
class PixelCNN(nn.Module):    
    def __init__(self, **kwargs):
        super(PixelCNN, self).__init__()
        self.L = kwargs['L']
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.kernel_size = kwargs['kernel_size']
        self.bias = kwargs['bias']
        self.z2 = kwargs['z2']
        self.res_block = kwargs['res_block']
        self.x_hat_clip = kwargs['x_hat_clip']
        self.final_conv = kwargs['final_conv']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']

        self.dtype = torch.float32
        self.padding = self.kernel_size // 2
        self.n_pixels_out = 1

        if self.bias and not self.z2:
            self.register_buffer('x_hat_mask', torch.ones([self.L] * 2))
            self.x_hat_mask[0, 0] = 0
            self.register_buffer('x_hat_bias', torch.zeros([self.L] * 2))
            self.x_hat_bias[0, 0] = 0.5
        
        layers = []
        layers.append(
            MaskedConv2d(
                1,
                1 if self.net_depth == 1 else self.net_width,
                self.kernel_size,
                padding=self.padding,
                bias=self.bias,
                exclusive=True))
        for _ in range(self.net_depth - 2):
            if self.res_block:
                layers.append(
                    self._build_res_block(self.net_width, self.net_width))
            else:
                layers.append(
                    self._build_simple_block(self.net_width, self.net_width))
        if self.net_depth >= 2:
            layers.append(
                self._build_simple_block(
                    self.net_width, self.net_width if self.final_conv else 1))
        if self.final_conv:
            layers.append(nn.PReLU(self.net_width, init=0.5))
            layers.append(nn.Conv2d(self.net_width, 1, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
 
    
    def _build_simple_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels, init=0.5))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.kernel_size,
                padding=self.padding,
                bias=self.bias,
                exclusive=False))
        block = nn.Sequential(*layers)
        return block

    def _build_res_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, in_channels, 1, bias=self.bias))
        layers.append(nn.PReLU(in_channels, init=0.5))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.kernel_size,
                padding=self.padding,
                bias=self.bias,
                exclusive=False))
        block = ResBlock(nn.Sequential(*layers))
        return block
    
    def forward(self, x):
        x_hat = self.net(x)

        if self.x_hat_clip:
            with torch.no_grad():
                delta_x_hat = torch.clamp(x_hat, self.x_hat_clip, 1 - self.x_hat_clip) - x_hat
            assert not delta_x_hat.requires_grad
            x_hat = x_hat + delta_x_hat
        if self.bias and not self.z2:
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias

        return x_hat
    
    def sample(self, batch_size):
        sample = torch.zeros(
            [batch_size, 1, self.L, self.L],
            dtype=self.dtype,
            device=self.device)
        for i in range(self.L):
            for j in range(self.L):
                x_hat = self.forward(sample)
                sample[:, :, i, j] = torch.bernoulli(
                    x_hat[:, :, i, j]).to(self.dtype) * 2 - 1

        if self.z2:
            flip = torch.randint(
                2, [batch_size, 1, 1, 1],
                dtype=sample.dtype,
                device=sample.device) * 2 - 1
            sample *= flip

        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = (torch.log(x_hat + self.epsilon) * mask +
                    torch.log(1 - x_hat + self.epsilon) * (1 - mask))
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob
    
    def log_prob(self, sample):
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)

        if self.z2:
            sample_inv = -sample
            x_hat_inv = self.forward(sample_inv)
            log_prob_inv = self._log_prob(sample_inv, x_hat_inv)
            log_prob = torch.logsumexp(
                torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2) 

        return log_prob
    
    def flip(self, sample):
        if self.z2:

            batch_size = sample.shape[0]
            flip = torch.randint(
                2, [batch_size, 1, 1, 1],
                dtype=sample.dtype,
                device=sample.device) * 2 - 1
            sample *= flip
        return sample
    
    def energy(self, sample):
        term = sample[:, :, 1:, :] * sample[:, :, :-1, :]
        term = term.sum(dim=(1, 2, 3))
        output = term
        term = sample[:, :, :, 1:] * sample[:, :, :, :-1]
        term = term.sum(dim=(1, 2, 3))
        output += term

        term = sample[:, :, 0, :] * sample[:, :, -1, :]
        term = term.sum(dim=(1, 2))
        output += term
        term = sample[:, :, :, 0] * sample[:, :, :, -1]
        term = term.sum(dim=(1, 2))
        output += term

        output *= -1
        return output
    
    def loss_RL(self, beta, batch_size):
        with torch.no_grad():
            sample, x_hat = self.sample(batch_size)
        assert not sample.requires_grad
        assert not x_hat.requires_grad
        log_prob = self.log_prob(sample)
        with torch.no_grad():
            energy = self.energy(sample)
            loss = log_prob + beta * energy
        assert not energy.requires_grad
        assert not loss.requires_grad
        loss_reinforce = torch.sum((loss - loss.mean()) * log_prob)
        return loss_reinforce, loss

# -----------------------------------------------------------------------
# --------------------------- MLP/CNN -----------------------------------
# -----------------------------------------------------------------------

class MLP_SL(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=32, depth = 3, output_dim=2):
        super(MLP_SL, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(1, depth):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

class MLP_LBC(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=32, depth = 3, output_dim=2):
        super(MLP_LBC, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims))
        layers.append(nn.LeakyReLU())
        
        # Hidden layers
        for i in range(1, depth):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.LeakyReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

class MLP_PBM(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=32, depth = 3, output_dim=1):
        super(MLP_PBM, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(1, depth):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

class CNN_SL(nn.Module):
    def __init__(self, hidden_channels=10, hidden_dims=10):
        super(CNN_SL, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels, 3, padding=2, padding_mode='circular')
        self.ln = nn.LayerNorm(6)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_channels * 6 * 6, hidden_dims)
        self.ln1 = nn.LayerNorm(hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.ln2 = nn.LayerNorm(hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 2)

    def forward(self, x):
        x = self.conv1(x)  
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
        
class CNN_LBC(nn.Module):
    def __init__(self, hidden_channels=10, hidden_dims=10):
        super(CNN_LBC, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels, 3, padding=2, padding_mode='circular')
        self.ln = nn.LayerNorm(6)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_channels * 6 * 6, hidden_dims)
        self.ln1 = nn.LayerNorm(hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.ln2 = nn.LayerNorm(hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 2)

    def forward(self, x):
        x = self.conv1(x)  
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
        
class CNN_PBM(nn.Module):
    def __init__(self,hidden_channels=10, hidden_dims=10):
        super(CNN_PBM, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels, 3, padding=2, padding_mode='circular')
        self.ln = nn.LayerNorm(6)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_channels * 6 * 6, hidden_dims)
        self.ln1 = nn.LayerNorm(hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.ln2 = nn.LayerNorm(hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 1)

    def forward(self, x):
        x = self.conv1(x)  
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

