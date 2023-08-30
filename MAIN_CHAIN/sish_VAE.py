# The code is mostly borrow from the class 
# Berkeley's CS294-158 Deep Unsupervised Learning (https://github.com/rll/deepul)
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.net(x)

    
class SmallVectorQuantizedVAE(nn.Module):
    def __init__(self):
        super().__init__()
    

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )


        self.decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# Instantiate the model
model = SmallVectorQuantizedVAE(code_dim=8, code_size=128)

# Test with a dummy input
dummy_input = torch.randn(1, 3, 16, 16)  # Batch size of 1, RGB 16x16 image
output = model(dummy_input)
print(output.shape)
