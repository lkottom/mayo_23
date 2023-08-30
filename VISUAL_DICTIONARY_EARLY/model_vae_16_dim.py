import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input):
        return input.view(input.size(0), -1, self.size, self.size)
    
# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, patch_size, input_channels=3):
        super().__init__()
        
        hidden_sizes = [256, 128, 64, 32, 16, 8]
        
        
        self.input_dim = input_channels * patch_size * patch_size
        
        # Encoder layers
        self.encoder = nn.Sequential(
            Flatten(), 
            nn.Linear(self.input_dim, hidden_sizes[0]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[3], hidden_sizes[4]),
            nn.ReLU(True),
            # nn.Linear(hidden_sizes[4], hidden_sizes[5]),
            # nn.ReLU(True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            # nn.Linear(hidden_sizes[5], hidden_sizes[4]),
            # nn.ReLU(True),
            nn.Linear(hidden_sizes[4], hidden_sizes[3]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[3], hidden_sizes[2]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[2], hidden_sizes[1]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[1], hidden_sizes[0]),
            nn.ReLU(True), 
            nn.Linear(hidden_sizes[0], self.input_dim),
            nn.Sigmoid(), 
            UnFlatten(patch_size)
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_sizes[4], hidden_sizes[4])
        self.fc_logvar = nn.Linear(hidden_sizes[4], hidden_sizes[4])

    def encode(self, x):
        hidden = self.encoder(x)
        mu, logvar = self.fc_mu(hidden), self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)
    
    def loss_function(self, recon_x, x, mu, logvar):
        # Important: both reconstruction and KL divergence loss have to be summed over all element!
        # Here we also sum the over batch and divide by the number of elements in the data later
        rec = torch.nn.MSELoss()(recon_x, x)
        
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return rec, kl

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar


if __name__ == "__main__":
    # Create an instance of the Autoencoder model
    patch_size = 32
    autoencoder = Autoencoder(patch_size)

    # Generate a random 16x16 image as input to the model
    batch_size = 1024
    random_image = torch.randn(batch_size, 3, 32, 32)
    

    # Pass the image through the model
    reconstructed_image, mu, logvar = autoencoder(random_image)

    # Check the output shape
    print("Input Image Shape:", random_image.shape)
    print("Reconstructed Image Shape:", reconstructed_image.shape)
    print("Mu Shape:", mu.shape)
    print("Logvar Shape:", logvar.shape)
