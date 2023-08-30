import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)
    
# class UnFlatten(nn.Module):
#     def __init__(self, size):
#         super().__init__()
#         self.size = size

#     def forward(self, input):
#         return input.view(input.size(0), -1, self.size, self.size)
    
# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, patch_size, input_channels=3, latent_dim=8):
        super().__init__()
        
        hidden_sizes = [256, 128, 64, 32, 16, 8]
        
        
        self.input_dim = input_channels * patch_size * patch_size
        
        # Encoder layers
        self.encoder = nn.Sequential(
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
            nn.Linear(hidden_sizes[4], hidden_sizes[5]),
            nn.ReLU(True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_sizes[5], hidden_sizes[4]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[4], hidden_sizes[3]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[3], hidden_sizes[2]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[2], hidden_sizes[1]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[1], hidden_sizes[0]),
            nn.ReLU(True), 
            nn.Linear(hidden_sizes[0], self.input_dim),
            nn.Sigmoid()
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_sizes[5], latent_dim)
        self.fc_logvar = nn.Linear(hidden_sizes[5], latent_dim)


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
    
    # def loss_function(self, recon_x, x, mu, logvar, mask):
    #     # Reconstruction loss for the visible (non-masked) values
    #     rec_visible = torch.nn.MSELoss(reduction='none')(recon_x * mask, x * mask)
    #     # Reconstruction loss for the masked values
    #     rec_masked = torch.nn.MSELoss(reduction='none')(recon_x * (1 - mask), x * (1 - mask))

    #     # Compute the mean reconstruction loss over both visible and masked values
    #     rec = torch.sum(rec_visible + rec_masked) / torch.sum(mask)

    #     # KL divergence loss as before
    #     kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    #     return rec, kl


    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        # z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(hidden)
        return x_reconstructed, mu, logvar


if __name__ == "__main__":
    # Create an instance of the Autoencoder model
    patch_size = 16
    autoencoder = Autoencoder(patch_size)

    # Generate a random 16x16 image as input to the model
    batch_size = 1
    random_image = torch.randn(batch_size, 3, patch_size, patch_size)
    random_image = random_image.view(random_image.size(0), -1)
    

    # Pass the image through the model
    reconstructed_image, mu, logvar = autoencoder(random_image)

    # # Generate a binary mask with a random percentage of values set to 1
    # mask_percentage = np.random.uniform(0.05, 0.7)
    # mask = torch.rand(random_image.shape) < mask_percentage

    # # Apply the mask to the random_image to replace values with -1000
    # dummy_value = -1000
    # masked_image = torch.where(mask, dummy_value, random_image)
     

    
    # Pass the masked image through the model
    # reconstructed_image, mu, logvar = autoencoder(masked_image)

    # Check the output shape
    print("Input Image Shape:", random_image.shape)
    print("Reconstructed Image Shape:", reconstructed_image.shape)
    print("Mu Shape:", mu.shape)
    print("Logvar Shape:", logvar.shape)