import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, code_dim=8):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, code_dim)
        self.fc2 = nn.Linear(256 * 4 * 4, code_dim)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, code_dim=8):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(code_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.relu(self.fc(z))
        x = x.view(-1, 256, 4, 4)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.sigmoid(self.deconv3(x))
        return x

class VAE(nn.Module):
    def __init__(self, code_dim=8):
        super(VAE, self).__init__()
        self.encoder = Encoder(code_dim)
        self.decoder = Decoder(code_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# Instantiate the VAE model
vae_model = VAE(code_dim=8)

# Test with a dummy input
dummy_input = torch.randn(1, 3, 16, 16)  # Batch size of 1, RGB 16x16 image
output, _, _ = vae_model(dummy_input)
print(output.shape)  # Print the shape of the output
