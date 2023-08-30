import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=8):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # New layer
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Latent layer
        self.latent_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim),  # Adjust the input size
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),  # Adjust the output size
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # Adjust the output size
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.latent_layer(encoded)
        decoded = self.decoder(latent)
        return decoded
    
# Create an instance of the Autoencoder
autoencoder = Autoencoder()

if __name__ == '__main__':
    dummy_input = torch.randn(16, 3, 32, 32)

    # Pass the dummy input through the autoencoder
    output = autoencoder(dummy_input)

    # Print the shapes of the intermediate and output tensors
    print("Input shape:", dummy_input.shape)
    encoded_output = autoencoder.encoder(dummy_input)
    print("Encoded shape:", encoded_output.shape)
    latent_output = autoencoder.latent_layer(encoded_output)
    print("Latent shape:", latent_output.shape)
    decoded_output = autoencoder.decoder(latent_output)
    print("Decoded shape:", decoded_output.shape)
    print("Output shape:", output.shape)