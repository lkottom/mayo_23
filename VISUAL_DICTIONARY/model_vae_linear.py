import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Autoencoder neural network for feature compression and reconstruction.

    This class defines an autoencoder model composed of an encoder and a decoder.
    The encoder reduces the dimensionality of the input data, and the decoder
    attempts to reconstruct the original data from the encoded representation. The 
    encoder and decoder layers are connected by nn.linear and nn.LeakyReLU layers. 

    Args:
        patch_size (int): The size of the input patch (width/height).
        input_channels (int): The number of input channels in the image (default: 3).

    Attributes:
        input_dim (int): The total number of input features after flattening the patch.

    Example:
        autoencoder = Autoencoder(patch_size=64, input_channels=3)
    """

    def __init__(self, patch_size, input_channels=3):
        super().__init__()

        hidden_sizes = [256, 128, 64, 32, 16, 8]

        self.input_dim = input_channels * patch_size * patch_size

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_sizes[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[3], hidden_sizes[4]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[4], hidden_sizes[5])
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_sizes[5], hidden_sizes[4]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[4], hidden_sizes[3]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[3], hidden_sizes[2]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[2], hidden_sizes[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[0], self.input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Encoded representation and reconstructed output.

        The forward pass first flattens the input tensor, passes it through the encoder
        to obtain the encoded representation, and then through the decoder to reconstruct
        the original data. The reconstructed output is reshaped to the original shape.
        """
        x_flat = x.view(x.size(0), -1)
        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)
        decoded = decoded.view(decoded.size(0), x.size(1), x.size(2), x.size(3))

        return encoded, decoded
        
        
if __name__ == "__main__":
    # Create an instance of the Autoencoder model
    patch_size = 16
    autoencoder = Autoencoder(patch_size)

    # Generate a random 16x16 image as input to the model
    batch_size = 1
    random_image = torch.randn(batch_size, 3, 16, 16)

    

    # Pass the image through the model
    latent, reconstructed_image = autoencoder(random_image)

    
    # Check the output shape
    print("Input Image Shape:", random_image.shape)
    print("Reconstructed Image Shape:", reconstructed_image.shape)
    print("Latent Shape:", latent.shape)
        
        
    