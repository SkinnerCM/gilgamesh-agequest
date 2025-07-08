import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


class Autoencoder(nn.Module):
    """
    A simple fully-connected autoencoder in Flax.

    Attributes:
        encoder_dims: Sequence[int]  # sizes of encoder hidden layers
        latent_dim: int             # dimensionality of latent space
        decoder_dims: Sequence[int] # sizes of decoder hidden layers
    """
    encoder_dims: Sequence[int]
    latent_dim: int
    decoder_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        # Encoder
        for dim in self.encoder_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        # Latent representation
        z = nn.Dense(self.latent_dim, name="latent_layer")(x)

        # Decoder
        x = z
        for dim in self.decoder_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        # Reconstruction
        recon = nn.Dense(self.encoder_dims[0], name="reconstruction")(x)
        return recon, z

    def encode(self, x):
        """Return the latent encoding of input x."""
        x_enc, z = self.__call__(x)
        return z

    def decode(self, z):
        """Reconstruct input from latent code z."""
        x = z
        for dim in self.decoder_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        recon = nn.Dense(self.encoder_dims[0], name="reconstruction")(x)
        return recon
