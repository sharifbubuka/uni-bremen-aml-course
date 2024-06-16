import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OmniGlotEncoder(nn.Module):
    """
    Represents the network f which maps pre-processed Omniglot images (shape: 1, 28,1) to embeddings
    https://arxiv.org/pdf/1703.05175.pdf
    """

    def __init__(self, emb_dim: int, _hidden_channels: int = 64):
        """
        Initializes the encoder.
        :param emb_dim: The dimensionality of the embedding.
        :param _hidden_channels: The number kernels of of the hidden conv layers.
        """
        super().__init__()

        self.emb_dim = emb_dim
        # Four encoder blocks
        self.main = nn.Sequential(
            self.encoder_block(1, _hidden_channels),
            self.encoder_block(_hidden_channels, _hidden_channels),
            self.encoder_block(_hidden_channels, _hidden_channels),
            self.encoder_block(_hidden_channels, self.emb_dim),
        )

    def encoder_block(self, input_channels: int, output_channels: int):
        # Paper: Each block comprises a 64-filter 3×3 convolution,
        # batch normalization layer[10], a ReLU nonlinearity and a 2×2 max-pooling layer.
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return torch.flatten(self.main(x), start_dim=1)


if __name__ == "__main__":
    emb_dim = 128
    encoder = OmniGlotEncoder(emb_dim)
    batch_size = 8
    mock_batch = torch.randn(batch_size, 1, 28, 28)
    mock_embddings = encoder(mock_batch)
    assert mock_embddings.shape == (batch_size, emb_dim)
