import torch
import torch.nn as nn

class Encoder_1D(nn.Module):
    """
    Encoder of the cVAE
    """

    def __init__(self, image_size=150, hidden_dim=50, z_dim=10, c=7):
        """
        :param image_size: Size of 1D "images" of data set i.e. spectrum size
        :param hidden_dim: Dimension of hidden layer
        :param z_dim: Dimension of latent space
        :param c: Dimension of conditioning variables

        """
        super().__init__()

        # nn.Linear(latent_dims, 512)
        self.layers_mu = nn.Sequential(
            nn.Linear(image_size + c, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim),
        )

        self.layers_logvar = nn.Sequential(
            nn.Linear(image_size + c, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, x):
        """
        Compute single pass through the encoder

        :param x: Concatenated images and corresponding conditioning variables
        :return: Mean and log variance of the encoder's distribution
        """
        mean = self.layers_mu(x)
        logvar = self.layers_logvar(x)
        return mean, logvar
class Decoder_1D(nn.Module):
    """
    Decoder of cVAE
    """

    def __init__(self, image_size=150, hidden_dim=50, z_dim=10, c=7):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(z_dim + c, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, image_size),
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        Compute single pass through the decoder

        :param z: Concatenated sample of hidden variables and the originally inputted conditioning variables
        :return: Mean of decoder's distirbution
        """
        mean = self.layers(z)
        return mean
class Kamile_CVAE_1D(nn.Module):
    """
        Base pytorch cVAE class
    """
    name = "Kamile_CVAE"
    def __init__(self, image_size=150, hidden_dim=50, z_dim=10, c=7, init_weights=True, **kwargs):
        """
        :param image_size: Size of 1D "images" of data set i.e. spectrum size
        :param hidden_dim: Dimension of hidden layer
        :param z_dim: Dimension of latent space (latent_units)
        :param c: Dimension of conditioning variables
        """
        super(Kamile_CVAE_1D, self).__init__()
        self.z_dim = z_dim # needed for inference
        # self.c = c
        # image_size=image_size
        # self.hidden_dim=hidden_dim
        self.encoder = Encoder_1D(image_size, hidden_dim, z_dim, c) # self.encoder = Encoder(latent_dims)
        self.decoder = Decoder_1D(image_size, hidden_dim, z_dim, c) # self.decoder = Decoder(latent_dims)

        if init_weights:
            self.init_weights()

        self.model_settings={
            "name":"Kamile_CVAE",
            "image_size":image_size,
            "hidden_dim":hidden_dim,
            "z_dim":z_dim,
            "c":c,
            "init_weights":init_weights
        }

    @classmethod
    def init_from_dict(cls, dict : dict):
        # for key in cls._parameter_constraints.keys():
        #     if not (key in dict.keys):
        #         raise KeyError(f"Cannot initialize model. Parameter {key} is missing")

        return cls(**dict)

    def forward(self, y, x):
        """
        Compute one single pass through decoder and encoder

        :param x: Conditioning variables corresponding to images/spectra
        :param y: Images/spectra
        :return: Mean returned by decoder, mean returned by encoder, log variance returned by encoder
        """

        # print("1 x={} y={}".format(x.shape, y.shape))
        y = torch.cat((y, x), dim=1)
        mean, logvar = self.encoder(y)
        # print(f"2 y={y.shape}")
        # re-parametrize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mean + eps * std
        # print(f"3 sample={sample.shape}")
        z = torch.cat((sample, x), dim=1)
        # print(f"4 cat(sample,x) -> z={z.shape}")
        mean_dec = self.decoder(z)
        # print(f"5 mean_dec={mean_dec.shape}")
        return (mean_dec, mean, logvar, z)

    def init_weights(self):
        """
            Initialize weight of recurrent layers
        """
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.decoder.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder of the cVAE for 2D images
    """

    def __init__(self, image_channels=1, hidden_dim=50, z_dim=10, c=7):
        """
        :param image_channels: Number of channels in the 2D images
        :param hidden_dim: Dimension of hidden layer
        :param z_dim: Dimension of latent space
        :param c: Dimension of conditioning variables
        """
        super().__init__()
        self.conv1 = nn.Conv2d(image_channels + c, hidden_dim, kernel_size=4, stride=2, padding=1) # Output: (hidden_dim, 64, 32)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1) # Output: (hidden_dim*2, 32, 16)
        self.fc_mu = nn.Linear(hidden_dim * 2 * 32 * 16, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 * 32 * 16, z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        mean = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return (mean, logvar)

class Decoder(nn.Module):
    """
    Decoder of cVAE for 2D images
    """

    def __init__(self, output_channels=1, hidden_dim=50, z_dim=10, c=7):
        super().__init__()
        self.fc = nn.Linear(z_dim + c, hidden_dim * 2 * 32 * 16)
        self.deconv1 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1) # Output: (hidden_dim, 64, 32)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_channels, kernel_size=4, stride=2, padding=1) # Output: (output_channels, 128, 64)

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = z.view(-1, 50*2, 32, 16) # Reshape to match the output shape of the encoder
        z = F.relu(self.deconv1(z))
        reconstruction = torch.sigmoid(self.deconv2(z))
        return reconstruction


class CVAE(nn.Module):
    """
    Base PyTorch cVAE class for 2D images
    """
    name = "CVAE"
    def __init__(self, hidden_dim=50, z_dim=10, c=7, init_weights=True, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(1, hidden_dim, z_dim, c)
        self.decoder = Decoder(1, hidden_dim, z_dim, c)

        if init_weights:
            self.init_weights()
    @classmethod
    def init_from_dict(cls, dict : dict):
        return cls(**dict)

    def forward(self, y, x):
        y = torch.cat((y, x), dim=1)
        mean, logvar = self.encoder(y)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mean + eps * std
        z = torch.cat((sample, x), dim=1)
        mean_dec = self.decoder(z)
        return (mean_dec, mean, logvar, z)

    def init_weights(self):
        """
            Initialize weight of recurrent layers
        """
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.decoder.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
