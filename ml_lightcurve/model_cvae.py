import torch
import torch.nn as nn

class CVAE(nn.Module):
    """
        Base pytorch cVAE class
    """

    _parameter_constraints: dict = {
        "image_size": ["int"],
        "hidden_dim": ["int"],
        "z_dim": ["int"],
        "c":["int"]
    }

    def __init__(self, image_size=150, hidden_dim=50, z_dim=10, c=7):
        """
        :param image_size: Size of 1D "images" of data set i.e. spectrum size
        :param hidden_dim: Dimension of hidden layer
        :param z_dim: Dimension of latent space (latent_units)
        :param c: Dimension of conditioning variables
        """
        super(CVAE, self).__init__()
        self.z_dim = z_dim
        self.c = c
        self.image_size=image_size
        self.hidden_dim=hidden_dim
        self.encoder = Encoder(image_size, hidden_dim, z_dim, c) # self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(image_size, hidden_dim, z_dim, c) # self.decoder = Decoder(latent_dims)
        self.init_weights()

    @classmethod
    def init_from_dict(cls, dict : dict):
        for key in cls._parameter_constraints.keys():
            if not (key in dict.keys):
                raise KeyError(f"Cannot initialize model. Parameter {key} is missing")

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


class Encoder(nn.Module):
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


class Decoder(nn.Module):
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
