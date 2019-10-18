import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAEForDigits(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.lin_enc_mean = nn.Linear(64*7*7, 2)
        self.lin_enc_logvar = nn.Linear(64*7*7, 2)
        
        self.lin_dec = nn.Linear(2, 64*7*7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        shape = x.size()
        x = x.reshape(shape[0], -1)
        mean = self.lin_enc_mean(x)
        logvar = self.lin_enc_logvar(x)
        return mean, logvar
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, x):
        x = self.lin_dec(x)
        x = x.reshape(-1, 64, 7, 7)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample(mean, logvar)
        return self.decode(z), mean, logvar

class VAEForFaces(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        self.lin_enc_mean = nn.Linear(64*8*8, 200)
        self.lin_enc_logvar = nn.Linear(64*8*8, 200)
        
        self.lin_dec = nn.Linear(200, 64*8*8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        shape = x.size()
        x = x.reshape(shape[0], -1)
        mean = self.lin_enc_mean(x)
        logvar = self.lin_enc_logvar(x)
        return mean, logvar
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, x):
        x = self.lin_dec(x)
        x = x.reshape(-1, 64, 8, 8)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample(mean, logvar)
        return self.decode(z), mean, logvar

def conv_out_shape(h, w, conv_layer):
    return tuple([
        (x + 2*conv_layer.padding[i] - conv_layer.dilation[i]*(conv_layer.kernel_size[i] - 1) - 1)/conv_layer.stride[i] + 1
        for x,i in zip((h,w), range(2))
    ])

def conv_t_out_shape(h, w, conv_t_layer):
    return tuple([
        (x - 1)*conv_t_layer.stride[i] - 2*conv_t_layer.padding[i] + conv_t_layer.dilation[i]*(conv_t_layer.kernel_size[i] - 1) + conv_t_layer.output_padding[i] + 1
        for x,i in zip((h,w), range(2))
    ])

class VAE(nn.Module):
    def __init__(
            self,
            input_dim,
            encoder_conv_filters,
            encoder_conv_kernel_sizes,
            encoder_conv_strides,
            decoder_conv_t_filters,
            decoder_conv_t_kernel_sizes,
            decoder_conv_t_strides,
            z_dim,
            use_batch_norm=False,
            use_dropout=False
        ):
        super().__init__()

        in_channels, h, w = input_dim

        encoder = []
        for i in range(len(encoder_conv_filters)):
            conv = nn.Conv2d(in_channels, encoder_conv_filters[i], encoder_conv_kernel_sizes[i], stride=encoder_conv_strides[i], padding=1)
            in_channels = encoder_conv_filters[i]
            h, w = conv_out_shape(h, w, conv)
            encoder.append(conv)
            if use_batch_norm:
                bn = nn.BatchNorm2d(encoder_conv_filters[i])
                encoder.append(bn)
            encoder.append(nn.LeakyReLU())
            if use_dropout:
                encoder.append(nn.Dropout())
                
        self.encoder = nn.Sequential(encoder)

        flattened_dim = encoder_conv_filters[i]*h*w

        self.lin_enc_mean = nn.Linear(flattened_dim, z_dim)
        self.lin_enc_logvar = nn.Linear(flattened_dim, z_dim)
        self.lin_dec = nn.Linear(z_dim, flattened_dim)

        decoder = []
        for i in range(len(decoder_conv_t_filters)):
            conv_t = nn.ConvTranspose2d(in_channels, decoder_conv_t_filters[i], decoder_conv_t_kernel_sizes[i], stride=decoder_conv_t_strides[i], padding=1, output_padding=decoder_conv_t_strides[i]-1)
            in_channels = decoder_conv_t_filters[i]
            h, w = conv_t_out_shape(h, w, conv_t)
            decoder.append(conv_t)
            if i < len(decoder_conv_t_filters)-1:
                if use_batch_norm:
                    bn = nn.BatchNorm2d(decoder_conv_t_filters[i])
                    decoder.append(bn)
                decoder.append(nn.LeakyReLU())
                if use_dropout:
                    decoder.append(nn.Dropout())
            else:
                decoder.append(nn.Sigmoid())
    
    def encode(self, x):
        x = self.encoder(x)
        shape = x.size()
        x = x.reshape(shape[0], -1)
        mean = self.lin_enc_mean(x)
        logvar = self.lin_enc_logvar(x)
        return mean, logvar
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, x):
        x = self.lin_dec(x)
        x = x.reshape(-1, 64, 7, 7)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample(mean, logvar)
        return self.decode(z), mean, logvar