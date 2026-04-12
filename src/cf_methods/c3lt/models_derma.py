import torch.nn as nn


class Generator(nn.Module):
    """
    Generator for DermaMNIST (64x64, 3-channel RGB images).
    
    Changes from MNIST version:
    - nc=3 (RGB instead of grayscale)
    - Added one more ConvTranspose2d layer to reach 64x64 output
    - Adjusted kernel/stride/padding to produce exact 64x64 spatial dims
    - ngf=64 retained (sufficient for 64x64 RGB)
    """
    def __init__(self, latent_dim, ngpu=1, nc=3, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z (latent_dim x 1 x 1)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 4 x 4
 
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 8
 
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 16
 
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 32 x 32
 
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # output size: (nc) x 64 x 64
        )
 
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
 
 
class Discriminator(nn.Module):
    """
    Discriminator for DermaMNIST (64x64, 3-channel RGB images).
    
    Changes from MNIST version:
    - nc=3 (RGB instead of grayscale)
    - Added one more Conv2d layer to handle 64x64 → 1x1 spatial reduction
    - This is the standard DCGAN discriminator for 64x64 images
    """
    def __init__(self, ngpu=1, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 32 x 32
 
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 16 x 16
 
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 8 x 8
 
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 4 x 4
 
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # output size: 1 x 1 x 1
        )
 
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# ---- Autoencoder variants (updated for 64x64 RGB) ----
 
class Encoder(nn.Module):
    def __init__(self, latent_dim, ngpu=1, nc=3, ndf=16):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, latent_dim, 4, 1, 0, bias=False),
            # (latent_dim) x 1 x 1   ← was 4,2,1 giving 2x2; now 4,1,0 giving 1x1
        )

 
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
 
 
class Decoder(nn.Module):
    def __init__(self, latent_dim, ngpu=1, nc=3, ndf=16):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input: (latent_dim) x 2 x 2
            nn.ConvTranspose2d(latent_dim, ndf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4
 
            nn.ConvTranspose2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8
 
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
 
            nn.ConvTranspose2d(ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32
 
            nn.ConvTranspose2d(ndf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # (nc) x 64 x 64
        )
 
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
 
 
class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_dim, nc=3, ndf=16, ngpu=1):
        super(ConvolutionalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.nc = nc
        self.ndf = ndf
        self.ngpu = ngpu
        self.encoder = Encoder(self.latent_dim, nc=self.nc, ndf=self.ndf)
        self.decoder = Decoder(self.latent_dim, nc=self.nc, ndf=self.ndf)
 
    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output