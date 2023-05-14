import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        super(Generator, self).__init__()
        self.main = nn.Sequential(

            # Z : nz x 1 x 1
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Mish(True),

            # (ngf * 8) x 2 x 2
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.Mish(True),

            # (ngf * 4) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.Mish(True),

            # (ngf * 2) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.Mish(True),

            # ngf x 16 x 16
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input.view(-1, self.nz, 1, 1))
        return output

class Discriminator(nn.Module):
    def __init__(self, nz, ndf, nc):
        self.nz = nz
        self.ndf = ndf
        self.nc = nc
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_generator(nz, ngf, nc):
    netG = Generator(nz=nz, ngf=ngf, nc=nc)
    netG.apply(weights_init)
    return netG

def get_discriminator(nz, ndf, nc):
    netD = Discriminator(nz=nz, ndf=ndf, nc=nc)
    netD.apply(weights_init)
    return netD