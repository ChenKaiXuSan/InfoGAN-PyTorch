# %%
'''
pure dcgan structure.
code similar sample from the pytorch code.
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
import torch
import torch.nn as nn

import numpy as np
# %%
class Generator(nn.Module):
    '''
    pure Generator structure

    '''    
    def __init__(self, image_size=64, z_dim=100, conv_dim=64, channels = 1, n_classes = 10, code_dim = 2):
        
        super(Generator, self).__init__()
        self.imsize = image_size
        self.channels = channels

        self.z_dim = z_dim
        self.n_classes = n_classes
        self.code_dim = code_dim

        input_dim = self.z_dim + self.n_classes + self.code_dim

        repeat_num = int(np.log2(self.imsize)) - 3  # 3
        mult = 2 ** repeat_num  # 8

        self.l1 = nn.Sequential(
            # input is Z, going into a convolution.
            nn.ConvTranspose2d(input_dim, conv_dim * mult, 4, 1, 0, bias=False),
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(True)
        )

        curr_dim = conv_dim * mult

        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )

        curr_dim = int(curr_dim / 2)

        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True),
        )

        curr_dim = int(curr_dim / 2)

        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )
        
        curr_dim = int(curr_dim / 2)
        
        self.last = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels, code):
        gen_input = torch.cat((z, labels, code), -1)

        gen_input = gen_input.view(z.size(0), -1, 1, 1)

        out = self.l1(gen_input)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)

        out = self.last(out)

        return out

# %%
class Discriminator(nn.Module):
    '''
    pure discriminator structure

    '''
    def __init__(self,  image_size = 64, conv_dim = 64, channels = 1, n_classes = 10, code_dim = 2):
        super(Discriminator, self).__init__()
        self.imsize = image_size

        self.channels = channels
        self.n_classes = n_classes
        self.code_dim = code_dim

        self.l1 = nn.Sequential(
            nn.Conv2d(self.channels, conv_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = conv_dim

        self.l2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2

        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        curr_dim = curr_dim * 2

        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2
        
        # output layers

        self.last_adv = nn.Sequential(
            nn.Conv2d(curr_dim, 1, 4, 1, 0, bias=False),
            # without sigmoid, used in the loss funciton
            )
        self.last_aux = nn.Sequential(
            nn.Conv2d(curr_dim, self.n_classes, 4, 1, 0, bias=False),
            nn.Softmax(dim=1)
        )
        self.latent_layer = nn.Sequential(
            nn.Conv2d(curr_dim, self.code_dim, 4,  1, 0, bias=False)
        )

    def forward(self, x):
        out = self.l1(x) # (*, 64, 32, 32)
        out = self.l2(out) # (*, 128, 16, 16)
        out = self.l3(out) # (*, 256, 8, 8)
        out = self.l4(out) # (*, 512, 4, 4)
        
        validity = self.last_adv(out) # (*, 1, 1, 1)
        label = self.last_aux(out) # (*, 10, 1, 1)
        latent_code = self.latent_layer(out) # (*, code_dim, 1, 1)

        return validity.squeeze(), label.squeeze(), latent_code.squeeze()