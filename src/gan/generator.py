"""
Implement the Generator part of the ArtGAN.

The purpose of this file is to properly define the generative part
of the ArtGAN as described in the paper. To do so, we separate the
zNet and the Dec, which are then combined to make the Generator class.
"""


# Importing Python packages
import os
import sys 
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
import torch
import torch.nn as nn


class zNet(nn.Module):
    """Implement the module converting dense features to latent features."""

    def __init__(self, img_size: int, start_channels: int = 110):
        """Initialize the zNet network.

        This is the first part of the Generator. Hence, its task is
        to convert the dense code into a latent code to be used in the
        Decoder

        Args:
            img_size: the size of the output images to be created
                at the end of the Generator, which are expected to be
                squared images of size img_size*img_size*input_channels where
                input_channels is from the Discriminator initialization (not
                this one)
            start_channels: the number of channels in
                the input data

        Raises:
            ValueError if the size of image is less than 64 due to
            the operations executed by the deconv layers
        """
        super(zNet, self).__init__()

        # Parameters

        if img_size < 64:
            raise ValueError("The size of the input images has to be of at least 64 pixels!")

        self.start_channels = start_channels
        self.img_size = img_size

        # Deconvolution layers

        self.deconv1 = nn.ConvTranspose2d(in_channels=start_channels, out_channels=1024, kernel_size=4, stride=1, padding=0)
        self.deconv1bn = nn.BatchNorm2d(num_features=1024, affine=True)
        self.deconv1relu = nn.ReLU(inplace=False)

        self.deconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.deconv2bn = nn.BatchNorm2d(num_features=512, affine=True)
        self.deconv2relu = nn.ReLU(inplace=False)

    def forward(self, dense: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the propagation of this network on input data.

        Args:
            dense: a torch tensor of dimension
                (N, C, H, W)

        Returns:
            latent: the result of propagation with this
                network
        """
        # We need to ensure that the data has correct size, for this we used the formula to
        # calculate the result of convolutions (reversed)
        latent = dense.view(dense.size(0), self.start_channels, (self.img_size//16)-3, (self.img_size//16)-3)
        latent = self.deconv1(latent)
        latent = self.deconv1bn(latent)
        latent = self.deconv1relu(latent)

        latent = self.deconv2(latent)
        latent = self.deconv2bn(latent)
        latent = self.deconv2relu(latent)

        return latent


class Dec(nn.Module):
    """Implement the Decoder part of the Generator."""

    def __init__(self):
        """Initialize the Decoder (or Dec).

        This is the second part of the Generator. Hence, its task is
        to work with the latent code and generate a fake image to be sent
        to the Discriminator
        """
        super(Dec, self).__init__()

        self.deconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv3bn = nn.BatchNorm2d(num_features=256, affine=True)
        self.deconv3dropout = nn.Dropout(p=0.5)
        self.deconv3relu = nn.ReLU(inplace=False)

        self.deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv4bn = nn.BatchNorm2d(num_features=128, affine=True)
        self.deconv4dropout = nn.Dropout(p=0.5)
        self.deconv4relu = nn.ReLU(inplace=False)

        self.deconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv5bn = nn.BatchNorm2d(num_features=128, affine=True)
        self.deconv5relu = nn.ReLU(inplace=False)

        self.deconv6 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.deconv6sig = nn.Sigmoid()

    def forward(self, latent: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the forward propagation of this network on input data.

        Args:
            latent: a torch tensor of dimension (N, C, H, W)
                to be treated

        Returns:
            dec_img: the result of the propagation with this network
        """
        dec_img = self.deconv3(latent)
        dec_img = self.deconv3bn(dec_img)
        dec_img = self.deconv3dropout(dec_img)
        dec_img = self.deconv3relu(dec_img)

        dec_img = self.deconv4(dec_img)
        dec_img = self.deconv4bn(dec_img)
        dec_img = self.deconv4dropout(dec_img)
        dec_img = self.deconv4relu(dec_img)

        dec_img = self.deconv5(dec_img)
        dec_img = self.deconv5bn(dec_img)
        dec_img = self.deconv5relu(dec_img)

        dec_img = self.deconv6(dec_img)
        dec_img = self.deconv6sig(dec_img)

        return dec_img


class Generator(nn.Module):
    """Gather the two parts of the Generator in one class."""

    def __init__(self, img_size: int, start_channels: int = 110):
        """Initialize the complete generator part of the GAN.

        Args:
            img_size: the size of the output images to be built
                and sent to the Discriminator, which are expected to be
                squared images of size img_size*img_size*input_channels
                where input_channels is used for the Discriminator's
                initialization
            start_channels: the number of channels in the
                input images, free to choose

        Raises:
            ValueError if the value of img_size is less than 64 pixels due
            to the operations executed by the deconv layers
        """
        if img_size < 64:
            raise ValueError("The size of the input images has to be of at least 64 pixels!")

        super(Generator, self).__init__()

        self.znet = zNet(img_size, start_channels=start_channels)
        self.dec = Dec()

    def forward(self, dense: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the forward propagation on input data.

        Args:
            dense: a torch tensor of dimension
                (N, C, H, W) where C = input_channels from the
                initialization

        Returns:
            X_hat: the result of both propagations
        """
        gen_img = self.znet(dense)
        gen_img = self.dec(gen_img)

        return gen_img

    def decode(self, latent: torch.FloatTensor) -> torch.FloatTensor:
        """Decode a vector of latent features.

        Args:
            latent: a torch tensor of dimension (N, C, H, W)
                where C = 512 from latent features

        Returns:
            dec_img: resulting image via decoding
        """
        dec_img = self.dec(latent)

        return dec_img
        