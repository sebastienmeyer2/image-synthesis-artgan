"""
Implement the Discriminator part of the ArtGAN.

The purpose of this file is to properly define the discriminative part
of the ArtGAN, as described in the paper. To do so, we separate the Enc
and the clsNet which are then combined to make the Discriminator.
"""


# Importing Python packages
import os
import sys 
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
import torch
import torch.nn as nn


class Enc(nn.Module):
    """The first part of the Discriminator, namely the Encoder."""

    def __init__(self, input_channels: int = 3, alpha: float = 0.2):
        """Initialize the Encoder par of the Discriminator.

        It takes the input image or a fake image
        and runs several convolutional layers on it in order to
        create features that can be sent to the following classifier

        Args:
            input_channels: the number of channels in the input data
                e.g. 3 for RGB channels
            alpha: positive coefficient to be used as the parameter
                for leakyReLU activation e.g. 0.2 as in the original paper
        """
        super(Enc, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv1lrelu = nn.LeakyReLU(negative_slope=alpha, inplace=True)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2bn = nn.BatchNorm2d(num_features=128, affine=True)
        self.conv2lrelu = nn.LeakyReLU(negative_slope=alpha, inplace=True)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv3bn = nn.BatchNorm2d(num_features=256, affine=True)
        self.conv3lrelu = nn.LeakyReLU(negative_slope=alpha, inplace=True)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv4bn = nn.BatchNorm2d(num_features=512, affine=True)
        self.conv4lrelu = nn.LeakyReLU(negative_slope=alpha, inplace=True)

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the forward propagation on input data.

        Args:
            image: a torch tensor of dimension (N, C, H, W) where
                C is equal to input_channels from initialization e.g. 3 for RGB

        Returns:
            enc_img: the result of the propagation
        """
        enc_img = self.conv1(image)
        enc_img = self.conv1lrelu(enc_img)

        enc_img = self.conv2(enc_img)
        enc_img = self.conv2bn(enc_img)
        enc_img = self.conv2lrelu(enc_img)

        enc_img = self.conv3(enc_img)
        enc_img = self.conv3bn(enc_img)
        enc_img = self.conv3lrelu(enc_img)

        enc_img = self.conv4(enc_img)
        enc_img = self.conv4bn(enc_img)
        enc_img = self.conv4lrelu(enc_img)

        return enc_img


class clsNet(nn.Module):
    """Implement the clsNet which will classify the data."""

    def __init__(self, img_size: int, nb_classes: int, alpha: float = 0.2):
        """Initialize the clsNet.

        This is a classifier that takes the transformed data from Encoder
        and tries to find the class of the input image, which is
        in a set of K classes plus a FAKE class

        Args:
            img_size: the size of the image, which is supposed to be
                a square image of size img_size*img_size*input_channels
            nb_classes: the number of classes for the classifier
                to choose from excepting FAKE class e.g. 10 for CIFAR-10
            alpha: positive coefficient for the leakyReLU
                activation for convolutional layer(s) e.g. 0.2 as in the
                original paper
        """
        super(clsNet, self).__init__()

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)
        self.conv5bn = nn.BatchNorm2d(num_features=1024, affine=True)
        self.conv5lrelu = nn.LeakyReLU(negative_slope=alpha, inplace=True)

        self.fc6 = nn.Linear(in_features=(img_size//16)*(img_size//16)*1024, out_features=nb_classes+1)
        self.fc6sig = nn.Sigmoid()

    def forward(self, enc_img: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the forward propagation of this network on input data.

        Args:
            enc_img: a torch tensor of dimensions (N, C, H, W)
                where C = 512

        Returns:
            clas: the predicted classes for this data
        """
        clas = self.conv5(enc_img)
        clas = self.conv5bn(clas)
        clas = self.conv5lrelu(clas)

        clas = clas.view(clas.size(0), -1)
        clas = self.fc6sig(self.fc6(clas))

        return clas


class Discriminator(nn.Module):
    """Gather the two parts of the Discriminator into one class."""

    def __init__(self, img_size: int, nb_classes: int, input_channels: int = 3, alpha: float = 0.2):
        """Initialize the Discriminator part of the GAN.

        It is meant to treat an input image and classify it whether in
        a FAKE class or in its best class.

        Args:
            img_size: the size of the input image, which is expected to
                be a squared image of size img_size*img_size*input_channels
            nb_classes: the number of classes in the input images,
                except FAKE class e.g. 10 for CIFAR-10
            input_channels: the number of channels in the input images,
                e.g. 3 for RGB channels
            alpha: positive coefficient for the slope in the
                leakyReLU activation e.g. 0.2 as in the original paper
        """
        super(Discriminator, self).__init__()

        self.enc = Enc(input_channels=input_channels, alpha=alpha)
        self.clsnet = clsNet(img_size, nb_classes, alpha=alpha)

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the forward propagation on input images.

        Args:
            img: a torch tensor containing images of dimension
                (N, C, H, W) where C = nb_classes, H = img_size & W = img_size from initialization
                parameters

        Returns:
            Y: the predicted classes for the image
        """
        Y = self.enc(img)
        Y = self.clsnet(Y)

        return Y

    def encode(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """Encode an input image into features.

        Args:
            img: a torch tensor containing images of
                dimension (N, C, H, W) where C = nb_classes, H = img_size & W = img_size
                from initialization

        Returns:
            enc_img: the result of encoding
        """
        enc_img = self.enc(img)

        return enc_img
        