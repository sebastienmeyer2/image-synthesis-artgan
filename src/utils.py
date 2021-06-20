"""
Gather multiple functions to enhance the readibility of the code.

The majority of the functions are generating noise to train the ArtGAN,
as well as transforming vectors.
Some functions allow to work with images.
"""


# Importing Python packages
import os
import sys 
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
from typing import Tuple
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def gen_noise(batch_size: int, input_channels: int, img_size: int,
              epoch: int, device: torch.device) -> torch.FloatTensor:
    """Generate gaussian noise to be added to input data.

    Args:
        batch_size: the size of batches
        input_channels: the number of channels in the input images
            e.g. 3 for RGB
        img_size: the size of the input images, expected to be
            squared images
        epoch: the epoch of training
        device: the device to use everywhere

    Returns:
        noise: a tensor of normal noise to add to images
    """
    size = (batch_size, input_channels, img_size, img_size)
    noise = torch.normal(0.0, 0.1/(epoch+1), size, device=device)

    return noise

def fake_noise(batch_size: int, start_channels: int, nb_classes: int,
               device: torch.device) -> Tuple[torch.FloatTensor]:
    """Generate sample noise for the Generator.

    Generate a sample of data from a noise distribution (a
    Gaussian distribution) and a choice of class for the fake
    image construction for the Generator.

    Args:
        batch_size: the number of images per batch
        start_channels: the number of channels for the
            first deconvolutional layer of the Generator, e.g.
            100
        nb_classes: the number of classes of the dataset,
            excluding the FAKE class, e.g. 10 for CIFAR-10
        device: the device to be used to put the
            data on

    Returns:
        Z_hat: a tensor of size (N, C) where N is
            the batch_size and C the start_channels
        Yk_hat: a tensor of size (N, C) where N is
            the batch_size and C the nb_classes, where a random
            class is set to 1
    """
    Z_hat = torch.randn(batch_size, start_channels-nb_classes, device=device)

    Yk_hat = torch.zeros(batch_size, nb_classes, device=device)
    for j in range(batch_size):
        random_class = random.randint(0, nb_classes-1)
        Yk_hat[j][random_class] = 1

    return Z_hat, Yk_hat

def fake_class(batch_size: int, nb_classes: int, device: torch.device) -> torch.LongTensor:
    """Create the vector encoding FAKE attribute.

    Creates a vector of given length where the class for
    every image is the FAKE attribute.

    Args:
        batch_size: the number of images in the fake data
        nb_classes: the number of classes in the real data,
            excluding FAKE class
        device: the device to put everywhere

    Returns:
        fake_vector: a tensor of size (N) where N is the batch
            size and each value is the number of classes, ie the
            FAKE class
    """
    fake_vector = torch.full((batch_size,), nb_classes, device=device)
    fake_vector = fake_vector.float()

    return fake_vector

def prob_to_class(prob_vector: torch.FloatTensor, device: torch.device) -> torch.LongTensor:
    """Convert one-hot vector to a vector of probabilities.

    Convert a vector containing probabilities per each class
    to a new vector having for each value the class with highest
    probability.

    Args:
        prob_vector: a tensor of size (N, C) where N
            is the batch size and C is the number of classes
        device: the device to put the data on

    Returns:
        class_vector: a tensor of size (N, 1) where
            N is the batch size and the value contained is the class
            with highest probability in prob_vector

    Raises:
        ValueError if the dimension of the tensor is incorrect
    """
    if len(prob_vector.size()) != 2:
        raise ValueError("This function needs a 2-dimensional tensor!")

    batch_size = prob_vector.size(0)
    class_vector = torch.zeros(batch_size, dtype=torch.long, device=device)

    for j in range(batch_size):
        class_vector[j] = torch.argmax(prob_vector[j])

    return class_vector

def class_to_prob(class_vector: torch.LongTensor, nb_classes: int,
                  device: torch.device) -> torch.FloatTensor:
    """Convert a vector containing classes to a one-hot vector.

    Convert a vector containing unique value for a class attribute to
    a one-hot vector containing probability, zero everywhere except at the
    class index.

    Args:
        class_vector: a tensor of dimension (N, 1) where N is the batch
            size and the value is the class
        nb_classes: the number of classes in the input images, excepting the
            FAKE class
        device: the device to use everywhere

    Returns:
        prob_vector: a vector of probabilities where 1 is set to the former
            class attribute of each vector

    Raises:
        ValueError if the dimension of the input tensor is larger than 2
    """
    if len(class_vector.size()) > 2:
        raise ValueError("The vector of classes must be of dimension less than 3!")

    batch_size = class_vector.size(0)

    prob_vector = torch.zeros((batch_size, nb_classes+1), device=device)
    for i in range(batch_size):
        if len(class_vector.size()) == 1:
            prob_vector[i][class_vector[i].long()] = 1
        else:
            prob_vector[i][class_vector[i][0].long()] = 1

    return prob_vector

def decrease_lr(optimizer: torch.optim, current_epoch: int, lr_decay_epoch: int = 80,
                lr_decay_rate: float = 10) -> float:
    """Help managing the learning rate of RMSProp optimizers.

    The learning rate of the given optimizer is modified after each
    step of epochs to enhance learning.

    Args:
        optimizer: the optimizer from PyTorch
        current_epoch: the current epoch of the training process
        lr_decay_epoch: step of the decreasing
        lr_decay_rate: the rate to divide the learning rate with,
            after each step of epochs

    Returns:
        new_lr: the new learning rate for the optimizer
    """
    new_lr = None
    if current_epoch > 0 and (current_epoch%lr_decay_epoch) == 0:

        for param_group in optimizer.param_groups:
            param_group["lr"] /= lr_decay_rate
            new_lr = param_group["lr"]

    return new_lr

def fake_noise_one_class(start_channels: int, nb_classes: int, chosen_class: int,
                         device: torch.device) -> torch.FloatTensor:
    """Generate an input vector for the Generator with specified class.

    Args:
        start_channels: the number of channels in the input data for
            the generator
        nb_classes: the number of classes in the input images
        chosen_class: the index of the chosen class
        device: the device to use everywhere

    Returns:
        Z_Yk_fixed: a tensor of noise and the chosen class index

    Raises:
        ValueError if the chosen class is greater than the number of
            classes
    """
    if chosen_class >= nb_classes:
        raise ValueError("The chosen class is greater than the number of classes!")

    Z_hat = torch.randn(1, start_channels-nb_classes, device=device)

    Yk_hat = torch.zeros(1, nb_classes, device=device)
    Yk_hat[0][chosen_class] = 1

    Z_Yk_hat = torch.cat([Z_hat, Yk_hat], dim=1)

    return Z_Yk_hat

def fake_noise_all_classes(start_channels: int, nb_classes: int,
                           device: torch.device) -> torch.FloatTensor:
    """Generate an input vector for the Generator with all classes.

    Args:
        start_channels: the number of channels in the input data for
            the generator
        nb_classes: the number of classes in the input images
        device: the device to use everywhere

    Returns:
        Z_hat: a tensor of noise
        Yk_hat: the corresponding labels
    """
    Z_hat = torch.randn(nb_classes, start_channels-nb_classes, device=device)

    Yk_hat = torch.zeros(nb_classes, nb_classes, device=device)
    for i in range(nb_classes):
        Yk_hat[i][i] = 1

    return Z_hat, Yk_hat

def imshow(img: np.ndarray, epoch: int, label: str) -> None:
    """Plot and save an image from Tensor.

    Args:
        img: the image to plot as a tensor
        epoch: the epoch to display in the title
        label: the label/class to display in the title
    """
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.title("Epoch {} and class {}".format(epoch, label))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/images/gan_{}_class_{}".format(epoch, label))
    plt.show()

def concat_h(img1: Image, img2: Image) -> Image:
    """Concatenate two PIL images horizontally.

    Args:
        img1: the first image
        img2: the second image

    Returns:
        dst: the concatenated image

    Raises:
        ValueError if the images do not share same
        modes or heights
    """
    assert img1.height == img2.height
    assert img1.mode == img2.mode

    img_mode = img1.mode
    dst = Image.new(img_mode, (img1.width + img2.width, img1.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))

    return dst

def concat_v(img1: Image, img2: Image) -> Image:
    """Concatenate two PIL images vertically.

    Args:
        img1: the first image
        img2: the second image

    Returns:
        dst: the concatenated image

    Raises:
        ValueError if the images do not share same
        modes or widths
    """
    assert img1.width == img2.width
    assert img1.mode == img2.mode

    img_mode = img1.mode
    dst = Image.new(img_mode, (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))

    return dst
    