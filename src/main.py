"""
The purpose of this file is to train our networks.

This is the file you want to run from root folder with
specified dataset and options.
"""


# Importing Python packages
import argparse
import json
import torch
import torchvision
import torchvision.transforms as transforms

# Importing our own files and classes
from gan.artgan import ArtGAN
from datasets.wikiart import Wikiart


if __name__ == "__main__":

    # Command lines
    parser = argparse.ArgumentParser(description="Main file to train and evaluate ArtGAN.")
    parser.add_argument("data_type", type=str, help="Please choose a dataset from those supported.")
    parser.add_argument("-v", "--version", type=str, help="Please choose a version for saving results. Default: temp.")
    parser.add_argument("-d", "--duration", type=int, help="You can choose a number of epochs for training. Default: 0.")
    parser.add_argument("-r", "--retrain", type=int, help="Choose an epoch from which you want to continue training. Default: None.")
    parser.add_argument("-l", "--loss", type=int, help="Type any number if you want to save loss in a file. Default: False.")
    parser.add_argument("-s", "--score", type=int, help="Type any number if you want to save score in a file. Default: False.")
    args = parser.parse_args()

    data_type = args.data_type
    version = args.version if args.version else "temp"
    training_epochs = args.duration if args.duration else 0
    retrain_epoch = args.retrain if args.retrain else None
    save_loss = bool(args.loss)
    save_score = bool(args.score)

    # Turning on CUDA globally
    USE_CUDA = torch.cuda.is_available()
    print("Will we use CUDA? {}".format(USE_CUDA))
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    # Global parameters for our GANs
    batch_size = 128
    img_size = 64
    save_model_step = 1

    if data_type == "cifar":

        with open("src/datasets/cifar.json", "r") as f:
            CIFAR10_CLASSES = json.load(f)

        transform = transforms.Compose([
                                        transforms.Resize(64),
                                        transforms.ToTensor(),
                                        ])

        trainset = torchvision.datasets.CIFAR10(root="data/", train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        data_classes = CIFAR10_CLASSES
        nb_classes = len(data_classes)
        data_classes.append("FAKE")

    elif data_type in {"artist", "genre", "style"}:

        with open("src/datasets/wikiart.json", "r") as f:
            WIKIART_CLASSES = json.load(f)

        # Data is expected to be resized prior, in order to accelerate training
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ])

        trainset = Wikiart(data_type, transform, train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        data_classes = WIKIART_CLASSES[data_type]
        nb_classes = len(data_classes)
        data_classes.append("FAKE")

    else:

        raise ValueError("This dataset is not supported!")

    # Modified GAN parameters (optional, but can yield better results)
    start_channels = 100 + nb_classes

    # GAN initialization
    artgan = ArtGAN(data_type, version, img_size, nb_classes,
                    start_channels=start_channels, retrain_epoch=retrain_epoch,
                    device=DEVICE)
    if USE_CUDA:
        artgan.cuda()

    # Modified training parameters (optional, but can yield better results)
    initial_lr = 2e-4
    lr_ratio = 0.5
    G_decrease_epoch = 50
    G_decrease_rate = 5
    D_decrease_epoch = 50
    D_decrease_rate = 5

    # Training
    if training_epochs > 0:
        loss_list = artgan.train_model(trainloader, DEVICE,
                                       epochs=training_epochs, initial_lr=initial_lr, lr_ratio=lr_ratio,
                                       G_decrease_epoch=G_decrease_epoch, G_decrease_rate=G_decrease_rate,
                                       D_decrease_epoch=D_decrease_epoch, D_decrease_rate=D_decrease_rate,
                                       data_classes=data_classes, save_loss=save_loss, save_score=save_score,
                                       save_model_step=save_model_step)

    # Printing images
    artgan.show_img(data_classes, DEVICE)
