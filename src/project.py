"""
Evaluate ArtGAN on different epochs.

The purpose of this file is to implement functions
using trained models and utility functions in order
to create visualizations of the process at different
steps
"""


# Importing Python packages
import os
from typing import List
import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
font = {"size": 18}
mpl.rc("font", **font)
from mpl_toolkits.axes_grid1 import ImageGrid

# Importing our own files and classes
from gan.artgan import ArtGAN
import utils


def artgan_evolution(starting_epoch: int, ending_epoch: int, step_epoch: int,
                     device: torch.device, use_cuda: bool,
                     data_type:str, version: str, data_classes: List,
                     start_channels: int = 110, img_size: int = 64,
                     input_channels: int = 3, nb_classes: int = 10,
                     alpha: float = 0.2) -> None:
    """Plot the evolution of the ArtGAN.

    Images are generated from same initial noise for
    every model saved during the chosen epochs.

    Args:
        starting_epoch: first epoch to generate image
        ending_epoch: last epoch to generate image
        step_epoch: step between each model in number of
            epochs
        device: the device to use everywhere
        use_cuda: whether we are using CUDA globally
        data_type: the name of the dataset
        version: the version of the GANs
        data_classes: a list containing the name of the classes
        start_channels: the number of channels for the
            Generator part
        img_size: the size of the input images, expected to
            be squared images
        input_channels: the number of channels in the input
            images
        nb_classes: the number of classes in the dataset,
            except FAKE class e.g. 10 for CIFAR-10
        alpha: the negative slope for LeakyReLU activation
    """
    Z_hat, Yk_hat = utils.fake_noise_all_classes(start_channels, nb_classes, device)
    img_labels = torch.argmax(Yk_hat, dim=1)
    Z_Yk_hat = torch.cat([Z_hat, Yk_hat], dim=1)

    nb_models = 1+(ending_epoch-starting_epoch)//step_epoch

    T = list(range(starting_epoch, ending_epoch+1, step_epoch))

    generated_imgs = []

    for i in range(nb_models):

        step_artgan = ArtGAN(data_type, version, img_size, nb_classes,
                             input_channels=input_channels, start_channels=start_channels,
                             alpha=alpha, retrain_epoch=T[i], device=device)

        if use_cuda:
            step_artgan.cuda()
        step_artgan.eval()

        model_imgs = step_artgan.G(Z_Yk_hat)
        model_imgs = model_imgs.cpu().detach().numpy()
        model_imgs = np.transpose(model_imgs, (0, 2, 3, 1))
        generated_imgs.append(model_imgs)

    for j in range(nb_classes):

        print("\r\033[KSaving {}...".format(data_classes[img_labels[j]]), end="", flush=True)

        nrows = 1+nb_models//5
        ncols = 4
        fig = plt.figure(figsize=(4*ncols, 4*nrows))
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=(0.05, 0.33))

        row = 0
        col = 0
        for ax, i in zip(grid, range(nb_models)):

            ax.imshow(generated_imgs[i][j])
            ax.set_title("Epoch {}".format(T[i]))
            ax.axis("off")

            col += 1
            if T[i]%4 == 0:
                row += 1
                col = 0

        # fig.suptitle("Generation of {} at different epochs".format(data_classes[img_labels[j]]))

        eval_folder = "results/" + data_type + "_" + version + "/evol/"
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)
        path_to_file = eval_folder + "evolution_{}.png".format(data_classes[img_labels[j]])
        plt.tight_layout()
        plt.savefig(path_to_file)
        plt.close()

    print("Done.")

def save_loss(data_type:str, version: str) -> None:
    """Save the loss evolution for the given ArtGAN.

    Args:
        data_type: the input dataset
        version: the version of the GAN
    """
    loss_folder = "results/" + data_type + "_" + version + "/losses/"
    path_to_loss = loss_folder + "loss.csv"

    loss_data = pd.read_csv(path_to_loss)

    T = loss_data["Epoch"]
    G_loss = loss_data["G_loss"]
    D_loss = loss_data["D_loss"]

    plt.plot(T, G_loss, label="Loss (G)", color="blue")
    plt.plot(T, D_loss, label="Loss (D)", color="red")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title("Loss of {} version on {} dataset".format(version, data_type))

    path_to_file = loss_folder + "loss.jpg"
    plt.tight_layout()
    plt.savefig(path_to_file)
    plt.close()

def save_score(data_type:str, version: str) -> None:
    """Save the score evolution for the given ArtGAN.

    Args:
        data_type: the input dataset
        version: the version of the GAN
    """
    score_folder = "results/" + data_type + "_" + version + "/scores/"
    path_to_score = score_folder + "score.csv"

    score_data = pd.read_csv(path_to_score)

    T = score_data["Epoch"]
    D_score = score_data["Score"]

    plt.plot(T, D_score, label="Specificity", color="green")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Specificity (in %)")
    # plt.title("Score of {} version on {} dataset".format(version, data_type))

    path_to_file = score_folder + "score.jpg"
    plt.tight_layout()
    plt.savefig(path_to_file)
    plt.close()

def save_both(data_type:str, version: str) -> None:
    """Save loss and score evolution for the given ArtGAN.

    Args:
        data_type: the input dataset
        version: the version of the GAN
    """
    # Score data
    score_folder = "results/" + data_type + "_" + version + "/scores/"
    path_to_score = score_folder + "score.csv"

    score_data = pd.read_csv(path_to_score)

    T = score_data["Epoch"]
    D_score = score_data["Score"]

    # Loss data
    loss_folder = "results/" + data_type + "_" + version + "/losses/"
    path_to_loss = loss_folder + "loss.csv"

    loss_data = pd.read_csv(path_to_loss)

    T = loss_data["Epoch"]
    G_loss = loss_data["G_loss"]
    D_loss = loss_data["D_loss"]

    # Plotting both
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].plot(T, G_loss, label="Loss (G)", color="blue")
    ax[0].plot(T, D_loss, label="Loss (D)", color="red")
    ax[0].tick_params(axis="y")
    ax[0].legend()

    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Specificity (in %)")
    ax[1].plot(T, D_score, label="Specificity", color="green")
    ax[1].tick_params(axis="y")
    ax[1].legend()

    both_folder = "results/" + data_type + "_" + version + "/both/"
    if not os.path.exists(both_folder):
        os.makedirs(both_folder)

    path_to_file = both_folder + "loss_score.jpg"
    plt.tight_layout()
    plt.savefig(path_to_file)
    plt.close()


if __name__ == "__main__":

    # Command lines
    parser = argparse.ArgumentParser(description="A file to evaluate your GANs!")
    available_datasets = ["cifar", "artist", "genre", "style"]
    parser.add_argument("data_type", type=str, choices=available_datasets, help="Please choose a dataset from those supported.")
    parser.add_argument("version", type=str, help="Please choose the version of your GAN.")

    # Sub command lines for mode
    subparsers = parser.add_subparsers(dest="subcommands", description="Please choose a mode of evaluation.")
    parser_evol = subparsers.add_parser("evolution")
    parser_evol.add_argument("start", type=int, help="Please choose the epoch to start from.")
    parser_evol.add_argument("stop", type=int, help="Please choose the epoch at which the program ends.")
    parser_evol.add_argument("step", type=int, help="Please choose the step between models.")

    parser_all = subparsers.add_parser("all")
    parser_all.add_argument("start", type=int, help="Please choose the epoch to start from.")
    parser_all.add_argument("stop", type=int, help="Please choose the epoch at which the program ends.")
    parser_all.add_argument("step", type=int, help="Please choose the step between models.")

    parser_score = subparsers.add_parser("score")

    parser_loss = subparsers.add_parser("loss")

    parser_both = subparsers.add_parser("both")

    # Args selecton
    args = parser.parse_args()

    data_type = args.data_type
    version = args.version
    mode = args.subcommands

    if mode == "evolution" or mode == "all":
        starting_epoch = args.start
        ending_epoch = args.stop
        step_epoch = args.step

    # Turning on CUDA globally
    USE_CUDA = torch.cuda.is_available()
    print("Will we use CUDA? {}".format(USE_CUDA))
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    if data_type == "cifar":

        with open("src/datasets/cifar.json", "r") as f:
            CIFAR10_CLASSES = json.load(f)
        data_classes = CIFAR10_CLASSES

    elif data_type in {"artist", "genre", "style"}:

        with open("src/datasets/wikiart.json", "r") as f:
            WIKIART_CLASSES = json.load(f)
        data_classes = WIKIART_CLASSES[data_type]

    else:

        raise ValueError("This dataset is not supported!")

    # ArtGAN parameters
    nb_classes = len(data_classes)
    start_channels = 100 + nb_classes
    img_size = 64
    input_channels = 3
    alpha = 0.2

    # Plotting evolution
    if mode == "all" or mode == "evolution":
        artgan_evolution(starting_epoch=starting_epoch, ending_epoch=ending_epoch, step_epoch=step_epoch,
                         device=DEVICE, use_cuda=USE_CUDA,
                         data_type=data_type, version=version, data_classes=data_classes,
                         start_channels=start_channels, img_size=img_size,
                         input_channels=input_channels, nb_classes=nb_classes, alpha=alpha)

    # Plotting loss
    if mode == "all" or mode == "loss":
        save_loss(data_type, version)

    # Plotting score
    if mode == "all" or mode == "score":
        save_score(data_type, version)

    # Plotting both score & loss
    if mode == "all" or mode == "both":
        save_both(data_type, version)
