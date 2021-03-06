"""
Evaluate the ArtGAN using k-NN search.

This is a file you want to run from root folder with
specified dataset and options.
"""


# Importing Python packages
import os
from typing import List
import argparse
import json
import heapq
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Importing our own files and classes
from gan.artgan import ArtGAN
from datasets.wikiart import Wikiart


def knnShow(k : int, n : int, starting_epoch: int, ending_epoch: int,  step_epoch: int,
            device: torch.device, use_cuda: bool,
            data_type:str, version: str, data_classes: List,
            start_channels: int, img_size: int,
            nb_classes: int, trainset: Dataset) -> None:
    """Plot the k-NN search for the given period of time.

    Args:
        k: the number of neighbors
        n: the number of images per class
        starting_epoch: the epoch to start with
        ending_epoch: the epoch at which the algorithm ends
        step_epoch: the step between lines
        device: the device to use everywhere
        use_cuda: whether we want to use cuda here
        data_type: the name of the dataset
        version: the name of the ArtGAN version
        data_classes: a list of the names of the classes
            in the dataset
        start_channels: the number of starting channels of
            the Generator
        img_size: the size of the input images
        nb_classes: the number of classes in the dataset
        trainset: the dataset itself
    """
    Z_hat = torch.randn(n*nb_classes, start_channels-nb_classes, device=device)
    Yk_hat = torch.zeros(n*nb_classes, nb_classes, device=device)
    for i in range(nb_classes):
        for j in range(n):
            Yk_hat[i*n +j][i] = 1
    Z_Yk_fixed = torch.cat([Z_hat, Yk_hat], dim=1)
    img_labels = torch.argmax(Yk_hat, dim=1)
    T = [i for i in range(starting_epoch, ending_epoch+1, step_epoch)]

    images = []

    epoch_pbar = tqdm(T, desc="Epoch: {}".format(T[0]))
    for epoch in epoch_pbar:
        epoch_pbar.set_description("Epoch: {}".format(epoch))
        epochImages = []

        step_artgan = ArtGAN(data_type, version, img_size, nb_classes,
                            start_channels=start_channels, retrain_epoch=epoch,
                            device=device)

        if use_cuda: step_artgan.cuda()
        step_artgan.eval()

        model_imgs = step_artgan.G(Z_Yk_fixed)
        model_imgs = model_imgs.cpu().detach()

        nn_pbar = tqdm(range(nb_classes), desc="Label: {}".format(data_classes[0]))
        for i in nn_pbar:
            nn_pbar.set_description("Label: {}".format(data_classes[i]))
            for j in range(n):
                gen_img = model_imgs[i*n+j]
                neighbors = knn(k , gen_img, trainset)
                neighbors = [trainset.__getitem__(idx)[0].numpy() for idx in neighbors]
                gen_img = np.transpose(gen_img, (1, 2, 0))
                neighbors = [ np.transpose(img, (1, 2, 0)) for img in neighbors]
                l = [gen_img] + neighbors
                epochImages.append(l)

        images.append(epochImages)

    for j in range(n*nb_classes):
        ncols = k+1
        nrows = len(T)

        fig = plt.figure(figsize=(4, 4))
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=(0.05, 0.3))

        row = 0
        col = 0
        for ax in grid:

            ax.imshow(images[row][j][col])
            ax.axis("off")
            if col == 0:
                ax.set_title("Epoch {}".format(T[row]))

            col += 1
            col = col%ncols
            if col == 0: row += 1

        # fig.suptitle("Generation of {} at different epochs".format(data_classes[img_labels[j]]))
        plt.tight_layout()
        eval_folder = "results/" + data_type + "_" + version + "/knn/"
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)
        path_to_file = eval_folder + "neighbor_evo_{}.png".format(data_classes[img_labels[j]])
        plt.savefig(path_to_file)
        plt.close()

def knn(k: int, genImg: torch.FloatTensor, trainData: Dataset) -> List[torch.FloatTensor]:
    """Compute k-NN search on given images.

    Args:
        k: the number of neighbors
        genImg: the images generated by the Generator
        trainData: the dataset itself

    Returns:
        res: the nearest neighbors of specified images
    """
    lib = {}
    h = []
    for i in range(trainData.__len__()):
        ele, _ = trainData.__getitem__(i)

        v = torch.sqrt(torch.sum( torch.square(ele -genImg) ))
        lib[v] = i
        heapq.heappush(h,v)
    resVal = [heapq.heappop(h) for i in range(k)]
    res = [lib[val] for val in resVal]

    return res


if __name__ == "__main__":

    # Command lines
    parser = argparse.ArgumentParser(description="Main file to train and evaluate ArtGAN.")
    parser.add_argument("data_type", type=str, help="Please choose a dataset from those supported.")
    parser.add_argument("-v", "--version", type=str, help="Please choose a version for saving results. Default: temp.")
    parser.add_argument("-a", "--startepoch", type=int, help="Please choose a starting epoch. Default: 5.")
    parser.add_argument("-e", "--endepoch", type=int, help="Please choose an ending epoch. Default: 5.")
    parser.add_argument("-s", "--stepepoch", type=int, help="Type the amount of epochs to step between lines. Default: 1.")
    parser.add_argument("-n", "--n", type=int, help="Type the amount of images to generate for each class of images. Default: 1.")
    parser.add_argument("-k", "--kneighbors", type=int, help="Type the amount of nearest neighbours you wish to find. Default: 1.")
    # parser.add_argument("-d", "--distance", type=int, help="Type any number if you want to save distance in a file. Default: False.")
    args = parser.parse_args()

    data_type = args.data_type
    version = args.version if args.version else "temp"
    startEpoch = args.startepoch if args.startepoch else 5
    endEpoch = args.endepoch if args.endepoch else 5
    stepEpoch = args.stepepoch if args.stepepoch else 1
    n = args.n if args.n else 1
    k = args.kneighbors if args.kneighbors else 1
    # save_distance = True if args.distance else False


    # Turning on CUDA globally
    USE_CUDA = torch.cuda.is_available()
    print("Will we use CUDA? {}".format(USE_CUDA))
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    # Global parameters for our GANs
    batch_size = 128
    img_size = 64

    if data_type == "cifar":

        with open("src/datasets/cifar.json", "r") as f:
            CIFAR10_CLASSES = json.load(f)

        transform = transforms.Compose([
                                        transforms.Resize(64),
                                        transforms.ToTensor(),
                                        ])

        trainset = torchvision.datasets.CIFAR10(root="data/", train=True,
                                                download=True, transform=transform)

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

        data_classes = WIKIART_CLASSES[data_type]
        nb_classes = len(data_classes)
        data_classes.append("FAKE")

    else:

        raise ValueError("This dataset is not supported!")

    # GAN parameters
    start_channels = 100 + nb_classes

    knnShow(k,n,startEpoch,endEpoch, stepEpoch,DEVICE,USE_CUDA,data_type, version, data_classes,start_channels,img_size,nb_classes, trainset)
