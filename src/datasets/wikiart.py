"""
Implement the Wikiart dataset.

The purpose of this file is to implement a class
reading and organizing data from the Wikiart
datasets. For instance, it enables the use of style, genre
and artist datasets.
"""


# Importing Python packages
import os
import sys 
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
from typing import Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms
import pandas as pd
from PIL import Image


class Wikiart(Dataset):
    """Implement any Wikiart dataset."""

    def __init__(self, data_type: str, transform: torchvision.transforms = None,
                 train: bool = True):
        """Initialize a Dataset containing images from Wikiart.

        Args:
            data_type: the type of label (style - artist - genre)
            transform: a transformation to apply to every image
            train: whether we want the training or testing data
        """
        self.data_type = data_type

        selection = "src/datasets/" + data_type + ("_train.csv" if train else "_val.csv")

        self.img_folder = "resized_data/wikiart/" # images have to be downloaded & resized prior
        self.data = pd.read_csv(selection, delimiter=",", names=["img_path", "label"])

        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the Dataset.

        Returns:
            len(self.data)
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, int]:
        """Get images and labels.

        Args:
            idx: the index of the image in the dataset

        Returns:
            image: the image after transformation
            label: the label after transformation
        """
        img_path = self.data.loc[idx, "img_path"]
        label = self.data.loc[idx, "label"]
        image = Image.open(self.img_folder + img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def type(self) -> str:
        """Return the type of data contained in this Dataset.

        Returns:
            type: the type of  data
        """
        return self.data_type
