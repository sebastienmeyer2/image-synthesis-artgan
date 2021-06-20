"""
Convert a raw data folder into a resized one.

The purpose of this file is to resize the image
before using them, so much so that we avoid resizing
them at each iteration
"""


# Importing Python packages
import os
import glob
import argparse
from PIL import Image


def resize_images(imgs_path: str, img_size: int = 64,
                  rec: bool = True, filetype: str = "jpg"):
    """Resize and save all the images contained in the specified folder.

    Args:
        imgs_path: the path to the folder containing images
        img_size: the desired output image size
        rec: if you want to perform recursivity
        filetype: the format of the images

    Raises:
        ValueError if it does not find any images in the
            specified folder
    """
    # Get all the images
    images = []

    if rec:

        file_list = glob.glob(imgs_path + "*") # recursivity

        for class_path in file_list:
            for img_path in glob.glob(class_path + "/*.{}".format(filetype)):
                images.append(img_path)

    else:

        for img_path in glob.glob(imgs_path + "/*.{}".format(filetype)):
            images.append(img_path)

    if len(images) == 0:

        raise ValueError("Your folder seems to be empty, please check selected options.")

    # Save new ones, resized
    for img_path in images:

        img = Image.open(img_path)
        img = img.convert("RGB") # 3 channels as input

        split_img_path = img_path.split("/")
        split_img_path[0] = "resized_" + split_img_path[0]
        new_img_path = "/".join(split_img_path)

        img_dir = "/".join(new_img_path.split("\\")[:-1])
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        img = img.resize((img_size, img_size))
        img = img.save(new_img_path)


if __name__ == "__main__":

    # Command lines
    parser = argparse.ArgumentParser(description="This file enables to resize images to accelerate training.")
    parser.add_argument("folder", type=str, help="Please enter the name of your folder from data/.")
    parser.add_argument("-r", "--recursivity", type=int, help="Enter a number if your data folder DO NOT contain subfolders. Default: contains subfolders.")
    parser.add_argument("-i", "--img_size", type=int, help="Please enter the selected size for your images. Default: 64.")
    parser.add_argument("-f", "--format", type=str, help="Indicate the format of your images. Default: jpg.")
    args = parser.parse_args()

    img_folder = "data/" + args.folder
    recursivity = False if args.recursivity else True
    img_size = args.img_size if args.img_size else 64
    filetype = args.format if args.format else "jpg"

    # Resizing images
    print("Resizing images...")
    resize_images(img_folder, img_size, recursivity, filetype)
    print("Done.")
