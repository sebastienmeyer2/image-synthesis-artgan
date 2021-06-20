"""
Gather the Generator and the Discriminator into the ArtGAN.

The purpose of this file is to combine the Generator and
the Discriminator part from their own files to build up
the ArtGAN. To do so, we just have to initialize every
parameter of the sub-networks. Then, we add methods to
ArtGAN that can train itself with specified
parameters, show or generate images and save loss values.
"""


# Importing Python packages
import os
import sys 
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
parentdir = os.path.dirname(path)
sys.path.insert(0, parentdir)
from typing import List
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing our own files and classes
from generator import Generator
from discriminator import Discriminator
import utils


class ArtGAN():
    """Implement the ArtGAN."""

    def __init__(self, data_type: str, version: str,
                 img_size: int, nb_classes: int, input_channels: int = 3,
                 start_channels: int = 110, alpha: float = 0.2,
                 retrain_epoch: int = None, device: torch.device = None):
        """Initialize the ArtGAN as described in the original paper.

        Args:
            data_type: the type of input data
            version: the version of this GAN
            img_size: the size of the input images, which
                are expected to be squared images of size
                img_size*img_size*input_channels where input_channels is
                used to initialize the Discriminator
            nb_classes: the number of classes for the input
                images classification, except the FAKE class, e.g. 10
                for CIFAR-10 dataset
            input_channels: the number of channels on the input
                images, e.g. 3 for RGB channels
            start_channels: the number of channels to
                build the Generator with, that is, the number of
                channels that are used for the dense data
            alpha: positive coefficient for the leakyReLU
                negative slope e.g. 0.2 as in the original paper
            retrain_epoch: if positive, the model will be loaded from
                the results folder
            device: the device to use everywhere, used in case of
                change for retraining

        Raises:
            ValueError if the size of the input images is less than 64 pixels
            due to the operations exectued by the generator deconv layers
        """
        # General parameters

        self.data_type = data_type
        self.version = version

        self.total_epochs = 0
        self.retrain = False

        if img_size < 64:
            raise ValueError("The size of the input images has to be of at least 64 pixels!")

        self.img_size = img_size
        self.nb_classes = nb_classes
        self.input_channels = input_channels
        self.start_channels = start_channels
        self.alpha = alpha

        # Modules

        self.D = Discriminator(img_size, nb_classes, input_channels, alpha)
        self.G = Generator(img_size, start_channels)

        # In case of retraining

        if retrain_epoch is not None:

            models_folder = "results/" + self.data_type + "_" + self.version + "/models/"
            path_to_model = models_folder + "gan_" + str(retrain_epoch) + ".pth"
            model_data = torch.load(path_to_model, map_location=device)

            # Tests for other parameters
            if self.img_size != model_data["img_size"]:
                raise ValueError("Not same image size! Saved {}.".format(model_data["img_size"]))
            if self.nb_classes != model_data["nb_classes"]:
                raise ValueError("Not same number of classes! Saved {}.".format(model_data["nb_classes"]))
            if self.input_channels != model_data["input_channels"]:
                raise ValueError("Not same number of channels for images! Saved {}.".format(model_data["input_channels"]))
            if self.start_channels != model_data["start_channels"]:
                raise ValueError("Not same number of start channels! Saved {}.".format(model_data["start_channels"]))
            if self.alpha != model_data["alpha"]:
                raise ValueError("Not same negative slope for LeakyReLU! Saved {}.".format(model_data["alpha"]))

            self.total_epochs = model_data["epoch"]
            self.G.load_state_dict(model_data["G"])
            self.D.load_state_dict(model_data["D"])

            self.retrain = True

    def cuda(self) -> None:
        """Enable CUDA on this class, since it is not a Module."""
        self.G.cuda()
        self.D.cuda()

        return

    def train(self) -> None:
        """Go into training mode."""
        self.G.train()
        self.D.train()

        return

    def eval(self) -> None:
        """Go into evaluation mode."""
        self.G.eval()
        self.D.eval()

        return

    def encode(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """Encode an input image into features.

        Args:
            img: a torch tensor containing images of
                dimension (N, C, H, W) where C = nb_classes, H = img_size & W = img_size
                from initialization

        Returns:
            self.D.encode(img): the result of encoding
        """
        return self.D.encode(img)

    def decode(self, latent: torch.FloatTensor) -> torch.FloatTensor:
        """Decode a vector of latent features.

        Args:
            latent: a torch tensor of dimension (N, C, H, W)
                where C = 512 from latent features

        Returns:
            self.G.decode(latent): resulting image via decoding
        """
        return self.G.decode(latent)

    def train_model(self, trainloader: torch.utils.data.DataLoader, device: torch.device,
                    epochs: int = 5, initial_lr: float = 1e-3, lr_ratio: float = 1.0, opt_decay: float = 0.9,
                    G_decrease_epoch: int = 80, G_decrease_rate: float = 10,
                    D_decrease_epoch: int = 80, D_decrease_rate: float = 10,
                    save_model_step: int = 1, save_image_step: int = 5, data_classes: List = None,
                    save_loss: bool = False, save_score: bool = False) -> pd.DataFrame:
        """Train the GAN using the algorithm provided in the original paper.

        Args:
            trainloader: the input data for training
            device: device to be used everywhere
            epochs: the number of epochs to train the model
            initial_lr: the learning rate to begin training with
            lr_ratio: the ratio between the initial learning rate of the
                discriminator and the initial learning rate of the
                generator
            opt_decay: the rate to use in the RMSProp optimizers
            G_decrease_epoch: the epoch when we want to decrease the
                learning rate of the generator
            G_decrease_rate: the decrease rate to use for each
                decrease epoch step for the generator
            D_decrease_epoch: the epoch when we want to decrease the
                learning rate of the discriminator
            D_decrease_rate: the decrease rate to use for each
                decrease epoch step for the discriminator
            save_model_step: after each save model step, the model will
                be saved in an appropriate folder
            save_image_step: after each save image step, the model will go
                in eval mode to generate an image per class and save them
                in an appropriate folder
            data_classes: the list of all classes in the input data
            save_loss: whether we have to save this new loss
                into the results folder
            save_score: whether we want to save the score for each epoch
                into the results folder

        Returns:
            loss_list: a DataFrame containing the mean loss for the Discriminator
                and the Generator after each epoch
        """
        self.train()

        # Initializing optimizers
        G_opt = torch.optim.RMSprop(self.G.parameters(), lr=initial_lr, alpha=opt_decay)
        D_opt = torch.optim.RMSprop(self.D.parameters(), lr=lr_ratio*initial_lr, alpha=opt_decay)

        if self.retrain:

            models_folder = "results/" + self.data_type + "_" + self.version + "/models/"
            path_to_model = models_folder + "gan_" + str(self.total_epochs) + ".pth"
            model_data = torch.load(path_to_model, map_location=device)

            G_opt.load_state_dict(model_data["G_opt"])
            D_opt.load_state_dict(model_data["D_opt"])

        # Update both learning rates
        new_G_lr = utils.decrease_lr(G_opt, self.total_epochs, G_decrease_epoch, G_decrease_rate)
        if new_G_lr != None:
            print("Learning rate of G has been set to {}.".format(new_G_lr))
            G_lr = new_G_lr

        new_D_lr = utils.decrease_lr(D_opt, self.total_epochs, D_decrease_epoch, D_decrease_rate)
        if new_D_lr != None:
            print("Learning rate of D has been set to {}.".format(new_D_lr))
            D_lr = new_D_lr

        # Initializing loss lists
        loss_list = []
        D_loss_value = 0
        D_loss_list = []
        D_loss_per_epoch = 0
        G_loss_value = 0
        G_loss_list = []
        G_loss_per_epoch = 0

        # Initializing loss functions
        prob_loss = torch.nn.BCELoss()
        mse_loss = torch.nn.MSELoss()

        # Initialiazing accuracy score
        batch_size = 0
        score = 0
        score_list = []

        G_lr = np.around(G_opt.param_groups[0]["lr"], 4)
        D_lr = np.around(D_opt.param_groups[0]["lr"], 4)

        epoch_pbar = tqdm(range(epochs), desc="Epoch: {}, lr: {} (G) {} (D), TN: {}%".format(self.total_epochs, G_lr, D_lr, score))
        for _ in epoch_pbar:

            batch_pbar = tqdm(trainloader, desc="Batch loss: {} (G) {} (D)".format(np.around(G_loss_value, 3), np.around(D_loss_value, 3)))
            for X_r, k in batch_pbar:

                # Setting discriminator's grad to zero
                D_opt.zero_grad()

                # Input images & labels
                X_r = X_r.to(device)
                k = k.to(device)
                batch_size = X_r.size(0)

                # Noise can help both networks to learn more
                X_noise = utils.gen_noise(batch_size, self.input_channels, self.img_size, self.total_epochs, device)
                X_r += X_noise

                # Initialization of samples for the generator
                Z_hat, Yk_hat = utils.fake_noise(batch_size, self.start_channels, self.nb_classes, device)
                k_hat = utils.fake_class(batch_size, self.nb_classes, device)
                Z_Yk_cat = torch.cat([Z_hat, Yk_hat], dim=1)

                # Making a one-hot vector from input labels
                k_hot = utils.class_to_prob(k, self.nb_classes, device)

                # Making a one-hot vector from FAKE class
                k_hat_hot = utils.class_to_prob(k_hat, self.nb_classes, device)

                # Construction of generated image and prediction for both real and fake ones
                Y = self.D(X_r)
                X_hat = self.G(Z_Yk_cat)
                Y_hat = self.D(X_hat)

                # Computing accuracy of the discriminator
                disc_class = torch.argmax(Y_hat, dim=1)
                for j in range(batch_size):
                    if disc_class[j] == self.nb_classes:
                        score += 1

                # Backpropagation for discriminator's parameters
                D_real_loss = prob_loss(Y, k_hot)
                D_fake_loss = prob_loss(Y_hat, k_hat_hot)
                D_loss = D_real_loss + D_fake_loss
                D_loss_value = D_loss.item()
                D_loss_list.append(D_loss_value)
                D_loss.backward(retain_graph=True)

                D_opt.step()

                # Setting generator's grad to zero
                G_opt.zero_grad()

                # Recalculation of prediction (modified parameters) and adv loss
                new_Y_hat = self.D(X_hat)
                k_fake_hot = torch.cat([Yk_hat, torch.zeros(batch_size, 1, device=device)], dim=1)
                G_loss_adv = prob_loss(new_Y_hat, k_fake_hot)

                # New encoding and decoding of image and L2 loss
                Z = self.encode(X_r)
                Xz_hat = self.decode(Z)
                G_loss_L2 = mse_loss(X_r, Xz_hat)

                # Backpropagation for generator's parameters
                G_loss = G_loss_adv + G_loss_L2
                G_loss.backward()
                G_loss_value = G_loss.item()
                G_loss_list.append(G_loss_value)

                G_opt.step()

                batch_pbar.set_description("Batch loss: {} (G) {} (D)".format(np.around(G_loss_value, 3), np.around(D_loss_value, 3)))

            # Updating net's own epoch counter
            self.total_epochs += 1

            # Updating both learning rates
            new_G_lr = utils.decrease_lr(G_opt, self.total_epochs, G_decrease_epoch, G_decrease_rate)
            if new_G_lr != None:
                print("Learning rate of G has been set to {}.".format(new_G_lr))
                G_lr = np.around(new_G_lr, 4)

            new_D_lr = utils.decrease_lr(D_opt, self.total_epochs, D_decrease_epoch, D_decrease_rate)
            if new_D_lr != None:
                print("Learning rate of D has been set to {}.".format(new_D_lr))
                D_lr = np.around(new_D_lr, 4)

            # Updating progression bar
            score = np.around(100*score/len(trainloader.dataset), 2)
            score_list.append([self.total_epochs, score])
            epoch_pbar.set_description("Epochs done: {}, lr: {} (G) {} (D), TN: {}%".format(self.total_epochs, G_lr, D_lr, score))
            score = 0

            # Saving loss of both networks
            D_loss_per_epoch = np.mean(D_loss_list)
            D_loss_list = []
            G_loss_per_epoch = np.mean(G_loss_list)
            G_loss_list = []
            loss_list.append([self.total_epochs, D_loss_per_epoch, G_loss_per_epoch])

            # Saving model
            if save_model_step > 0:
                if self.total_epochs%save_model_step == 0:

                    models_folder = "results/" + self.data_type + "_" + self.version + "/models/"
                    model_filename = models_folder + "gan_" + str(self.total_epochs) + ".pth"

                    if not os.path.exists(models_folder):
                        os.makedirs(models_folder)

                    model_data = {}
                    model_data["epoch"] = self.total_epochs
                    model_data["G"] = self.G.state_dict()
                    model_data["D"] = self.D.state_dict()
                    model_data["G_opt"] = G_opt.state_dict()
                    model_data["D_opt"] = D_opt.state_dict()
                    model_data["start_channels"] = self.start_channels
                    model_data["img_size"] = self.img_size
                    model_data["input_channels"] = self.input_channels
                    model_data["nb_classes"] = self.nb_classes
                    model_data["alpha"] = self.alpha

                    torch.save(model_data, model_filename)

            # Saving generated images
            if save_image_step > 0:
                if self.total_epochs%save_image_step == 0:

                    self.save_img(data_classes, device)

        loss_list = pd.DataFrame(loss_list, columns=["Epoch", "D_loss", "G_loss"])

        # Saving loss
        if save_loss:
            self.write_loss(loss_list)

        score_list = pd.DataFrame(score_list, columns=["Epoch", "Score"])

        # Saving accuracy of the discriminator
        if save_score:
            self.write_score(score_list)

        self.eval()

        return loss_list

    def show_img(self, data_classes: List, device: torch.device) -> None:
        """Show a collection of images for all classes.

        Args:
            data_classes: the classes contained in the input data
            device: the device to use everywhere
        """
        self.eval()

        Z_hat, Yk_hat = utils.fake_noise_all_classes(self.start_channels, self.nb_classes, device)
        Z_Yk_fixed = torch.cat([Z_hat, Yk_hat], dim=1)

        gen_imgs = self.G(Z_Yk_fixed)
        class_probs = self.D(gen_imgs)
        _, predicted_classes = torch.max(class_probs.data, 1)

        for i in range(self.nb_classes):
            gen_img = gen_imgs[i]
            pred = predicted_classes[i]

            gen_img = gen_img.cpu().detach().numpy()

            plt.imshow(np.transpose(gen_img, (1, 2, 0)))
            plt.axis("off")
            plt.title("Label: {} (G) {} (D)".format(data_classes[i], data_classes[pred]))

            plt.show()

        self.train()

        return

    def save_img(self, data_classes: List, device: torch.device) -> None:
        """Plot and save a collection of images for all classes.

        Args:
            data_classes: the classes contained in the input data
            device: the device to use everywhere
        """
        self.eval()

        Z_hat, Yk_hat = utils.fake_noise_all_classes(self.start_channels, self.nb_classes, device)
        Z_Yk_fixed = torch.cat([Z_hat, Yk_hat], dim=1)

        gen_imgs = self.G(Z_Yk_fixed)
        class_probs = self.D(gen_imgs)
        _, predicted_classes = torch.max(class_probs.data, 1)

        images_folder = "results/" + self.data_type + "_" + self.version + "/images/epoch_" + str(self.total_epochs) + "/"
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        for i in range(self.nb_classes):
            gen_img = gen_imgs[i]
            pred = predicted_classes[i]

            gen_img = gen_img.cpu().detach().numpy()

            plt.imshow(np.transpose(gen_img, (1, 2, 0)))
            plt.axis("off")
            plt.title("Label: {} (G) {} (D)".format(data_classes[i], data_classes[pred]))

            path_to_image = images_folder + "G_" + data_classes[i] + "_D_" + data_classes[pred] + ".jpg"
            plt.savefig(path_to_image)

        self.train()

        return

    def save_raw_img(self, data_classes: List, device: torch.device) -> None:
        """Save a collection of images for all classes as PIL images.

        Args:
            data_classes: the classes contained in the input data
            device: the device to use everywhere
        """
        pil_transform = transforms.ToPILImage()

        self.eval()

        Z_hat, Yk_hat = utils.fake_noise_all_classes(self.start_channels, self.nb_classes, device)
        Z_Yk_fixed = torch.cat([Z_hat, Yk_hat], dim=1)

        gen_imgs = self.G(Z_Yk_fixed)

        images_folder = "results/" + self.data_type + "_" + self.version + "/raw_images/"
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        for i in range(self.nb_classes):
            gen_img = gen_imgs[i]

            gen_img = pil_transform(gen_img)

            path_to_image = images_folder + data_classes[i] + "_" + str(self.total_epochs) + ".jpg"
            gen_img.save(path_to_image)

        self.train()

        return

    def write_loss(self: nn.Module, loss_list: pd.DataFrame) -> None:
        """Write the content of loss_list into a csv file for future reading.

        It will append the data if the model is being retrained.

        Args:
            loss_list: a dataframe containing the total epochs counter, the
                loss for Generator and the loss for Discriminator
        """
        losses_folder = "results/" + self.data_type + "_" + self.version + "/losses/"
        if not os.path.exists(losses_folder):
            os.makedirs(losses_folder)
        path_to_loss = losses_folder + "loss.csv"

        if self.retrain == False:
            loss_list.to_csv(path_to_loss, columns=loss_list.columns, header=True, index=False)
        else:
            loss_data = pd.read_csv(path_to_loss, header=0)
            loss_list = pd.concat([loss_data, loss_list], ignore_index=True)
            loss_list.to_csv(path_to_loss, columns=loss_list.columns, header=True, index=False)

        return

    def write_score(self: nn.Module, score_list: pd.DataFrame) -> None:
        """Write the content of score_list into a csv file for future reading.

        It will append the data if the model is being retrained.

        Args:
            score_list: a dataframe containing the total epochs counter, and the
                score for Discriminator
        """
        scores_folder = "results/" + self.data_type + "_" + self.version + "/scores/"
        if not os.path.exists(scores_folder):
            os.makedirs(scores_folder)
        path_to_score = scores_folder + "score.csv"

        if self.retrain == False:
            score_list.to_csv(path_to_score, columns=score_list.columns, header=True, index=False)
        else:
            score_data = pd.read_csv(path_to_score, header=0)
            score_list = pd.concat([score_data, score_list], ignore_index=True)
            score_list.to_csv(path_to_score, columns=score_list.columns, header=True, index=False)

        return
        