from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import os.path as osp

plt.rcParams.update({"figure.max_open_warning": 0})

from early_stopping import EarlyStopping
from net import Autoencoder
from sklearn.model_selection import train_test_split

CUDA = torch.cuda.is_available()
print(f"IS CUDA AVAILABLE: {CUDA}")


class AECTraining:
    """Train autoencoder net, use transformation of data if defined"""

    def __init__(
        self,
        ds_aec,
        io_dim=None,
        ignore_first=18,
        epochs=300,
        verbose=False,
        transform=None,
        learning_rate=0.0001,
        output_path="./",
    ):
        self.__net = None
        self.__verbose = verbose
        self.__ds_aec = ds_aec
        self.__epochs = epochs
        self.ignore_first = ignore_first
        self.__io_dim = io_dim
        self.__cuda = False
        self.__transform = transform
        self.__learning_rate = learning_rate
        self.output_path = output_path

    def cuda(self):
        self.__cuda = True

    def fit(self, output_file):
        batch_size = 128

        X_train, X_val = train_test_split(self.__ds_aec, test_size=0.2, shuffle=True)

        dataset_train = AECDataset(
            X_train, ignore_first=self.ignore_first, transform=self.__transform
        )
        dataset_val = AECDataset(
            X_val, ignore_first=self.ignore_first, transform=self.__transform
        )

        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        # initialize the early_stopping object
        early_stopping = EarlyStopping(
            initial_patience=100,
            patience=10,
            verbose=True,
            path=osp.join(self.output_path, "aec_checkpoint.pt"),
        )

        self.__net = Autoencoder(self.__io_dim)
        if CUDA:
            self.__net = self.__net.cuda()

        criterion = BCELoss()
        optimizer = torch.optim.Adam(self.__net.parameters(), lr=self.__learning_rate)
        losses = []
        losses_val = []
        for epoch in range(self.__epochs):
            flag = True
            loss_accum = 0
            self.__net.train()
            for data in dataloader_train:
                if CUDA:
                    data = data.cuda()
                # data = data
                # ===================forward=====================
                output = self.__net(data)
                if flag:
                    flag = False
                loss = criterion(output, data)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lost_item = loss.item()
                loss_accum += lost_item
            # ===================log========================
            loss_accum /= len(dataloader_train)
            losses.append(loss_accum)

            self.__net.eval()
            with torch.no_grad():
                loss_accum_val = 0
                for data_val in dataloader_val:
                    if CUDA:
                        data_val = data_val.cuda()
                    y_pred = self.__net(data_val)
                    loss_val = criterion(y_pred, data_val)
                    loss_accum_val += loss_val.item()
                loss_accum_val /= len(dataloader_val)
                losses_val.append(loss_accum_val)

            print(
                "epoch [{}/{}], loss:{:.7f}".format(
                    epoch + 1, self.__epochs, loss.detach().item()
                )
            )
            print(
                "epoch [{}/{}], loss_val:{:.7f}".format(
                    epoch + 1, self.__epochs, loss_val.detach().item()
                )
            )
            if epoch > 4 and epoch % 4 == 0:
                self.plot_history({"loss": losses[4:], "loss_val": losses_val[4:]})
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            best_model = early_stopping(np.average(losses), self.__net)
            if best_model is not None:
                self.__best_model = best_model

            plt.plot()
            if early_stopping.early_stop:
                print("Early stopping the training.")
                break

        print("Saving AEC on: " + output_file)
        torch.save(self.__best_model.state_dict(), output_file)

        # import pickle
        # file_to_store = open(osp.join(output_file, "prediction_aec.pickle"), "wb")
        # pickle.dump({"loss": losses, "loss_val": losses_val}, file_to_store)
        # file_to_store.close()
        return self.__net

    def plot_history(self, history):
        plt.clf()
        plt.figure(figsize=(12, 8))

        plt.plot(history["loss"])
        plt.plot(history["loss_val"])
        plt.savefig(osp.join(self.output_path, "aec_histories_metrics_val.png"))
        plt.cla()


class AECDataset(Dataset):
    """Autoencoder treen nodes dataset."""

    def __init__(self, data, ignore_first=18, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform = transform
        self.ignore_first = ignore_first

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        if self.transform:
            row = self.transform(row)
        row = torch.from_numpy(row[self.ignore_first :]).to(torch.float32)
        return row


######################################################################
