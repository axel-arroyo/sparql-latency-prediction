# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import logging

import gc

import numpy as np
import torch
import torch.optim
import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os.path as osp

from .model_base import BaseRegression


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
# file logger
fh = logging.FileHandler("../output.log", mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
# console logger
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)


plt.rcParams.update({"figure.max_open_warning": 0})

from early_stopping import EarlyStopping
from net import NeoNet
from sklearn.metrics import mean_squared_error, mean_absolute_error

CUDA = torch.cuda.is_available()

print(f"IS CUDA AVAILABLE: {CUDA}")

###################################################################


class NeoRegression(BaseRegression):
    def __init__(self, aec=None, **kvargs):

        super().__init__(**kvargs)

    def fit(self, X, X_query, y, X_val, X_val_query, y_val):
        if isinstance(y, list):
            y = np.array(y)

        X, X_query, y = self.json_loads(X, X_query, y)
        X = [self.fix_tree(x) for x in X]
        print("X_train loaded")

        X_val, X_val_query, y_val = self.json_loads(X_val, X_val_query, y_val)
        X_val = [self.fix_tree(x) for x in X_val]
        print("X_val loaded")

        self.n = len(X)
        max_y = np.max(y)

        # Fit target transformer
        self.pipeline.fit_transform(y.reshape(-1, 1))

        print("Transforming Trees")
        X = self.tree_transform.transform(X)
        X_val = self.tree_transform.transform(X_val)

        # determine the initial number of channels
        io_dim = len(self.get_pred())

        pairs = list(zip(list(zip(X, X_query)), y))
        pairs_val = list(zip(list(zip(X_val, X_val_query)), y_val))

        if self.maxcardinality == 0:
            # Case for no cardinalities encoded data
            dataset = DataLoader(
                pairs,
                batch_size=64,
                num_workers=0,
                shuffle=True,
                collate_fn=self.collate,
            )
            dataset_val = DataLoader(
                pairs_val,
                batch_size=64,
                num_workers=0,
                shuffle=True,
                collate_fn=self.collate,
            )
        else:
            dataset = DataLoader(
                pairs,
                batch_size=64,
                num_workers=0,
                shuffle=True,
                collate_fn=self.collate_with_card,
            )
            dataset_val = DataLoader(
                pairs_val,
                batch_size=64,
                num_workers=0,
                shuffle=True,
                collate_fn=self.collate_with_card,
            )

        self.query_input_size = len(X_query[0])
        if self.maxcardinality != 0:
            # Case when cardinalities are added. We need to extend queries features to queries features+ len(pred2index) - 1
            self.query_input_size = self.query_input_size + io_dim - 1

        self.log("Initial input channels of tree model:", self.in_channels)
        self.net = NeoNet(
            io_dim,
            self.query_input_size,
            self.query_hidden_inputs,
            self.query_output,
            tree_units=self.tree_units,
            tree_units_dense=self.tree_units_dense,
            activation_tree=self.tree_activation_tree,
            activation_dense=self.tree_activation_dense,
            in_cuda=CUDA,
        )
        if CUDA:
            self.net = self.net.cuda()

        # initialize the early_stopping object
        early_stopping = EarlyStopping(
            initial_patience=self.early_stop_initial_patience,
            patience=self.early_stop_patience,
            verbose=True,
            path=osp.join(self.output_path, "../checkpoint.pt"),
        )

        if self.optimizer["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                self.net.parameters(), **self.optimizer["args"]
            )
        elif self.optimizer["optimizer"] == "Adagrad":
            optimizer = torch.optim.Adagrad(
                self.net.parameters(), **self.optimizer["args"]
            )
        else:
            optimizer = torch.optim.SGD(self.net.parameters(), **self.optimizer["args"])

        loss_fn = torch.nn.MSELoss()

        losses = []

        assert np.mean(y_val) > 5, "y_val must be in real scale"
        print("Max epochs to run:", self.epochs)
        for epoch in range(self.epochs):
            self.net.train()
            loss_accum = 0
            results_train = []
            for (x, y_train) in dataset:
                y_train_scaled = torch.tensor(
                    self.pipeline.transform(y_train.reshape(-1, 1)).astype(np.float32)
                )
                if CUDA:
                    y_train_scaled = y_train_scaled.cuda()
                y_pred = self.net(x)
                loss = loss_fn(y_pred, y_train_scaled)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lost_item = loss.item()
                loss_accum += lost_item

                results_train.extend(
                    list(
                        zip(
                            self.pipeline.inverse_transform(
                                y_pred.cpu().detach().numpy()
                            ),
                            y_train,
                        )
                    )
                )

            loss_accum /= len(dataset)
            losses.append(loss_accum)

            print(
                "{} Epoch {}, Training loss {}".format(
                    datetime.datetime.now(), epoch, loss_accum / len(dataset)
                )
            )

            # Prediction in subsample of train
            torch.cuda.empty_cache()

            y_pred_train, y_real_train = zip(*results_train)
            msetrain = mean_squared_error(y_real_train, y_pred_train)
            maetrain = mean_absolute_error(y_real_train, y_pred_train)
            rmsetrain = np.sqrt(msetrain)
            self.history["mse_by_epoch"].append(msetrain)
            self.history["rmse_by_epoch"].append(rmsetrain)
            self.history["mae_by_epoch"].append(maetrain)

            # Testing the model

            results_val = self.predict(dataset_val)
            y_pred_val, y_real_val = zip(*results_val)
            mseval = mean_squared_error(y_real_val, y_pred_val)
            maeval = mean_absolute_error(y_real_val, y_pred_val)
            rmseval = np.sqrt(mseval)
            self.history["mse_val_by_epoch"].append(mseval)
            self.history["rmse_val_by_epoch"].append(rmseval)
            self.history["mae_val_by_epoch"].append(maeval)
            #             print(f"RMSE in TRAIN: {rmsetrain} : RMSE in VAL: {rmseval}")
            logger.info(
                "==> Epoch {},\tTRAIN_LOSS: {}\t_TRAIN_RMSE: {},\tVAL_LOSS: {},\tVAL_RMSE: {}".format(
                    epoch, msetrain, rmsetrain, mseval, rmseval
                )
            )
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            best_model = early_stopping(
                np.average(
                    self.history["rmse_val_by_epoch"][-self.early_stop_patience :]
                ),
                self.net,
            )
            if best_model is not None:
                self.best_model = best_model
            if early_stopping.early_stop:
                print("Early stopping the training.")
                break

            if epoch:  # % 4 == 0:
                self.scatter_plot_history(
                    y_pred_train,
                    y_real_train,
                    y_pred_val,
                    y_real_val,
                    "Scatter real latency vs prediction on: ",
                    osp.join(
                        self.output_path,
                        "neo_with_aec_scatter_train_val_epoch_"
                        + "{:03d}".format(epoch),
                    ),
                    self.history,
                    max_reference=int(max_y + 10),
                    figsize=self.figimage_size,
                    title_all=f"Scatter and history, RMSE Train: {rmsetrain}, RMSE VAL: {rmseval}, Epoch: {epoch}",
                    start_history_from_epoch=self.start_history_from_epoch,
                )
            gc.collect()

    def index2sparse(self, tree, sizeindexes):
        resp = []
        for el in tree:
            if type(el[0]) == tuple:
                resp.append(self.index2sparse(el, sizeindexes))
            else:
                a = np.array(el)
                b = np.zeros((a.size, sizeindexes))
                b[np.arange(a.size), a] = 1
                onehot = np.sum(b, axis=0, keepdims=True)[0]
                resp.append(onehot)
        return tuple(resp)

    def index2sparse2(self, tree, sizeindexes):
        resp = []
        for el in tree:
            if type(el[0]) == tuple:
                resp.append(self.index2sparse2(el, sizeindexes))
            else:
                a = np.array(el)
                b = np.zeros((a.size, sizeindexes))
                b[np.arange(a.size), a] = 1
                onehot = np.sum(b, axis=0, keepdims=True)[0]
                resp.append(onehot)
        return tuple(resp)
