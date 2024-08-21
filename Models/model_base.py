# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import logging
import os.path as osp
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import joblib
import os
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
# file logger
fh = logging.FileHandler("./output.log", mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
# console logger
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)


plt.rcParams.update({"figure.max_open_warning": 0})

from featurize import SPARQLTreeFeaturizer
from net import NeoNet

CUDA = torch.cuda.is_available()

print(f"IS CUDA AVAILABLE: {CUDA}")


def _nn_path(base):
    return os.path.join(base, "nn_weights")


def _x_transform_path(base):
    return os.path.join(base, "x_transform")


def _y_transform_path(base):
    return os.path.join(base, "y_transform")


def _channels_path(base):
    return os.path.join(base, "channels")


def _n_path(base):
    return os.path.join(base, "n")


# General Methods
def scatter_image(y_pred, y_test, title, name, max_reference=300, figsize=None):
    plt.clf()
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.scatter(y_pred, y_test)
    plt.plot(range(max_reference))
    plt.xlabel("Prediction")
    plt.ylabel("Real latency")
    plt.savefig(name + ".png")
    plt.clf()


def plot_history(history, path):
    plt.clf()
    fig, axis = plt.subplots(1, 3)
    axis[0].plot(history["rmse_by_epoch"])
    axis[0].set_title("RMSE by epoch")
    axis[1].plot(history["mse_by_epoch"])
    axis[1].set_title("MSE by epoch")
    axis[2].plot(history["mae_by_epoch"])
    axis[2].set_title("MAE by epoch")
    fig.savefig(osp.join(path, "histories_mse_rmse_mae_valdataset" + ".png"))
    plt.cla()


def scatter_plot_history(
    y_pred,
    y_true,
    y_predval,
    y_trueval,
    title_scatter,
    name,
    history,
    max_reference=300,
    figsize=None,
    title_all="Scatter and history",
    start_history_from_epoch=1,
):
    plt.clf()

    fig, axis = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title_all, fontsize=16)
    colors = []
    markers = []
    colorsval = []
    markersval = []

    for pred, real in zip(y_pred, y_true):
        difference = real - pred
        abs_diff = np.abs(difference)
        p20 = real * 0.2
        p40 = real * 0.4
        if abs_diff < p20:
            colors.append("green")
            markers.append(".")
        elif abs_diff < p40:
            colors.append("blue")
            markers.append("x")
        else:
            colors.append("red")
            markers.append("d")

    for pred, real in zip(y_predval, y_trueval):
        difference = real - pred
        abs_diff = np.abs(difference)
        p20 = real * 0.2
        p40 = real * 0.4
        if abs_diff < p20:
            colorsval.append("green")
            markersval.append(".")
        elif abs_diff < p40:
            colorsval.append("blue")
            markersval.append("x")
        else:
            colorsval.append("red")
            markersval.append("d")

    axis[0, 0].set_title(f"{title_scatter} TrainSet")
    axis[0, 0].scatter(y_pred, y_true, c=colors, marker=".")
    axis[0, 0].plot(range(max_reference), "g--")
    axis[0, 0].set_xlabel("Prediction")
    axis[0, 0].set_ylabel("Real latency")

    axis[1, 0].set_title(f"{title_scatter} ValidationSet")
    axis[1, 0].scatter(y_predval, y_trueval, c=colorsval, marker=".")
    axis[1, 0].plot(range(max_reference), "g--")
    axis[1, 0].set_xlabel("Prediction")
    axis[1, 0].set_ylabel("Real latency")

    axis[0, 1].plot(
        history["rmse_by_epoch"][start_history_from_epoch:], label="train", marker="."
    )
    axis[0, 1].plot(
        history["rmse_val_by_epoch"][start_history_from_epoch:],
        label="validation",
        marker=".",
    )
    axis[0, 1].set_title("RMSE by epoch")
    axis[0, 1].legend(loc="upper right")

    axis[1, 1].plot(
        history["mae_by_epoch"][start_history_from_epoch:], label="train", marker="."
    )
    axis[1, 1].plot(
        history["mae_val_by_epoch"][start_history_from_epoch:],
        label="validation",
        marker=".",
    )
    axis[1, 1].set_title("MAE by epoch")
    axis[1, 1].legend(loc="upper right")

    fig.savefig(name + ".png")
    plt.cla()


def _inv_log1p(x):
    return np.exp(x) - 1


###################################################################
###################################################################


class BaseRegression:
    def __init__(
        self,
        output_path="./",
        verbose=False,
        epochs=100,
        maxcardinality=1,
        in_channels=None,
        in_channels_neo_net=512,
        tree_units=None,
        tree_units_dense=None,
        query_input_size=None,
        query_hidden_inputs=None,
        query_output=240,
        early_stop_patience=10,
        early_stop_initial_patience=10,
        optimizer=None,
        figimage_size=(10, 8),
        tree_activation_tree=nn.LeakyReLU,
        tree_activation_dense=nn.LeakyReLU,
        ignore_first_aec_data=18,
        start_history_from_epoch=2,
    ):
        if tree_units_dense is None:
            tree_units_dense = [32, 28]

        if tree_units is None:
            tree_units = [256, 128, 64]

        if query_hidden_inputs is None:
            query_hidden_inputs = [260, 300]

        if optimizer is None:
            optimizer = {"optimizer": "Adam", "args": {"lr": 0.00015}}

        self.output_path = output_path
        self.net = None
        self.verbose = verbose
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.early_stop_initial_patience = early_stop_initial_patience
        # This is the output units for encoder of autoencoder model.
        self.in_channels_neo_net = in_channels_neo_net
        self.ignore_first_aec_data = ignore_first_aec_data
        self.query_input_size = query_input_size
        self.query_hidden_inputs = query_hidden_inputs
        self.query_output = query_output
        self.best_model = None
        self.optimizer = optimizer
        print(
            f"Model optimizer: {self.optimizer['optimizer']} lr: {self.optimizer['args']['lr']}"
        )
        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p, validate=True
        )
        scale_transformer = preprocessing.MinMaxScaler()

        self.pipeline = Pipeline(
            [("log", log_transformer), ("scale", scale_transformer)]
        )

        self.tree_transform = SPARQLTreeFeaturizer()
        self.in_channels = in_channels
        self.n = 0
        self.aec_net = None
        self.figimage_size = figimage_size

        # configs of tree model
        self.tree_units = tree_units
        self.tree_units_dense = tree_units_dense
        # configure the activation function of tree convolution layers(see model)
        self.tree_activation_tree = tree_activation_tree
        # configure the activation function of tree convolution dense layer(see model)
        self.tree_activation_dense = tree_activation_dense
        self.start_history_from_epoch = start_history_from_epoch

        self.history = {
            "rmse_by_epoch": [],
            "mse_by_epoch": [],
            "mae_by_epoch": [],
            "rmse_val_by_epoch": [],
            "mse_val_by_epoch": [],
            "mae_val_by_epoch": [],
        }
        self.maxcardinality = maxcardinality

    def log(self, *args):
        if self.verbose:
            print(*args)

    def num_items_trained_on(self):
        return self.n

    def get_pred(self):
        return self.tree_transform.get_pred_index()

    def load(self, path, best_model_path):
        with open(_n_path(path), "rb") as f:
            self.n = joblib.load(f)
        with open(_channels_path(path), "rb") as f:
            self.in_channels = joblib.load(f)

        self.net = NeoNet(
            self.aec_net,
            self.in_channels_neo_net,
            self.query_input_size,
            self.query_hidden_inputs,
            self.query_output,
            in_cuda=CUDA,
        )

        if best_model_path is not None:
            self.net.load_state_dict(torch.load(best_model_path))
        else:
            self.net.load_state_dict(torch.load(_nn_path(path)))

        if CUDA:
            self.net = self.net.cuda()
        self.net.eval()

        with open(_y_transform_path(path), "rb") as f:
            self.pipeline = joblib.load(f)
        with open(_x_transform_path(path), "rb") as f:
            self.tree_transform = joblib.load(f)

    def fix_tree(self, tree):
        """
        Trees in data must include in first position join type follow by predicates of childs. We check and fix this.
        """
        try:
            if len(tree) == 1:
                assert isinstance(tree[0], str)
                return tree
            else:
                assert len(tree) == 3
                assert isinstance(tree[0], str)
                preds = []
                if len(tree[0].split("ᶲ")) == 1:

                    tree_left = self.fix_tree(tree[1])
                    preds.extend(tree_left[0].split("ᶲ")[1:])

                    tree_right = self.fix_tree(tree[2])
                    preds.extend(tree_right[0].split("ᶲ")[1:])
                    preds = list(set(preds))
                    tree[0] = tree[0] + "ᶲ" + "ᶲ".join(preds)
                    return tree
                else:
                    return tree

        except Exception as ex:
            print(tree)
            return tree

    def save(self, path):
        # try to create a directory here
        os.makedirs(path, exist_ok=True)

        torch.save(self.net.state_dict(), _nn_path(path))
        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.pipeline, f)
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.tree_transform, f)
        with open(_channels_path(path), "wb") as f:
            joblib.dump(self.in_channels, f)
        with open(_n_path(path), "wb") as f:
            joblib.dump(self.n, f)

    def fit_transform_tree_data(self, ds_train, ds_val, ds_test):
        ds_train = self.json_loads_trees_ds(ds_train)
        ds_val = self.json_loads_trees_ds(ds_val)
        ds_test = self.json_loads_trees_ds(ds_test)
        data = []
        data.extend(ds_train)
        data.extend(ds_val)
        data.extend(ds_test)

        self.tree_transform.fit(data)

    def transform_trees(self, data):
        return self.tree_transform.transform(data)

    def fit(self, X, X_query, y, X_val, X_val_query, y_val):
        pass

    def predict(self, val_loader):
        results = []
        self.net.eval()
        with torch.no_grad():
            for (x, y_val) in val_loader:
                y_pred = self.net(x)
                results.extend(
                    list(
                        zip(
                            self.pipeline.inverse_transform(
                                y_pred.cpu().detach().numpy()
                            ),
                            y_val.cpu().detach().numpy(),
                        )
                    )
                )
        return results

    def predict_raw_data(self, trees, queries):
        results = []

        trees, queries, _ = self.json_loads(
            trees, queries, [None for _ in range(len(queries))]
        )
        trees = [self.fix_tree(x) for x in trees]
        print("X_val loaded")

        trees = self.tree_transform.transform(trees)
        pares = list(zip(trees, queries))
        dataloader = DataLoader(
            pares,
            batch_size=128,
            shuffle=False,
            collate_fn=self.collate_predict_with_card,
        )
        self.net.eval()
        with torch.no_grad():
            for x in dataloader:
                y_pred = self.net(x)
                results.extend(
                    self.pipeline.inverse_transform(y_pred.cpu().detach().numpy())
                )
        return results

    def predict_best(self, val_loader):

        results = []
        self.best_model.eval()
        with torch.no_grad():
            for (x, y_val) in val_loader:
                y_pred = self.best_model(x)
                results.extend(
                    list(
                        zip(
                            self.pipeline.inverse_transform(
                                y_pred.cpu().detach().numpy()
                            ),
                            y_val,
                        )
                    )
                )
        return results

    def index2sparse(self, tree, sizeindexes):
        pass

    def index2sparse2(self, tree, sizeindexes):
        pass

    def collate_with_card(self, x):
        """
        Preprocess inputs values, transform index2vec values,
        them predict aec.encoder to dimensionality reduction
        """
        trees = []
        targets = []
        sizeindexes = len(self.get_pred())
        other_pred_index = self.get_pred()["OTHER_PRED"]
        for x_features, target in x:
            tree, query = x_features
            b = np.zeros((sizeindexes))
            *query_features, card_features = query
            try:
                if type(card_features) == str:
                    raise Exception("you need to preprocess json_cardinality features")
                for key in card_features.keys():
                    if type(key) == int:
                        b[key] = card_features[key]
                    else:
                        b[other_pred_index] = card_features[key]
                        print("Predicate not found", key)
            except Exception as ex:
                print("Error en cardinalidades", card_features)
            trees.append(
                tuple(
                    [
                        self.index2sparse(tree, sizeindexes),
                        np.concatenate([query[:-1], b]).tolist(),
                    ]
                )
            )
            targets.append(target)

        targets = torch.tensor(np.array(targets))
        return trees, targets

    def collate_predict_with_card(self, x):
        """
        Preprocess inputs values, transform index2vec values,
        them predict aec.encoder to dimensionality reduction
        """
        trees = []
        sizeindexes = len(self.get_pred())
        for x_features in x:
            b = np.zeros((sizeindexes))
            tree, query = x_features
            try:
                for key in query[-1].keys():
                    b[key] = query[-1][key]
            except:
                print("Error en cardinalidades", str(query[-1]))
            trees.append(
                tuple(
                    [
                        self.index2sparse(tree, sizeindexes),
                        np.concatenate([query[:-1], b]).tolist(),
                    ]
                )
            )

        return trees

    def collate(self, x):
        """Preprocess inputs values, transform index2vec values, them predict aec.encoder to dimensionality reduction"""
        trees = []
        targets = []
        sizeindexes = len(self.get_pred())
        for tree, target in x:
            trees.append(tuple([self.index2sparse(tree[0], sizeindexes), tree[1]]))
            targets.append(target)

        targets = torch.tensor(targets)
        return trees, targets

    def collate2(self, x):
        """Only collocate x_data"""
        trees = []
        sizeindexes = len(self.get_pred())
        for tree in x:
            trees.append(tuple([self.index2sparse(tree[0], sizeindexes), tree[1]]))
        return trees

    def json_loads(self, X, X_query, Y):
        """read string with json data as json object"""
        respX = []
        respX_query = []
        respY = []
        for x, y in list(zip(list(zip(X, X_query)), Y)):
            try:
                x_tree, x_query = x
                x_tree = json.loads(x_tree)
                respX.append(x_tree)
                respX_query.append(x_query)
                respY.append(y)
            except:
                print("Error in data ignored!", x)
        return respX, respX_query, np.array(respY).reshape(-1, 1)

    def json_loads_trees_ds(self, ds):
        """read string with json data as json object from Dataframe, ignore bad jsons trees"""
        respX = []
        for index in range(ds.shape[0]):
            row = ds.iloc[index]
            try:
                data = json.loads(row["trees"])
                respX.append(data)
            except:
                print("Error in data ignored!", row["trees"])
        return respX

    def scatter_image(
        self, y_pred, y_test, title, name, max_reference=300, figsize=None
    ):
        scatter_image(
            y_pred, y_test, title, name, max_reference=max_reference, figsize=figsize
        )

    def plot_history(self, history):
        plot_history(history, self.output_path)

    def scatter_plot_history(
        self,
        y_pred,
        y_true,
        y_predval,
        y_trueval,
        title_scatter,
        name,
        history,
        max_reference=300,
        figsize=None,
        title_all="Scatter and history",
        start_history_from_epoch=1,
    ):
        scatter_plot_history(
            y_pred,
            y_true,
            y_predval,
            y_trueval,
            title_scatter,
            name,
            history,
            max_reference=max_reference,
            figsize=figsize,
            title_all=title_all,
            start_history_from_epoch=start_history_from_epoch,
        )

    def get_bestmodel(self):
        return self.best_model

    def get_model(self):
        return self.net
