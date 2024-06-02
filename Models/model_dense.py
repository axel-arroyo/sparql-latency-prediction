import gc
import numpy as np
import torch.nn as nn
import torch.optim
import joblib
import os
from model import scatter_image, plot_history, scatter_plot_history
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from torch.nn import MSELoss
import datetime
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from torch import from_numpy, float32
from net_baseline import DenseNet

plt.rcParams.update({'figure.max_open_warning': 0})

from featurize import SPARQLTreeFeaturizer
from early_stopping import EarlyStopping
from net import Autoencoder, NeoNet, BaoNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from custom_loss import CustomMSELoss
from memory_profiler import profile
import logging

CUDA = torch.cuda.is_available()


# np.seterr(all='raise')

def _inv_log1p(x):
    return np.exp(x) - 1

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



class DenseRegression:
    def __init__(self,
                 verbose=False,
                 epochs=100,
                 query_input_size=None,
                 query_hidden_inputs=None,
                 query_output=240,
                 loss=None,
                 early_stop_patience=10,
                 early_stop_initial_patience=10,
                 optimizer=None,
                 figimage_size=(10, 8),
                 activation_dense=nn.LeakyReLU,
                 preds2index={},
                 encode_cardinalities=False
                 ):

        if query_hidden_inputs is None:
            query_hidden_inputs = [260, 300]

        if optimizer is None:
            optimizer = {'optimizer': 'Adam', 'args': {'lr': 0.00015}}

        self.__net = None
        self.__verbose = verbose
        self.__epochs = epochs
        self.__early_stop_patience = early_stop_patience
        self.__early_stop_initial_patience = early_stop_initial_patience
        # This is the output units for encoder of autoencoder model.
        self.__query_input_size = query_input_size
        self.__query_hidden_inputs = query_hidden_inputs
        self.__query_output = query_output
        self.__best_model = None
        self.__optimizer = optimizer
        self.preds2index = preds2index
        self.loss = loss
        print(f"Model optimizer: {self.__optimizer['optimizer']} lr: {self.__optimizer['args']['lr']}")
        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p,
            validate=True)
        scale_transformer = preprocessing.MinMaxScaler()

        self.pipeline = Pipeline([("log", log_transformer),
                                    ("scale", scale_transformer)])

#         log_transformer = preprocessing.FunctionTransformer(
#             np.log, np.exp,
#             validate=True)
#         scale_transformer = preprocessing.StandardScaler()

        self.pipeline = Pipeline([("log", log_transformer),
                                    ("scale", scale_transformer)])

        self.__n = 0
        self.__aec_net = None
        self.__figimage_size = figimage_size
        self.__activation_dense=activation_dense
        self.history = {
            'rmse_by_epoch': [],
            'mse_by_epoch': [],
            'mae_by_epoch': [],
            'rmse_val_by_epoch': [],
            'mse_val_by_epoch': [],
            'mae_val_by_epoch': []
        }
        self.encode_cardinalities = encode_cardinalities
    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def num_items_trained_on(self):
        return self.__n

    def get_pred(self):
        return self.__tree_transform.get_pred_index()

    def load(self, path, best_model_path):
        with open(_n_path(path), "rb") as f:
            self.__n = joblib.load(f)
        with open(_channels_path(path), "rb") as f:
            self.__in_channels = joblib.load(f)

        self.__net = NeoNet(self.__aec_net, self.__in_channels_neo_net, self.__query_input_size,
                            self.__query_hidden_inputs, self.__query_output)

        if best_model_path is not None:
            self.__net.load_state_dict(torch.load(best_model_path))
        else:
            self.__net.load_state_dict(torch.load(_nn_path(path)))
        self.__net.cuda()
        self.__net.eval()

        with open(_y_transform_path(path), "rb") as f:
            self.pipeline = joblib.load(f)
        with open(_x_transform_path(path), "rb") as f:
            self.__tree_transform = joblib.load(f)

    def fix_tree(self, tree):
        """
        Trees in data must include in first position join type follow by predicates of childs. We check and fix this.
        """
        try:
            if len(tree) == 1:
                assert (isinstance(tree[0], str))
                return tree
            else:
                assert (len(tree) == 3)
                assert (isinstance(tree[0], str))
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

        torch.save(self.__net.state_dict(), _nn_path(path))
        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.pipeline, f)
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.__tree_transform, f)
        with open(_channels_path(path), "wb") as f:
            joblib.dump(self.__in_channels, f)
        with open(_n_path(path), "wb") as f:
            joblib.dump(self.__n, f)

    def fit_transform_tree_data(self, ds_train, ds_val, ds_test):
        ds_train = self.json_loads(ds_train)
        ds_val = self.json_loads(ds_val)
        ds_test = self.json_loads(ds_test)
        data = []
        data.extend(ds_train)
        data.extend(ds_val)
        data.extend(ds_test)

        self.__tree_transform.fit(data)

    def transform_trees(self, data):
        return self.__tree_transform.transform(data)

    def fit(self, X_query, y, X_val_query, y_val):
        if isinstance(y, list):
            y = np.array(y)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
        # file logger
        fh = logging.FileHandler('./aarroyo/output.log', mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # console logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        self.__n = len(X_query)
        max_y = np.max(y)
        
        # Fit target transformer
        self.pipeline.fit_transform(y.reshape(-1, 1))
        
        pairs = list(zip(X_query, y))
        pairs_val = list(zip(X_val_query, y_val))

        if self.encode_cardinalities:
            dataset = DataLoader(pairs, batch_size=256, num_workers=0, shuffle=True, collate_fn=self.collate)
            dataset_val = DataLoader(pairs_val, batch_size=256, num_workers=0, shuffle=True, collate_fn=self.collate)
        else:
            dataset = DataLoader(pairs, batch_size=256, num_workers=0, shuffle=True, collate_fn=self.collatenocard)
            dataset_val = DataLoader(pairs_val, batch_size=256, num_workers=0, shuffle=True, collate_fn=self.collatenocard)
        self.__query_input_size = len(X_query[0])
        if self.encode_cardinalities:
            self.__query_input_size = self.__query_input_size + len(self.preds2index) -1
            
        self.__log("Initial input channels of query model:", self.__query_input_size)
        self.__net = DenseNet(
            self.__query_input_size,
            self.__query_hidden_inputs,
            self.__query_output,
            activation_dense=self.__activation_dense,
            dropout_rate=0.25
        )
        if CUDA:
            self.__net = self.__net.cuda()

        # initialize the early_stopping object
        early_stopping = EarlyStopping(initial_patience=self.__early_stop_initial_patience,
                                       patience=self.__early_stop_patience, verbose=True)

        if self.__optimizer["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.__net.parameters(), **self.__optimizer["args"])
        elif self.__optimizer["optimizer"] == "Adagrad":
            optimizer = torch.optim.Adagrad(self.__net.parameters(), **self.__optimizer["args"])
        else:
            optimizer = torch.optim.SGD(self.__net.parameters(), **self.__optimizer["args"])

        if self.loss and self.loss['func'] == "CustomMSELoss":
            loss_fn = CustomMSELoss(self.loss['threshold'], weight_factor=self.loss['weight_factor'])
        else:
            loss_fn = torch.nn.MSELoss()
        


        losses = []
        
        # Fot ID of images
        import random
        id_label = "".join([str(a) for a in random.sample(range(20), 5)])
        if not os.path.exists("./" + id_label + "/"):
            os.makedirs("./" + id_label + "/")
            
        assert np.mean(y_val) > 5, "y_val must be in real scale"
        self.__log("Epochs to run:", 4)
        for epoch in range(self.__epochs):
            self.__net.train()
            loss_accum = 0
            results_train = []
            for (query_data, y_train) in dataset:
                y_train_scaled = torch.tensor(self.pipeline.transform(y_train.reshape(-1, 1)).astype(np.float32))
                query_data = torch.from_numpy(np.asarray(query_data)).to(torch.float32)


                if CUDA:
                    y_train_scaled = y_train_scaled.cuda()
                    query_data = query_data.cuda()
                y_pred = self.__net(query_data)
                loss = loss_fn(y_pred, y_train_scaled)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lost_item = loss.item()
                loss_accum += lost_item
                #                 print('{} Epoch {}, Training loss batch {}'.format(datetime.datetime.now(), epoch, lost_item))
                inverted = self.pipeline.inverse_transform(y_pred.cpu().detach().numpy())

                results_train.extend \
                    (list(zip(inverted, y_train)))
            #                 y_train_scaled.detach()
            #                 del y_train_scaled
            #                 torch.cuda.empty_cache()

            loss_accum /= len(dataset)
            losses.append(loss_accum)

            print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_accum / len(dataset)))

            # Prediction in subsample of train
            torch.cuda.empty_cache()

            y_pred_train, y_real_train = zip(*results_train)
            y_pred_train
            msetrain = mean_squared_error(y_real_train, y_pred_train)
            maetrain = mean_absolute_error(y_real_train, y_pred_train)
            rmsetrain = np.sqrt(msetrain)
            self.history['mse_by_epoch'].append(msetrain)
            self.history['rmse_by_epoch'].append(rmsetrain)
            self.history['mae_by_epoch'].append(maetrain)

            # Testing the model

            results_val = self.predict(dataset_val)
            y_pred_val, y_real_val = zip(*results_val)
            mseval = mean_squared_error(y_real_val, y_pred_val)
            maeval = mean_absolute_error(y_real_val, y_pred_val)
            rmseval = np.sqrt(mseval)
            self.history['mse_val_by_epoch'].append(mseval)
            self.history['rmse_val_by_epoch'].append(rmseval)
            self.history['mae_val_by_epoch'].append(maeval)
            #             print(f"RMSE in TRAIN: {rmsetrain} : RMSE in VAL: {rmseval}")
            logger.info('==> Epoch {}, \tTRAIN_LOSS: {}\t_TRAIN_RMSE: {}, \tVAL_LOSS: {}, \tVAL_RMSE: {}'.format(
                epoch, msetrain, rmsetrain, mseval, rmseval))
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            best_model = early_stopping(np.average(self.history['rmse_val_by_epoch']), self.__net)
            if best_model is not None:
                self.__best_model = best_model
            if early_stopping.early_stop:
                print("Early stopping the training.")
                break
            # selfscatter_plot_history(y_pred, y_test, title, name, history, max_refference=300, figsize=None)

            #             print("ID corrida: ", id_label)
            if epoch: # % 4 == 0:
                self.scatter_plot_history(
                    y_pred_train,
                    y_real_train,
                    y_pred_val,
                    y_real_val,
                    "Scatter real latency vs prediction on: ",
                    "./" + id_label + "/" + "baseline_scatter_train_val_epoch_" + "{:03d}".format(epoch),
                    self.history,
                    max_refference=int(max_y + 10),
                    figsize=self.__figimage_size,
                    title_all=f"Scatter and history, RMSE Train: {rmsetrain}, RMSE VAL: {rmseval}, Epoch: {epoch}"
                )
            gc.collect()

    #         self.plot_history(history)

    def predict(self, val_loader):
        results = []
        self.__net.eval()
        with torch.no_grad():

            for (query_data, y_val) in val_loader:
                query_data = torch.from_numpy(np.asarray(query_data)).to(torch.float32)
                if CUDA:
                    query_data = query_data.cuda()

                y_pred = self.__net(query_data)
                results.extend(list(zip(self.pipeline.inverse_transform(y_pred.cpu().detach().numpy()), y_val)))
        return results

    def predict_raw_data(self, queries):
        results = []
        dataloader = DataLoader(queries, batch_size=128, shuffle=False)
        self.__net.eval()
        with torch.no_grad():
            for x in dataloader:
                query_data = torch.from_numpy(np.asarray(x)).to(torch.float32)
                if CUDA:
                    query_data = query_data.cuda()

                y_pred = self.__net(query_data)
                results.extend(self.pipeline.inverse_transform(y_pred.cpu().detach().numpy()))
        return results

    def predict_best(self, val_loader):

        results = []
        self.__best_model.eval()
        with torch.no_grad():
            for (query_data, y_val) in val_loader:
                query_data = torch.from_numpy(np.asarray(query_data)).to(torch.float32)
                if CUDA:
                    query_data = query_data.cuda()

                y_pred = self.__best_model(query_data)
                results.extend(list(zip(self.pipeline.inverse_transform(y_pred.cpu().detach().numpy()), y_val)))
        return results

    def index2sparse(self, tree, sizeindexes):
        resp = []
        for el in tree:
            if type(el[0]) == tuple:
                resp.append(self.index2sparse(el, sizeindexes))
            else:
                a = np.array(el)
                b = np.zeros((a.size, sizeindexes))
                b[np.arange(a.size), a] = 1
                onehot = np.sum(b, axis=0, keepdims=True)
                # Split in 9 because it are de init index for predicates, @see SparqlTreeBuilder.get_index_seq
                onehot2pred = from_numpy(onehot[0][9:]).to(float32).cuda()
                pred = self.__aec_net.encoder(onehot2pred).cpu().detach().numpy()
                resp.append((np.concatenate(onehot[:9], pred)))
        return tuple(resp)

    def collatenocard(self, x):
        """Preprocess inputs values, transform index2vec values, them predict aec.encoder to dimensionality reduction"""
        queries = []
        targets = []
        for query, target in x:
            
            queries.append(query)
            targets.append(target)

        targets = torch.tensor(targets, dtype=torch.float32)
        return queries, targets
    def collate(self, x):
        """Preprocess inputs values, transform index2vec values, them predict aec.encoder to dimensionality reduction"""
        queries = []
        targets = []
        for query, target in x:
            b = np.zeros((len(self.preds2index)))
            try:
                for key in query[-1].keys():
                    b[key] = query[-1][key]
            except:
                print("Error en cardinalidades",str(query[-1]))
            
            if target <= 0 or target > 70:
                print(target)
                continue
            queries.append(np.concatenate([query[:-1], b]).tolist())
            
            targets.append(target)

        targets = torch.tensor(targets, dtype=torch.float32)
        return queries, targets

    def collate2(self, x):
        queries = []

        for query in x:
            b = np.zeros((len(self.preds2index)))
            try:
                for key in query[-1].keys():
                    b[key] = query[-1][key]
            except:
                print("Error en cardinalidades",str(query[-1]))
            
            queries.append(np.concatenate([query[:-1], b]))
        return queries


    def scatter_image(self, y_pred, y_test, title, name, max_refference=300, figsize=None):
        scatter_image(y_pred, y_test, title, name, max_refference=max_refference, figsize=figsize)

    def plot_history(self, history):
        plot_history(history)

    def scatter_plot_history(self, y_pred, y_true, y_predval, y_trueval, title_scatter, name, history,
                             max_refference=300,
                             figsize=None,
                             title_all="Scatter and history"):
        scatter_plot_history(y_pred, y_true, y_predval, y_trueval, title_scatter, name, history,
                             max_refference=max_refference,
                             figsize=figsize, title_all=title_all)
