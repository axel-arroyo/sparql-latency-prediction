import torch
import torch.nn as nn
from torch import from_numpy, float32
import numpy as np
from TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm, BinaryTreeConvWithQData
from TreeConvolution.tcnn import TreeActivation, DynamicPooling
from TreeConvolution.util import prepare_trees


def left_child(x):
    if len(x) != 3:
        return None
    return x[1]


def right_child(x):
    if len(x) != 3:
        return None
    return x[2]


class Autoencoder(nn.Module):
    def __init__(self, io_dim, pred_to_index):
        super(Autoencoder, self).__init__()
        self._pred_to_index = pred_to_index
        self.encoder = nn.Sequential(
            nn.Linear(io_dim, 1024), nn.ReLU(True), nn.Linear(1024, 512), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(True), nn.Linear(1024, io_dim)
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x


class QueryDataModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(QueryDataModel, self).__init__()
        self.linear1 = nn.Linear(D_in, H[0])
        self.linear2 = nn.Linear(H[0], H[1])
        self.out = nn.Linear(H[1], D_out)
        self.activation = nn.ReLU()
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.activation(self.linear1(x))
        x = self.drop(x)
        x = self.activation(self.linear2(x))
        x = self.drop(x)
        x = self.activation(self.out(x))
        return x


class DenseNet(nn.Module):
    def __init__(
        self,
        query_input_size,
        query_hidden_inputs=None,
        query_output=1,
        activation_dense=nn.LeakyReLU,
        dropout_rate=0.25,
    ):
        super(DenseNet, self).__init__()

        self.query_output = query_output
        self.query_hidden_inputs = query_hidden_inputs
        if self.query_hidden_inputs is None:
            self.query_hidden_inputs = [260, 300]
        self.__cuda = False

        self.activation_dense = activation_dense

        layers = []
        self.dropout_rate = dropout_rate
        # Next add dense layers
        for i, unit in enumerate(self.query_hidden_inputs):
            if i > 0:
                layers.append(nn.Linear(self.query_hidden_inputs[i - 1], unit))
            else:
                layers.append(nn.Linear(query_input_size, unit))

            layers.append(self.activation_dense())
            layers.append(nn.Dropout(self.dropout_rate))

        # Last output layer with 1 unit
        layers.append(nn.Linear(self.query_hidden_inputs[-1], self.query_output))

        print(layers)
        self.query_model = nn.Sequential(*layers)

    def forward(self, query_data):

        return self.query_model(query_data)

    def cuda(self, device=None):
        self.__cuda = True
        return super().cuda(device)
