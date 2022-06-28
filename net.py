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


class BaoNet(nn.Module):
    """TreeConv Net without query-level features"""
    def __init__(
            self,
            in_channels,
            activation_tree=nn.LeakyReLU,
            activation_dense=nn.LeakyReLU,
            tree_units=None,
            tree_units_dense=None
    ):
        super(BaoNet, self).__init__()

        # Default units
        if tree_units_dense is None:
            tree_units_dense = [32, 28]
        if tree_units is None:
            tree_units = [256, 128, 64]

        self.__in_channels = in_channels
        self.__cuda = False
        self.index = 0

        self.__in_channels = in_channels
        self.activation_tree = activation_tree
        self.activation_dense = activation_dense
        print(f"Activation function in treeConv layers: {self.activation_tree}")
        print(f"Activation function in tree dense output layers: {self.activation_dense}")

        # Add layers dynamically, first add TreeConvolution,TreeLayerNorm and TreeActivation
        layers = []
        for i, unit in enumerate(tree_units):
            if i == 0:
                layers.append(BinaryTreeConv(self.__in_channels, unit))
            else:
                layers.append(BinaryTreeConv(tree_units[i - 1], unit))
            layers.append(TreeLayerNorm())
            layers.append(TreeActivation(self.activation_tree()))
        layers.append(DynamicPooling())

        # Next add dense layers
        for i, unit in enumerate(tree_units_dense):
            if i == 0:
                layers.append(nn.Linear(tree_units[-1], unit))
            else:
                layers.append(nn.Linear(tree_units_dense[i - 1], unit))
            layers.append(self.activation_dense())

        # Last output layer with 1 unit
        layers.append(nn.Linear(tree_units_dense[-1], 1))
        print(layers)
        self.tree_conv = nn.Sequential(*layers)

    def in_channels(self):
        return self.__in_channels

    def forward(self, x):
        trees = prepare_trees(x, self.features, left_child, right_child,
                              cuda=self.__cuda)
        return self.tree_conv(trees)

    def cuda(self, device=None):
        self.__cuda = True
        return super().cuda(device)

    def features(self, x):
        return x[0]


class Autoencoder(nn.Module):

    def __init__(self, io_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(io_dim, 512),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, io_dim),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x


class QueryDataModel(nn.Module):
    """QueryLevel leyers used in NeoNet to concat query-level features to the tree features."""
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


class NeoNet(nn.Module):
    """
     Arquitectura de la red Neo para SPARQL. Query Level + Tree Level features.
    """
    def __init__(self,
                 in_channels,
                 query_input_size,
                 query_hidden_inputs=None,
                 query_output=240,
                 activation_tree=nn.LeakyReLU,
                 activation_dense=nn.LeakyReLU,
                 tree_units=None,
                 tree_units_dense=None
                 ):
        """
        Inicialización de la arquitectura. En esta se definen las unidades y funciones de activación utilizadas
        tanto para las capas densas a nivel de consulta como para las capas de las convoluciones sobre árboles y las
        capas finales de la red.

        :param in_channels: Plan level features size.
        :param query_input_size: Query level features size.
        :param query_hidden_inputs: list with hidden units in query layers.
        :param query_output: Size of last query layer.
        :param activation_tree: Activation function for tree convolutional layers.
        :param activation_dense: Activation function for query level layers.
        :param tree_units: list of  units in tree convolutional layers.
        :param tree_units_dense: list of  units in last layeres after convolutional layers.
        """
        super(NeoNet, self).__init__()
        if tree_units_dense is None:
            tree_units_dense = [32, 28]
        if tree_units is None:
            tree_units = [256, 128, 64]
        if query_hidden_inputs is None:
            query_hidden_inputs = [260, 300]
        self.__in_channels = in_channels
        self.__cuda = False
        
        self.activation_tree = activation_tree
        self.activation_dense = activation_dense
        self.query_model = QueryDataModel(query_input_size, query_hidden_inputs, query_output)
        print(f"Activation function in treeConv layers: {self.activation_tree}")
        print(f"Activation function in tree dense output layers: {self.activation_dense}")
        # Add layers dynamically, first add TreeConvolution,TreeLayerNorm and TreeActivation
        layers = []
        for i, unit in enumerate(tree_units):
            if i == 0:
                # If is the first layer use BinaryTreeConvWithQData to concat query level layers.
                layers.append(BinaryTreeConvWithQData(self.__in_channels, query_output, tree_units[i]))
            else:
                layers.append(BinaryTreeConv(tree_units[i - 1], unit))
            layers.append(TreeLayerNorm())
            layers.append(TreeActivation(self.activation_tree()))
        layers.append(DynamicPooling())

        # Next add dense layers
        for i, unit in enumerate(tree_units_dense):
            if i == 0:
                layers.append(nn.Linear(tree_units[-1], unit))
            else:
                layers.append(nn.Linear(tree_units_dense[i - 1], unit))
            layers.append(self.activation_dense())

        # Last output layer with 1 unit
        layers.append(nn.Linear(tree_units_dense[-1], 1))
        print(layers)
        self.tree_conv = nn.Sequential(*layers)

    def in_channels(self):
        return self.__in_channels

    def forward(self, data):
        """"""
        tree_data = [tree[0] for tree in data]
        query_data = [tree[1] for tree in data]
        query_data = torch.from_numpy(np.asarray(query_data)).to(torch.float32)

        query_data = query_data.cuda()

        qm_output = self.query_model(query_data)
        query_data.cpu()
        del query_data
        trees = prepare_trees(tree_data, self.features, left_child, right_child, cuda=self.__cuda)
        conv_result =  self.tree_conv((trees, qm_output))
        del trees
        return conv_result

    def cuda(self, device=None):
        self.__cuda = True
        return super().cuda(device)
    
    def features(self, x):
        return x[0]

class TreeNet(nn.Module):

    def __init__(self,
                 in_channels,
                 activation_tree=nn.LeakyReLU,
                 activation_dense=nn.LeakyReLU,
                 tree_units=None,
                 tree_units_dense=None
                 ):
        super(TreeNet, self).__init__()
        if tree_units_dense is None:
            tree_units_dense = [32, 28]
        if tree_units is None:
            tree_units = [256, 128, 64]

        self.__in_channels = in_channels
        self.__cuda = False

        self.activation_tree = activation_tree
        self.activation_dense = activation_dense
        print(f"Activation function in treeConv layers: {self.activation_tree}")
        print(f"Activation function in tree dense output layers: {self.activation_dense}")
        # Add layers dynamically, first add TreeConvolution,TreeLayerNorm and TreeActivation
        layers = []
        for i, unit in enumerate(tree_units):
            if i == 0:
                layers.append(BinaryTreeConv(self.__in_channels, unit))
            else:
                layers.append(BinaryTreeConv(tree_units[i - 1], unit))
            layers.append(TreeLayerNorm())
            layers.append(TreeActivation(self.activation_tree()))
        layers.append(DynamicPooling())

        # Next add dense layers
        for i, unit in enumerate(tree_units_dense):
            if i == 0:
                layers.append(nn.Linear(tree_units[-1], unit))
            else:
                layers.append(nn.Linear(tree_units_dense[i - 1], unit))
            layers.append(self.activation_dense())

        # Last output layer with 1 unit
        layers.append(nn.Linear(tree_units_dense[-1], 1))
        print(layers)
        self.tree_conv = nn.Sequential(*layers)

    def in_channels(self):
        return self.__in_channels

    def forward(self, tree_data):

        trees = prepare_trees(tree_data, self.features, left_child, right_child, cuda=self.__cuda)
        conv_result = self.tree_conv(trees)
        return conv_result

    def cuda(self, device=None):
        self.__cuda = True
        return super().cuda(device)

    def features(self, x):
        return x[0]
