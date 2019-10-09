from pca import PCA
from internal_coord import InternalCoordinateTransform
import proteins
import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import math
import numpy as np


def build_mixed_transformation_layers(
    input_node,
    training_data,
    topology,
    cartesian_selection,
    molecules=None,
    extra_basis=None,
):
    Z, cart_indices = proteins.mdtraj_to_z(
        topology, cartesian_selection, molecules, extra_basis
    )

    # First, make a list of coordinate indices from the list of atom indices.
    cart_coord_ind = []
    for i in cart_indices:
        cart_coord_ind.append(3 * i)
        cart_coord_ind.append(3 * i + 1)
        cart_coord_ind.append(3 * i + 2)
    # Next, make a list of all coordinate indices from the list of internal
    # coordinates.
    int_coord_ind = []
    for entry in Z:
        i = entry[0]
        int_coord_ind.append(3 * i)
        int_coord_ind.append(3 * i + 1)
        int_coord_ind.append(3 * i + 2)

    # Perform internal coordinate transformation
    ic_node = Ff.Node(
        input_node,
        InternalCoordinateTransform,
        {"training_data": training_data, "z_indices": Z, "cart_indices": cart_indices},
        name="internal",
    )

    # Shuffle the data so that all of the cartesian coordinates are
    # at the start, followed by the internals.
    permute_node = Ff.Node(
        ic_node,
        FixedPermute,
        {"perm_indices": cart_coord_ind + int_coord_ind},
        name="permute_cart_int",
    )

    # Split off the cartesian coordinates and the internal coordinates
    # into separate outputs.
    n_cart = len(cart_coord_ind)
    n_int = len(int_coord_ind)
    split_node = Ff.Node(
        permute_node,
        Fm.Split1D,
        {"split_size_or_sections": [n_cart, n_int], "dim": 0},
        name="split_cart_int",
    )

    # Perform PCA on the cartesian coordinates.
    pca_node = Ff.Node(
        split_node.out0,
        PCA,
        {"training_data": training_data[:, cart_coord_ind]},
        name="PCA",
    )

    # Merge the PCA coordinates and internal coordinates.
    merge_node = Ff.Node(
        [pca_node.out0, split_node.out1], Fm.Concat1d, {"dim": 0}, name="merge_cart_int"
    )

    nodes = [ic_node, permute_node, split_node, pca_node, merge_node]
    return nodes


class FixedPermute(nn.Module):
    """Permutes the input vector in a fixed way."""

    def __init__(self, dims_in, perm_indices):
        super().__init__()

        self.in_channels = dims_in[0][0]

        self.perm = perm_indices
        self.perm_inv = np.zeros_like(perm_indices)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = torch.LongTensor(self.perm)
        self.perm_inv = torch.LongTensor(self.perm_inv)

    def forward(self, x, rev=False):
        if not rev:
            return [x[0][:, self.perm]]
        else:
            return [x[0][:, self.perm_inv]]

    def jacobian(self, x, rev=False):
        # TODO: use batch size, set as nn.Parameter so cuda() works
        return 0.0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims
