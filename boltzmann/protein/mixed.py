from .pca import PCATransform
from .internal import InternalCoordinateTransform
from . import zmatrix
import torch
import torch.nn as nn
import math
import numpy as np
from collections import namedtuple
import itertools


PCABlock = namedtuple("PCABlock", "selection_string remove_dof")


class MixedTransform(nn.Module):
    def __init__(
        self,
        n_dim,
        topology,
        pca_blocks,
        training_data,
        molecule_extents=None,
        extra_basis=None,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.topology = topology
        self._remove_dof = 0

        # First, get all of the cartesian indices for the PCA blocks
        cart_indices = [
            self.topology.select(block.selection_string) for block in pca_blocks
        ]
        # Make sure there is no overlap.
        if len(cart_indices) == 1:
            intersection = []
        else:
            intersection = set.intersection(*[set(cart) for cart in cart_indices])
        if intersection:
            raise ValueError("Selections in PCA blocks must not overlap.")

        # Merge them together and calculate the z-matrix.
        union_cart_indices = list(itertools.chain(*cart_indices))
        z = zmatrix.mdtraj_to_z(
            topology,
            union_cart_indices,
            molecules=molecule_extents,
            extra_basis=extra_basis,
        )

        # Create our internal coordinate transform
        self.ic_transform = InternalCoordinateTransform(
            n_dim, z, union_cart_indices, training_data
        )

        # `permute` will put all of the coordinates for PCA1 first, followed by
        # PCA2, ..., then followed by all of the internal coordinates.
        # `permute_inv` does the opposite.
        permute = torch.zeros(n_dim, dtype=torch.long)
        permute_inv = torch.zeros(n_dim, dtype=torch.long)
        all_ind = union_cart_indices + [row[0] for row in z]
        for i, j in enumerate(all_ind):
            permute[3 * i + 0] = torch.as_tensor(3 * j + 0, dtype=torch.long)
            permute[3 * i + 1] = torch.as_tensor(3 * j + 1, dtype=torch.long)
            permute[3 * i + 2] = torch.as_tensor(3 * j + 2, dtype=torch.long)
            permute_inv[3 * j + 0] = torch.as_tensor(3 * i + 0, dtype=torch.long)
            permute_inv[3 * j + 1] = torch.as_tensor(3 * i + 1, dtype=torch.long)
            permute_inv[3 * j + 2] = torch.as_tensor(3 * i + 2, dtype=torch.long)
        self.register_buffer("permute", permute)
        self.register_buffer("permute_inv", permute_inv)

        # `forward_indices` is a list of lists of indices to go through each PCA
        # transform in the forward direction. `reverse_indices` is the same, but
        # in reverse. They might be different, because some degrees of freedom
        # may be dropped in the PCA transforms.
        forward_indices = []
        reverse_indices = []
        offset_fwd = 0
        offset_rev = 0
        for indices, (_, remove_dof) in zip(cart_indices, pca_blocks):
            fwd = [i + offset_fwd for i in range(len(indices))]
            offset_fwd += len(fwd)
            if remove_dof:
                rev = [i + offset_rev for i in range(len(indices[:-2]))]
                self._remove_dof += 6
            else:
                rev = [i + offset_rev for i in range(len(indices))]
            offset_rev += len(rev)
            fwd = list(
                itertools.chain(*[[3 * i + 0, 3 * i + 1, 3 * i + 2] for i in fwd])
            )
            rev = list(
                itertools.chain(*[[3 * i + 0, 3 * i + 1, 3 * i + 2] for i in rev])
            )
            forward_indices.append(fwd)
            reverse_indices.append(rev)
        self.forward_indices = forward_indices
        self.reverse_indices = reverse_indices

        # Now, setup the indexing for the internal coordinates that bypass the
        # PCA layers. The forward and reverse directions may have different indices
        # because the forward and reverse PCA layers may have different sizes due
        # to dropping degrees of freedom.
        self.forward_z_indices = list(range(3 * offset_fwd, n_dim))
        self.reverse_z_indices = list(range(3 * offset_rev, n_dim - self._remove_dof))

        # Create our PCA transforms
        # Permute the training data to have PCAs first
        training_data = training_data[:, self.permute]
        self.pca_transforms = torch.nn.ModuleList()
        for indices, (_, remove_dof) in zip(self.forward_indices, pca_blocks):
            pca = PCATransform(
                len(indices), training_data[:, indices], 6 if remove_dof else 0
            )
            self.pca_transforms.append(pca)

    @property
    def out_dim(self):
        return self.n_dim - self._remove_dof

    def forward(self, x, context=None):
        # Create the jacobian vector
        jac = x.new_zeros(x.shape[0])

        # Run transform to internal coordinates.
        x, new_jac = self.ic_transform.forward(x)
        jac = jac + new_jac

        # Permute to put PCAs first.
        x = x[:, self.permute]

        # Split off the PCA coordinates and internal coordinates
        pca_inputs = [x[:, indices] for indices in self.forward_indices]
        int_coords = x[:, self.forward_z_indices]

        # Run through PCA.
        pca_outputs = []
        for pca, coords in zip(self.pca_transforms, pca_inputs):
            new_coords, new_jac = pca.forward(coords)
            pca_outputs.append(new_coords)
            jac = jac + new_jac

        # Merge everything back together.
        x = torch.cat(pca_outputs + [int_coords], dim=1)

        return x, jac

    def inverse(self, x, context=None):
        # Create the jacobian vector
        jac = x.new_zeros(x.shape[0])

        # Separate out the PCAs and internal coordinates
        pca_inputs = [x[:, indices] for indices in self.reverse_indices]
        int_coords = x[:, self.reverse_z_indices]

        # Run through PCA
        pca_outputs = []
        for pca, coords in zip(self.pca_transforms, pca_inputs):
            new_coords, new_jac = pca.inverse(coords)
            pca_outputs.append(new_coords)
            jac = jac + new_jac

        # Merge everything back together
        x = torch.cat(pca_outputs + [int_coords], dim=1)

        # Permute back into atom order
        x = x[:, self.permute_inv]

        # Run through inverse internal coordinate transform
        x, new_jac = self.ic_transform.inverse(x)
        jac = jac + new_jac

        return x, jac
