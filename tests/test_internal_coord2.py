import torch
import torch.nn as nn
from boltzmann.protein import InternalCoordinateTransform
import numpy as np
import mdtraj as md
from tqdm import tqdm

t = md.load("../data/AIYFL.dcd", top="../data/AIYFL.pdb")
training_data = t.xyz
n_atoms = training_data.shape[1]
n_dim = n_atoms * 3
training_data_npy = training_data.reshape(-1, n_dim)
training_data = torch.from_numpy(training_data_npy.astype("float32"))

# Build a fake z-matrix
graph = []
for i in range(3, n_atoms):
    graph.append((i, (i - 1, i - 2, i - 3)))

transform = InternalCoordinateTransform(
    dims=n_dim,
    training_data=training_data,
    z_indices=graph,
    cart_indices=[0, 1, 2],
)

train_size = 64
x = training_data[:train_size, :]
z, jac_f = transform.forward(x)

# We should not change the first 6 coords
assert torch.allclose(x[:, :9], z[:, :9])

# We should be able to undo the transformation
x_prime, jac_r = transform.inverse(z)
assert torch.allclose(x, x_prime, atol=1e-3)
assert torch.allclose(jac_f, -1 * jac_r, atol=1e-6)