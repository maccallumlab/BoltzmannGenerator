import torch
import torch.nn as nn
from boltzmann.protein import InternalCoordinateTransform
import numpy as np
import mdtraj as md
from tqdm import tqdm

t = md.load("../data/5ura_traj.dcd", top="../data/5ura_start.pdb")
ind = t.top.select("backbone")
t = t.atom_slice(ind)
t.center_coordinates()
ind_super = t.top.select("name N CA C O and resid 35")
t.superpose(t, 0, atom_indices=ind_super, ref_atom_indices=ind_super)
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
coords = training_data[:train_size, :]
z, jac_f = transform.forward(coords)

# We should not change the first 6 coords
assert torch.allclose(coords[:, :9], z[:, :9])

# We should be able to undo the transformation
x, jac_r = transform.inverse(z)
assert torch.allclose(coords, x, atol=1e-3)
assert torch.allclose(jac_f, -1 * jac_r, atol=1e-6)