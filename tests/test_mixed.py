import torch
import torch.nn as nn
import numpy as np
import mdtraj as md
from boltzmann import protein

t = md.load("../data/AIYFL.dcd", top="../data/AIYFL.pdb")

# center everything
t.center_coordinates()

# superpose on the backbone
ind = t.top.select("backbone")
t.superpose(t, 0, atom_indices=ind, ref_atom_indices=ind)

# Gather the training data into a pytorch Tensor with the right shape
training_data = t.xyz
n_atoms = training_data.shape[1]
n_dim = n_atoms * 3
training_data_npy = training_data.reshape(-1, n_dim)
training_data = torch.from_numpy(training_data_npy.astype("float32"))

#
# Build the network
#
pca_block = protein.PCABlock("backbone", True)
transform = protein.MixedTransform(n_dim, t.topology, [pca_block], training_data)

#
# Test
#
with torch.no_grad():
    # Run some samples through the network forward
    x = training_data[:128, :]
    z_samples, jac_f = transform.forward(x)

    # Run them back through the network in reverse
    x_prime, jac_r = transform.inverse(z_samples)

assert torch.allclose(jac_f, -1 * jac_r, atol=0.1), "Jacobians in forward and reverse directions do not match"
assert torch.allclose(x_prime, x, atol=1e-3), "Coordinates do not match"