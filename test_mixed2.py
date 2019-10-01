import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import mixed_transform
import numpy as np
import mdtraj as md
from tqdm import tqdm

#
# Main script starts here
#
t = md.load("5ura_traj.dcd", top="5ura_start.pdb")

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

# First create the input node
nodes = [Ff.InputNode(n_dim, name="input")]

# Next create the node for internal coordinates
mixed_nodes = mixed_transform.build_mixed_transformation_layers(
    nodes[-1], training_data, t.topology, "backbone"
)
nodes.extend(mixed_nodes)

nodes.append(Ff.OutputNode(nodes[-1], name="output"))
net = Ff.ReversibleGraphNet(nodes, verbose=False)

#
# Test
#

with torch.no_grad():
    # Run some samples through the network forward
    x = training_data[:16, :]
    z_samples = net(x)
    z_jac = net.log_jacobian(run_forward=False)

    # Run them back through the network in reverse
    x_prime = net(z_samples, rev=True)
    x_jac = net.log_jacobian(run_forward=False, rev=True)

assert torch.allclose(z_jac, -1 * x_jac, atol=0.1), "Jacobians in forward and reverse directions do not match"
assert torch.allclose(x_prime, x, atol=1e-3), "Coordinates do not match"