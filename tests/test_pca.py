import torch
from boltzmann import protein
import numpy as np
import mdtraj as md


t = md.load("../data/5ura_traj.dcd", top="../data/5ura_start.pdb")
ind = t.top.select("backbone and resid 1 to 10")
t = t.atom_slice(ind)
t.center_coordinates()
t.superpose(t, 0)
training_data = t.xyz
n_dim = training_data.shape[1] * 3
training_data = training_data.reshape(-1, n_dim)
training_data = torch.from_numpy(training_data.astype("float32"))

# Build the network
transform = protein.PCATransform(n_dim, training_data=training_data)

# Pass the training data through the network, then reverse it.
x = training_data
z, jac_f = transform.forward(training_data)

x_prime, jac_r = transform.inverse(z)

# Uncomment to write output for debugging
# t.xyz = x_prime.detach().numpy().reshape(x_prime.shape[0], -1, 3)
# t.save("out.pdb")

assert torch.allclose(
    jac_f, -1 * jac_r, atol=0.1
), "Jacobians in forward and reverse directions do not match"
assert torch.allclose(x_prime, x, atol=1e-3), "Coordinates do not match"


#
# Now repeat everything, but without superposing or dropping the last
# six degrees of freedom
#
t = md.load("../data/5ura_traj.dcd", top="../data/5ura_start.pdb")
ind = t.top.select("backbone and resid 1 to 10")
t = t.atom_slice(ind)
t.center_coordinates()
training_data = t.xyz
n_dim = training_data.shape[1] * 3
training_data = training_data.reshape(-1, n_dim)
training_data = torch.from_numpy(training_data.astype("float32"))

transform = protein.PCATransform(n_dim, training_data=training_data, drop_dims=0)

x = training_data
z, jac_f = transform.forward(training_data)

x_prime, jac_r = transform.inverse(z)

# Uncomment to write output for debugging
# t.xyz = x_prime.detach().numpy().reshape(x_prime.shape[0], -1, 3)
# t.save("out.pdb")

assert torch.allclose(
    jac_f, -1 * jac_r, atol=0.1
), "Jacobians in forward and reverse directions do not match"
assert torch.allclose(x_prime, x, atol=1e-3), "Coordinates do not match"
