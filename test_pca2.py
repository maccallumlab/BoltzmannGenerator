import torch
import FrEIA.framework as Ff
import pca
import numpy as np
import mdtraj as md



t = md.load("5ura_traj.dcd", top="5ura_start.pdb")
ind = t.top.select("backbone and resid 1 to 10")
t = t.atom_slice(ind)
t.center_coordinates()
t.superpose(t, 0)
t.save("in.pdb")
training_data = t.xyz
n_dim = training_data.shape[1] * 3
training_data = training_data.reshape(-1, n_dim)
training_data = torch.from_numpy(training_data.astype('float32'))

# Build the network
nodes = [Ff.InputNode(n_dim, name="input")]
nodes.append(Ff.Node(nodes[-1], pca.PCA, {"training_data": training_data}, name="pca"))
nodes.append(Ff.OutputNode(nodes[-1], name="output"))
net = Ff.ReversibleGraphNet(nodes, verbose=False)

# Pass the training data through the network, then reverse it.
x = training_data
z = net(training_data)
print(net.log_jacobian(run_forward=False))
x_prime = net(z, rev=True)
print(net.log_jacobian(run_forward=False, rev=True))

t.xyz = x_prime.detach().numpy().reshape(x_prime.shape[0], -1, 3)
t.save("out.pdb")