import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import pca
import numpy as np
import mdtraj as md
from tqdm import tqdm
from matplotlib import pyplot as pp
import matplotlib
import time


matplotlib.use("Qt4Agg")
pp.ion()

# t = md.load("5ura_apo_dry_r9_p6t10_p100.pdb")
t = md.load("5ura_apo_solv_r9_p6t10_p5.dcd", top="5ura_apo_dry_r9_p6t10_p100.pdb")
ind = t.top.select("backbone and resid 20 to 40")
t = t.atom_slice(ind)
t.center_coordinates()
t.superpose(t, 0)
t.save("in.pdb")
training_data = t.xyz
n_dim = training_data.shape[1] * 3
training_data_npy = training_data.reshape(-1, n_dim)
training_data = torch.from_numpy(training_data_npy.astype("float32"))


class CreateFC:
    def __init__(self, n_hidden=None):
        self.n_hidden = n_hidden

    def __call__(self, c_in, c_out):
        if self.n_hidden is None:
            hidden = 2 * c_in
        else:
            hidden = self.n_hidden
        lin1 = nn.Linear(c_in, hidden)
        lin2 = nn.Linear(hidden, hidden)
        lin3 = nn.Linear(hidden, hidden)
        lin4 = nn.Linear(hidden, c_out)

        # Initialize the weights in each layer.
        # Kaiming initialization is suitable for
        # ReLU activation functions.
        torch.nn.init.kaiming_uniform_(lin1.weight)
        torch.nn.init.kaiming_uniform_(lin2.weight)
        torch.nn.init.kaiming_uniform_(lin3.weight)
        # Initialize the weights and biases in the last
        # Layer to zero, which gives the identity transform
        # as our starting point.
        torch.nn.init.zeros_(lin4.weight)
        torch.nn.init.zeros_(lin4.bias)

        return nn.Sequential(
            lin1, nn.LeakyReLU(), lin2, nn.LeakyReLU(), lin3, nn.LeakyReLU(), lin4
        )


# Build the network
nodes = [Ff.InputNode(n_dim, name="input")]
nodes.append(Ff.Node(nodes[-1], pca.PCA, {"training_data": training_data}, name="pca"))
n_glow = 4
for i in range(n_glow):
    nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": i}, name=f"permute_{i}"))
    nodes.append(
        Ff.Node(
            nodes[-1],
            Fm.GLOWCouplingBlock,
            {"subnet_constructor": CreateFC(128), "clamp": 2},
            name=f"glow_{i}",
        )
    )
nodes.append(Ff.OutputNode(nodes[-1], name="output"))
net = Ff.ReversibleGraphNet(nodes, verbose=False)

losses = []
val_losses = []
epochs = 2000
n_batch = 128  # This is the number of data points per batch

n = training_data_npy.shape[0]
n_val = 128
np.random.shuffle(training_data_npy)

val_data = torch.as_tensor(training_data_npy[:n_val, :])
train_data = torch.as_tensor(training_data_npy[n_val:, :])
I = np.arange(train_data.shape[0])  # A list of indices into the training set

learning_rate = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Setup stuff for animation
fig = pp.figure()
ax = pp.axes(xlim=(-4, 4), ylim=(-4, 4))
line, = ax.plot([], [], linestyle="None", marker="o")

pp.show(False)
pp.draw()

with tqdm(range(epochs)) as progress:
    for epoch in progress:
        net.train()
        index_batch = np.random.choice(I, n_batch, replace=True)
        x_batch = train_data[index_batch, :]

        z = net(x_batch)

        loss = 0.5 * torch.mean(z ** 2) - torch.mean(
            net.log_jacobian(run_forward=False) / z.shape[1]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                z_val = net(val_data)
                val_loss = (
                    0.5 * torch.mean(z ** 2)
                    - torch.mean(net.log_jacobian(run_forward=False)) / z_val.shape[1]
                )
                losses.append(loss.item())
                val_losses.append(val_loss.item())

                z_draw = z_val.detach().numpy()
                line.set_data(z_draw[:, 0], z_draw[:, 1])

                fig.canvas.draw()
                pp.draw()
                pp.pause(0.0001)

                progress.set_postfix(
                    loss=f"{loss.item():8.3f}", val_loss=f"{val_loss.item():8.3f}"
                )


samples = torch.normal(0, 1, size=(2048, z.shape[1]))
x = net(samples, rev=True)
x = x.detach().numpy()
x = x.reshape(x.shape[0], -1, 3)
t.unitcell_lengths = None
t.unitcell_angles = None
t.xyz = x
t.save("out.pdb")
pp.ioff()
pp.figure()
pp.plot(losses)
pp.plot(val_losses)
pp.show()
