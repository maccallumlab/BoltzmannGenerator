import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import internal_coord
import pca
import numpy as np
import mdtraj as md
from tqdm import tqdm
from matplotlib import pyplot as pp
import matplotlib

matplotlib.use("Qt4Agg")
pp.ion()

t = md.load("5ura_traj.dcd", top="5ura_start.pdb")
ind = t.top.select("backbone")
t = t.atom_slice(ind)
t.center_coordinates()
ind_super = t.top.select("name N CA C O and resid 35")
t.superpose(t, 0, atom_indices=ind_super, ref_atom_indices=ind_super)
t.save("in.pdb")
training_data = t.xyz
n_atoms = training_data.shape[1]
n_dim = n_atoms * 3
training_data_npy = training_data.reshape(-1, n_dim)
training_data = torch.from_numpy(training_data_npy.astype("float32"))

graph = []
for i in range(3, n_atoms):
    graph.append((i, (i - 1, i - 2, i - 3)))


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

        return nn.Sequential(lin1, nn.ReLU(), lin2, nn.ReLU(), lin3, nn.ReLU(), lin4)


# Build the network
nodes = [Ff.InputNode(n_dim, name="input")]
ic_node = Ff.Node(
    nodes[-1],
    internal_coord.InternalCoordinateTransform,
    {"training_data": training_data, "z_indices": graph, "cart_indices": [0, 1, 2]},
    name="internal",
)
split_node = Ff.Node(
    ic_node,
    Fm.Split1D,
    {"split_size_or_sections": [9, n_dim - 9], "dim": 0},
    name="split",
)

pca_node = Ff.Node(
    split_node.out0, pca.PCA, {"training_data": training_data[:, :9]}, name="pca"
)

merge_node = Ff.Node(
    [split_node.out1, pca_node.out0], Fm.Concat1d, {"dim": 0}, name="merge"
)

nodes.extend([ic_node, split_node, pca_node, merge_node])

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
n_val = n // 10
np.random.shuffle(training_data_npy)

val_data = torch.as_tensor(training_data_npy[:n_val, :])
train_data = torch.as_tensor(training_data_npy[n_val:, :])
I = np.arange(train_data.shape[0])  # A list of indices into the training set

learning_rate = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Setup stuff for animation
fig = pp.figure()
ax = pp.axes(xlim=(-8, 8), ylim=(-8, 8))
line, = ax.plot([], [], linestyle="None", marker="o")

pp.show(False)
pp.draw()

with tqdm(range(epochs)) as progress:
    for epoch in progress:
        net.train()
        index_batch = np.random.choice(I, n_batch, replace=True)
        x_batch = train_data[index_batch, :]

        z = net(x_batch)

        loss = 0.5 * torch.mean(torch.sum(z ** 2, axis=1)) - torch.mean(
            net.log_jacobian(run_forward=False)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                z_val = net(val_data)
                val_loss = 0.5 * torch.mean(torch.sum(z_val ** 2, axis=1)) - torch.mean(
                    net.log_jacobian(run_forward=False)
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
