import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import mixed_transform
import numpy as np
import mdtraj as md
from tqdm import tqdm
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--iterations", type=int, default=1000)
args = parser.parse_args()

# USe CUDA if available.
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using cuda")
else:
    device = torch.device("cpu")
    print("Using cpu")


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
            lin1,
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(hidden),
            lin2,
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(hidden),
            lin3,
            nn.ReLU(),
            lin4,
        )


#
# Main script starts here
#
print("Loading trajectory")
t = md.load("5ura_traj.dcd", top="5ura_start.pdb")

print("Processing trajectory")
# center everything
t.center_coordinates()

# superpose on the backbone
ind = t.top.select("backbone")
t.superpose(t, 0, atom_indices=ind, ref_atom_indices=ind)

print("Saving processed trajectory")
# save input coordinates for reference
t[::10].save("in.pdb")

# Gather the training data into a pytorch Tensor with the right shape
training_data = t.xyz
n_atoms = training_data.shape[1]
n_dim = n_atoms * 3
training_data_npy = training_data.reshape(-1, n_dim)
training_data = torch.as_tensor(training_data_npy, dtype=torch.float32)

#
# Build the network
#
print("Building network")

# First create the input node
nodes = [Ff.InputNode(n_dim, name="input")]

# Next create the node for internal coordinates
mixed_nodes = mixed_transform.build_mixed_transformation_layers(
    nodes[-1], training_data, t.topology, "backbone"
)
nodes.extend(mixed_nodes)

n_glow = 8
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
net = net.to(device=device)

print("Training")
#
# Training
#
losses = []
val_losses = []
epochs = args.iterations
n_batch = args.batch_size

writer = SummaryWriter()

n = training_data_npy.shape[0]
n_val = n // 10
np.random.shuffle(training_data_npy)

val_data = torch.as_tensor(
    training_data_npy[:n_val, :], device=device, dtype=torch.float32
)
train_data = torch.as_tensor(
    training_data_npy[n_val:, :], device=device, dtype=torch.float32
)

I = np.arange(train_data.shape[0])  # A list of indices into the training set
Ival = np.arange(val_data.shape[0])

learning_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=40, verbose=True
)

with tqdm(range(epochs)) as progress:
    for epoch in progress:
        net.train()
        index_batch = np.random.choice(I, n_batch, replace=True)
        x_batch = train_data[index_batch, :]

        z = net(x_batch)

        loss = (
            0.5 * torch.mean(z ** 2) - torch.mean(net.log_jacobian(run_forward=False))
        ) / z.shape[1]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                index_val = np.random.choice(Ival, n_batch, replace=True)
                x_val = val_data[index_val, :]

                z_val = net(x_val)
                val_loss = (
                    0.5 * torch.mean(z_val ** 2)
                    - torch.mean(net.log_jacobian(run_forward=False))
                ) / z_val.shape[1]

                losses.append(loss.item())
                val_losses.append(val_loss.item())

                writer.add_scalar("Loss/train", loss.item(), epoch)
                writer.add_scalar("Loss/validation", val_loss.item(), epoch)
                writer.add_scalars(
                    "learning_rates",
                    {
                        str(i): math.log(group["lr"], 10)
                        for i, group in enumerate(optimizer.param_groups)
                    },
                    epoch,
                )

                progress.set_postfix(
                    loss=f"{loss.item():8.3f}", val_loss=f"{val_loss.item():8.3f}"
                )

with torch.no_grad():
    net.eval()
    samples = torch.normal(0, 1, size=(128, z.shape[1]), device=device)
    x = net(samples, rev=True)
    x = x.cpu().detach().numpy()
x = x.reshape(x.shape[0], -1, 3)
t.unitcell_lengths = None
t.unitcell_angles = None
t.xyz = x
t.save("out.pdb")
