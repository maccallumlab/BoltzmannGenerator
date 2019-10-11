import torch
from boltzmann.protein import pca
from boltzmann.generative import transforms
from boltzmann import nn
from boltzmann import utils
import numpy as np
import mdtraj as md
from tqdm import tqdm
import time

# Use the GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading trajectory")
t = md.load("../data/AAAA_noconstraints.dcd", top="../data/AAAA.pdb")
t.center_coordinates()
t.superpose(t, 0)
training_data = t.xyz
n_dim = training_data.shape[1] * 3
training_data_npy = training_data.reshape(-1, n_dim)
training_data = torch.from_numpy(training_data_npy.astype("float32"))
print("Trajectory loaded")
print("Data has size:", training_data.shape)


# Build the network
N_COUPLING = 4
AFFINE_LAYER = False
layers = []
layers.append(pca.PCA(n_dim, training_data))
for _ in range(N_COUPLING):
    p = transforms.RandomPermutation(n_dim - 6, 1)
    mask_even = utils.create_alternating_binary_mask(features=n_dim - 6, even=True)
    mask_odd = utils.create_alternating_binary_mask(features=n_dim - 6, even=False)
    if AFFINE_LAYER:
        t1 = transforms.AffineCouplingTransform(
            mask=mask_even,
            transform_net_create_fn=lambda in_features, out_features: nn.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=128,
                num_blocks=3,
                dropout_probability=0.5,
                use_batch_norm=True,
            ),
        )
        t2 = transforms.AffineCouplingTransform(
            mask=mask_odd,
            transform_net_create_fn=lambda in_features, out_features: nn.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=128,
                num_blocks=3,
                dropout_probability=0.5,
                use_batch_norm=True,
            ),
        )
    else:
        t1 = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask_even,
            transform_net_create_fn=lambda in_features, out_features: nn.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=128,
                num_blocks=3,
                dropout_probability=0.5,
                use_batch_norm=True,
            ),
            tails="linear",
            tail_bound=5,
            num_bins=8,
            apply_unconditional_transform=False,
        )
        t2 = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask_odd,
            transform_net_create_fn=lambda in_features, out_features: nn.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=128,
                num_blocks=3,
                dropout_probability=0.5,
                use_batch_norm=True,
            ),
            tails="linear",
            tail_bound=5,
            num_bins=8,
            apply_unconditional_transform=False,
        )
    layers.append(p)
    layers.append(t1)
    layers.append(t2)

# build layers

net = transforms.CompositeTransform(layers).to(device)

losses = []
val_losses = []
epochs = 10000
n_batch = 1024  # This is the number of data points per batch

n = training_data_npy.shape[0]
n_val = 128
np.random.shuffle(training_data_npy)

val_data = torch.as_tensor(training_data_npy[:n_val, :], device=device)
train_data = torch.as_tensor(training_data_npy[n_val:, :], device=device)
I = np.arange(train_data.shape[0])  # A list of indices into the training set

learning_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-2)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, min_lr=1e-8, verbose=True)

# We're going choose a random latent vector and see what it transforms to
# as we train the network.

trained_coords = []
z_fixed = torch.normal(0, 1, size=(1, n_dim - 6), device=device)

with tqdm(range(epochs)) as progress:
    for epoch in progress:
        net.train()
        optimizer.zero_grad()
        index_batch = np.random.choice(I, n_batch, replace=True)
        x_batch = train_data[index_batch, :]
        x_batch = x_batch + torch.normal(0, 1e-3, size=x_batch.shape, device=x_batch.device)

        z, jac = net.forward(x_batch)

        loss = 0.5 * torch.mean(torch.sum(z ** 2, dim=1)) - torch.mean(jac)
        loss.backward()
        optimizer.step()
        # scheduler.step(epoch)

        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                z_val, z_jac = net(val_data)
                val_loss = 0.5 * torch.mean(torch.sum(z_val ** 2, dim=1)) - torch.mean(
                    z_jac
                )
                losses.append(loss.item())
                val_losses.append(val_loss.item())
                scheduler.step(val_loss.item())

                progress.set_postfix(
                    loss=f"{loss.item():8.3f}", val_loss=f"{val_loss.item():8.3f}"
                )

                x_fixed, _ = net.inverse(z_fixed)
                trained_coords.append(x_fixed.cpu().detach().numpy())

trained_coords = np.array(trained_coords)
trained_coords = trained_coords.reshape(trained_coords.shape[0], -1, 3)
t.unitcell_lengths = None
t.unitcell_angles = None
t.xyz = trained_coords
t.save("out.pdb")
