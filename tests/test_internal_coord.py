import torch
import torch.nn as nn
from boltzmann.protein import InternalCoordinateTransform
import numpy as np
import math

z_indices = [(3, [2, 1, 0]), (4, [2, 1, 0]), (5, [3, 2, 1])]

coords = np.array(
    [
        # 0000000000000   1111111111111  2222222222222  3333333333333  4444444444444  5555555555555
        [-1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ],
    dtype="float32",
)
coords = torch.as_tensor(coords)
n_dim = coords.shape[1]

transform = InternalCoordinateTransform(
    dims=n_dim,
    training_data=coords,
    z_indices=z_indices,
    cart_indices=[0, 1, 2],
)

z, jac_f = transform.forward(coords)

# We should only change the last 9 coordinates
assert torch.allclose(coords[:, :-9], z[:, :-9])

# We should be able to undo the transformation
x, jac_r = transform.inverse(z)
assert torch.allclose(coords, x, atol=1e-6)
assert torch.allclose(jac_f, -1 * jac_r, atol=1e-6)
