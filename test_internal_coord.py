import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import internal_coord
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

nodes = [Ff.InputNode(n_dim, name="input")]
nodes.append(
    Ff.Node(
        nodes[-1],
        internal_coord.InternalCoordinateTransform,
        {"training_data": coords, "z_indices": z_indices, "cart_indices": [0, 1, 2]},
        name="internal",
    )
)
nodes.append(Ff.OutputNode(nodes[-1], name="output"))
net = Ff.ReversibleGraphNet(nodes, verbose=False)

z = net(coords)
jac_f = net.log_jacobian(run_forward=False)

# We should only change the last 9 coordinates
assert torch.allclose(coords[:, :-9], z[:, :-9])

# We should be able to undo the transformation
x = net(z, rev=True)
jac_r = net.log_jacobian(run_forward=False, rev=True)
assert torch.allclose(coords, x, atol=1e-6)
assert torch.allclose(jac_f, -1 * jac_r, atol=1e-6)
