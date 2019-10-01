import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import math


def calc_bonds(ix1, iy1, iz1, ix2, iy2, iz2, coords):
    return torch.sqrt(
        (coords[:, ix1] - coords[:, ix2]) ** 2
        + (coords[:, iy1] - coords[:, iy2]) ** 2
        + (coords[:, iz1] - coords[:, iz2]) ** 2
    )


def calc_angles(ix1, iy1, iz1, ix2, iy2, iz2, ix3, iy3, iz3, coords):
    b = coords[:, [ix1, iy1, iz1]]
    c = coords[:, [ix2, iy2, iz2]]
    d = coords[:, [ix3, iy3, iz3]]
    bc = b - c
    bc = bc / torch.norm(bc, dim=1, keepdim=True)
    cd = d - c
    cd = cd / torch.norm(cd, dim=1, keepdim=True)
    cos_angle = torch.sum(bc * cd, dim=1)
    angle = torch.acos(cos_angle)
    return angle


def calc_dihedrals(ix1, iy1, iz1, ix2, iy2, iz2, ix3, iy3, iz3, ix4, iy4, iz4, coords):
    a = coords[:, [ix1, iy1, iz1]]
    b = coords[:, [ix2, iy2, iz2]]
    c = coords[:, [ix3, iy3, iz3]]
    d = coords[:, [ix4, iy4, iz4]]

    b0 = a - b
    b1 = c - b
    b1 = b1 / torch.norm(b1, dim=1, keepdim=True)
    b2 = d - c

    v = b0 - torch.sum(b0 * b1, dim=1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=1, keepdim=True) * b1
    x = torch.sum(v * w, dim=1)
    b1xv = torch.cross(b1, v, dim=1)
    y = torch.sum(b1xv * w, dim=1)
    angle = torch.atan2(y, x)
    return -angle


def _indices_for_atom(index):
    return [3 * index, 3 * index + 1, 3 * index + 2]


class InternalCoordinateTransform(nn.Module):
    def __init__(self, dims_in, z_indices=None, training_data=None):
        super().__init__()

        # do some sanity checks on our input connections
        assert len(dims_in) == 1, "InternalCoordinateTransform can only use one input"
        assert (
            len(dims_in[0]) == 1
        ), "InternalCoordinateTransform can only use one input channel"
        self.dims = dims_in[0][0]
        self.jac = None

        self.sorted_z_indices = None
        self.modified_indices = None
        self.bond_indices = None
        self.angle_indices = None
        self.dih_indices = None
        self._setup_indices(z_indices)

        self._validate_training_data(training_data)

        self.mean_bonds = None
        self.std_bonds = None
        self.mean_angles = None
        self.std_angles = None
        self.mean_dih = None
        self.std_dih = None
        with torch.no_grad():
            self.jac = torch.zeros(training_data.shape[0])
            transformed = self._fwd(training_data)
            self._setup_mean_bonds(transformed)
            transformed[:, self.bond_indices] -= self.mean_bonds
            self._setup_std_bonds(transformed)
            transformed[:, self.bond_indices] /= self.std_bonds
            self._setup_mean_angles(transformed)
            transformed[:, self.angle_indices] -= self.mean_angles
            self._setup_std_angles(transformed)
            transformed[:, self.angle_indices] /= self.std_angles
            self._setup_mean_dih(transformed)
            transformed[:, self.dih_indices] -= self.mean_dih
            self._fix_dih(transformed)
            self._setup_std_dih(transformed)
            transformed[:, self.dih_indices] /= self.std_dih

    def forward(self, x, rev=False):
        x = x[0]
        self.jac = torch.zeros(x.shape[0])

        if rev:
            trans = x
            trans[:, self.bond_indices] *= self.std_bonds
            trans[:, self.bond_indices] += self.mean_bonds
            trans[:, self.angle_indices] *= self.std_angles
            trans[:, self.angle_indices] += self.mean_angles
            trans[:, self.dih_indices] *= self.std_dih
            trans[:, self.dih_indices] += self.mean_dih
            self._fix_dih(trans)
            trans = self._rev(trans)
        else:
            trans = self._fwd(x)
            trans[:, self.bond_indices] -= self.mean_bonds
            trans[:, self.bond_indices] /= self.std_bonds
            trans[:, self.angle_indices] -= self.mean_angles
            trans[:, self.angle_indices] /= self.std_angles
            trans[:, self.dih_indices] -= self.mean_dih
            self._fix_dih(trans)
            trans[:, self.dih_indices] /= self.std_dih
        return [trans]

    def jacobian(self, x, rev=False):
        if rev:
            return -1 * self.jac
        else:
            return self.jac

    def output_dims(self, input_dims):
        assert (
            len(input_dims) == 1
        ), "InternalCoordinateTransform can only use one input"
        assert (
            len(input_dims[0]) == 1
        ), "InternalCoordinateTransform can only use one input channel"
        return input_dims

    def _fwd(self, x):
        x = x.clone()
        for ind4, (ind1, ind2, ind3) in reversed(self.sorted_z_indices):
            # Get the cartesian indices for these atoms.
            inds1 = _indices_for_atom(ind1)
            inds2 = _indices_for_atom(ind2)
            inds3 = _indices_for_atom(ind3)
            inds4 = _indices_for_atom(ind4)

            # Calculate the bonds, angles, and torions for a batch.
            bonds = calc_bonds(*inds1, *inds4, coords=x)
            angles = calc_angles(*inds2, *inds1, *inds4, coords=x)
            dihedrals = calc_dihedrals(*inds3, *inds2, *inds1, *inds4, coords=x)

            self.jac += 2 * torch.log(bonds) + torch.log(torch.abs(torch.sin(angles)))

            # Replace the cartesian coordinates with internal coordinates.
            x[:, inds4[0]] = bonds
            x[:, inds4[1]] = angles
            x[:, inds4[2]] = dihedrals
        return x

    def _rev(self, x):
        x = x.clone()
        for ind4, (ind1, ind2, ind3) in self.sorted_z_indices:
            # Get the cartesian indices for these atoms.
            inds1 = _indices_for_atom(ind1)
            inds2 = _indices_for_atom(ind2)
            inds3 = _indices_for_atom(ind3)
            inds4 = _indices_for_atom(ind4)

            # Get the positions of the 4 reconstructing atoms
            p1 = x[:, inds1]
            p2 = x[:, inds2]
            p3 = x[:, inds3]

            # Get the distance, angle, and torsion
            d14 = x[:, inds4[0]].unsqueeze(1)
            a124 = x[:, inds4[1]].unsqueeze(1)
            t1234 = x[:, inds4[2]].unsqueeze(1)

            self.jac += 2 * torch.log(d14.squeeze()) + torch.log(
                torch.abs(torch.sin(a124.squeeze()))
            )

            # Reconstruct the position of p4
            v1 = p1 - p2
            v2 = p1 - p3

            n = torch.cross(v1, v2)
            n = n / torch.norm(n, dim=1, keepdim=True)
            nn = torch.cross(v1, n)
            nn = nn / torch.norm(nn, dim=1, keepdim=True)

            n = n * torch.sin(t1234)
            nn = nn * torch.cos(t1234)

            v3 = n + nn
            v3 = v3 / torch.norm(v3, dim=1, keepdim=True)
            v3 = v3 * d14 * torch.sin(a124)

            v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
            v1 = v1 * d14 * torch.cos(a124)

            # Store the final position in x
            position = p1 + v3 - v1
            x[:, inds4] = position
        return x

    def _setup_mean_bonds(self, x):
        self.mean_bonds = torch.mean(x[:, self.bond_indices], dim=0)

    def _setup_std_bonds(self, x):
        self.std_bonds = torch.std(x[:, self.bond_indices], dim=0) + 1e-3

    def _setup_mean_angles(self, x):
        self.mean_angles = torch.mean(x[:, self.angle_indices], dim=0)

    def _setup_std_angles(self, x):
        self.std_angles = torch.std(x[:, self.angle_indices], dim=0) + 1e-3

    def _setup_mean_dih(self, x):
        sin = torch.mean(torch.sin(x[:, self.dih_indices]), dim=0)
        cos = torch.mean(torch.cos(x[:, self.dih_indices]), dim=0)
        self.mean_dih = torch.atan2(sin, cos)

    def _fix_dih(self, x):
        dih = x[:, self.dih_indices]
        dih[dih < -math.pi] += 2 * math.pi
        dih[dih > math.pi] -= 2 * math.pi
        x[:, self.dih_indices] = dih

    def _setup_std_dih(self, x):
        self.std_dih = torch.std(x[:, self.dih_indices], dim=0) + 1e-3

    def _validate_training_data(self, training_data):
        if training_data is None:
            raise ValueError(
                "InternalCoordinateTransform must be supplied with training_data."
            )

        if len(training_data.shape) != 2:
            raise ValueError("training_data must be n_samples x n_dim array")

        n_samp = training_data.shape[0]
        n_dim = training_data.shape[1]

        if n_dim != self.dims:
            raise ValueError(
                f"training_data must have {self.dims} dimensions, not {n_dim}."
            )

        if not n_samp >= 1:
            raise ValueError("training_data must have n_samp > 1.")

    def _setup_indices(self, z_indices):
        self.sorted_z_indices = topological_sort(z_indices)
        modified_indices = [item[0] for item in self.sorted_z_indices]
        self.modified_indices = []
        for index in modified_indices:
            self.modified_indices.extend(_indices_for_atom(index))
        self.bond_indices = list(self.modified_indices[0::3])
        self.angle_indices = list(self.modified_indices[1::3])
        self.dih_indices = list(self.modified_indices[2::3])


def topological_sort(graph_unsorted):
    graph_sorted = []
    graph_unsorted = dict(graph_unsorted)

    while graph_unsorted:
        acyclic = False
        for node, edges in list(graph_unsorted.items()):
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append((node, edges))

        if not acyclic:
            raise RuntimeError("A cyclic dependency occured.")

    return graph_sorted
