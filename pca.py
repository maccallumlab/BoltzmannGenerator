import torch
import torch.nn as nn
import math


class PCA(nn.Module):
    def __init__(self, dims_in, training_data=None):
        super().__init__()

        # do some sanity checks on our input connections
        assert len(dims_in) == 1, "PCA can only use one input"
        assert len(dims_in[0]) == 1, "PCA can only use one input channel"
        self.dims = dims_in[0][0]

        # compute our whiten / blackening matrices, etc
        training_data = torch.as_tensor(training_data)
        self._validate_training_data(training_data)
        self._compute_decomp(training_data)

    def forward(self, x, rev=False):
        x = x[0].unsqueeze(2)
        if rev:
            x = torch.matmul(self.blacken, x).squeeze(2)
            x = x + self.means.unsqueeze(0).expand_as(x)
            return [x]
        else:
            x = x - self.means.unsqueeze(1).expand_as(x)
            return [torch.matmul(self.whiten, x).squeeze(2)]

    def jacobian(self, x, rev=False):
        if rev:
            return self.jac.expand(x[0].shape[0])
        else:
            return -self.jac.expand(x[0].shape[0])

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "PCA can only use one input"
        assert len(input_dims[0]) == 1, "PCA can only use one input channel"
        in_dim = input_dims[0][0]
        assert in_dim > 6, "PCA removes 6 dof, so requires at least 7 input dimensions."
        return [(in_dim - 6,)]

    def _validate_training_data(self, training_data):
        if training_data is None:
            raise ValueError("PCA must be supplied with training_data.")

        if len(training_data.shape) != 2:
            raise ValueError("training_data must be n_samples x n_dim array")

        n_samp = training_data.shape[0]
        n_dim = training_data.shape[1]

        if n_dim != self.dims:
            raise ValueError(
                f"training_data must have {self.dims} dimensions, not {n_dim}."
            )

        if not n_samp >= n_dim:
            raise ValueError("training_data must have n_samp >= n_dim")

    def _compute_decomp(self, training_data):
        with torch.no_grad():
            # mean center the data
            means = torch.mean(training_data, 0)
            self.register_buffer("means", means)
            training_data = training_data - self.means.expand_as(training_data)

            # do the SVD
            U, S, V = torch.svd(training_data)

            # All eigenvalues should be positive.
            if torch.any(S[:-6] <= 0):
                raise RuntimeError("All eigenvalues should be positive.")

            # Throw away the last 6 dof.
            stds = S[:-6] / math.sqrt(training_data.shape[0] - 1)
            self.register_buffer("stds", stds)
            V = V[:, :-6]

            # Store the jacobian for later.
            jac = torch.sum(torch.log(self.stds))
            self.register_buffer("jac", jac)

            # Store the whitening / blackening matrices for later.
            # The unsqueeze(0) adds a dummy leading dimension, which 
            # allows us to matrix multiply by a batch of samples.
            whiten = (torch.diag(1.0 / self.stds) @ V.t()).unsqueeze(0)
            blacken = (V @ torch.diag(self.stds)).unsqueeze(0)
            self.register_buffer("whiten", whiten)
            self.register_buffer("blacken", blacken)

