import numpy as np
from torch import distributions
import torch


class ParticleFilter:
    def __init__(
        self, device, net, data, res_size, batch_size, energy_evaluator, step_size=1e-3
    ):
        self.device = device
        self.net = net
        self.res_size = res_size
        self.n_dim = data.shape[1]
        self.indices = np.arange(self.res_size)

        if data.shape[0] != self.res_size:
            sampled = np.random.choice(np.arange(data.shape[0]), size=self.res_size, replace=True)
            self.reservoir = data[sampled, :]
        else:
            self.reservoir = data

        self.batch_size = batch_size
        self.energy_evaluator = energy_evaluator
        self.step_size = step_size

        mu = torch.zeros(self.reservoir.shape[-1] - 6, device=device)
        cov = torch.eye(self.reservoir.shape[-1] - 6, device=device)
        self.latent_distribution = distributions.MultivariateNormal(
            mu, covariance_matrix=cov
        ).expand((batch_size,))

        # These statistics are updated during training.
        self.forward_loss = None
        self.forward_ml = None
        self.forward_jac = None
        self.inverse_loss = None
        self.inverse_kl = None
        self.inverse_jac = None
        self.mean_energy = None
        self.median_energy = None
        self.min_energy = None
        self.acceptance_probs = []

    def sample_and_compute_losses(self):
        # Choose a random batch of configurations and convert to latent.
        samples = torch.from_numpy(np.random.choice(self.indices, size=self.batch_size, replace=False))
        x = self.reservoir[samples, :].to(self.device)
        z, jac_forward = self.net.forward(x)

        # Choose a random batch of latents and convert to configurations.
        z_prime = self.latent_distribution.sample().to(self.device)
        x_prime, jac_inverse = self.net.inverse(z_prime)

        # Calculate training by example loss.
        self.forward_ml = -torch.mean(self.latent_distribution.log_prob(z))
        self.forward_jac = -torch.mean(jac_forward)
        self.forward_loss = self.forward_ml + self.forward_jac

        # Calculate training by energy loss.
        energies = self.energy_evaluator(x_prime)
        self.min_energy = torch.min(energies)
        self.median_energy = torch.median(energies)
        self.mean_energy = torch.mean(energies)
        self.inverse_kl = torch.mean(energies)
        self.inverse_jac = -torch.mean(jac_inverse)
        self.inverse_loss = self.inverse_kl + self.inverse_jac

        # Generate proposals by making perturbations to latent vectors.
        with torch.no_grad():
            z_prop = z + torch.normal(
                0, self.step_size, size=z.shape, device=self.device
            )
            x_prop, prop_jac = self.net.inverse(z_prop)
            prop_energies = self.energy_evaluator(x_prop)
            old_energies = self.energy_evaluator(x)
            logw_old = old_energies.squeeze(dim=1) + jac_forward
            logw_new = prop_energies.squeeze(dim=1) - prop_jac

        # accept or reject
        with torch.no_grad():
            metrop = -torch.log(torch.rand(self.batch_size, device=self.device))
            accepted = torch.gt(metrop, logw_new - logw_old)
            print("old_energy", old_energies[0, :].cpu())
            print("prop_energy", prop_energies[0, :].cpu())
            print("logw_new", logw_new[0].cpu())
            print("logw_old", logw_old[0].cpu())
            print("delta_logw", logw_new[0].cpu() - logw_old[0].cpu())
            print(metrop[0].cpu())
            print(accepted[0].cpu())
            print(samples[accepted])
            print(x_prop[accepted, :].shape)
            print(self.reservoir[samples[accepted], :].shape)
            self.acceptance_probs.append(torch.sum(accepted.float()).item() / self.batch_size)
            print(self.acceptance_probs[-1], self.step_size)
            self.reservoir[samples[accepted], :] = x_prop[accepted, :].cpu()

        # update step size
        if len(self.acceptance_probs) >= 50:
            mean = np.mean(self.acceptance_probs[-50:])
            if mean < 0.10:
                self.step_size = max(1e-8, self.step_size * 0.98)
            elif mean > 0.4:
                self.step_size = min(1, self.step_size * 1.02)
