from simtk import openmm as mm
from simtk.openmm import app
import torch


class OpenMMEnergyAdaptor(torch.autograd.function):
    @staticmethod
    def forward(ctx, input, openmm_context):
        # Save the openmm context and input for the
        # reverse step.
        ctx.openmm_context = openmm_context
        ctx.save_for_backward(input)

        n_batch = input.shape[0]
        n_dim = input.shape[1]
        energies = torch.zeros((n_batch, 1))
        forces = torch.zeros((n_batch, n_dim))

        input = input.cpu().detach().numpy()
        for i in range(n_batch):
            x = input[i, :].reshape(-1, 3)
            # beam up to openmm
            # get energy
            # get forces
        # store forces in ctx
        return energies.to(device=input.device)

        @staticmethod
        def backward(ctx, grad_output):
            input, forces = ctx.saved_tensors
            openmm_context = ctx.openmm_context

            forces = forces.to(device=input.device)
            # multiply forces by grad_output
            return forces, None

openmm_energy = OpenMMEnergyAdaptor.apply
