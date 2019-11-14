import torch
import torch.nn as nn
import numpy as np
import mdtraj as md
from boltzmann import protein

from simtk.openmm import app
import simtk.openmm as mm


PDB_PATH = "../data/AIYFL.pdb"
DCD_PATH = "../data/AIYFL.dcd"
TEMPERATURE = 298.0
TOLERANCE = 3e-2
N = 1024


def get_openmm_context(pdb_path):
    pdb = app.PDBFile(pdb_path)
    ff = app.ForceField("amber99sbildn.xml", "amber99_obc.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.0,
        constraints=None,
    )
    integrator = mm.LangevinIntegrator(298, 1.0, 0.002)
    simulation = app.Simulation(pdb.topology, system, integrator)
    context = simulation.context
    return context


def get_energy_evaluator(openmm_context, temperature, energy_high, energy_max, device):
    energy_high = torch.tensor(
        energy_high, dtype=torch.float32, device=device, requires_grad=False
    )
    energy_max = torch.tensor(
        energy_max, dtype=torch.float32, device=device, requires_grad=False
    )

    def eval_energy(x):
        return protein.regularize_energy(
            protein.openmm_energy(x, openmm_context, temperature),
            energy_high,
            energy_max,
        )

    return eval_energy


t = md.load(DCD_PATH, top=PDB_PATH)

# center everything
t.center_coordinates()

# superpose on the backbone
ind = t.top.select("backbone")
t.superpose(t, 0, atom_indices=ind, ref_atom_indices=ind)

# Gather the training data into a pytorch Tensor with the right shape
training_data = t.xyz
n_atoms = training_data.shape[1]
n_dim = n_atoms * 3
training_data_npy = training_data.reshape(-1, n_dim)
training_data = torch.from_numpy(training_data_npy.astype("float32"))

#
# Build the network
#
pca_block = protein.PCABlock("backbone", True)
transform = protein.MixedTransform(n_dim, t.topology, [pca_block], training_data)

#
# Get an openmm context and energy evaluator
#
context = get_openmm_context(PDB_PATH)
evaluator = get_energy_evaluator(context, TEMPERATURE, 1e10, 1e-20, torch.device("cpu"))

#
# Test
#
with torch.no_grad():
    # Run some samples through the network forward
    x = training_data[:N, :]
    z_samples, jac_f = transform.forward(x)

    # Run them back through the network in reverse
    x_prime, jac_r = transform.inverse(z_samples)

    energies_before = evaluator(x)
    energies_after = evaluator(x_prime)

assert torch.allclose(
    jac_f, -1 * jac_r, atol=0.1
), "Jacobians in forward and reverse directions do not match"

assert torch.allclose(x_prime, x, atol=1e-3), "Coordinates do not match"

if not torch.allclose(
    energies_before - energies_after, torch.zeros_like(energies_before), atol=TOLERANCE
):
    max_error = torch.max(torch.abs(energies_before - energies_after))
    print()
    print(
        f"Energies do not match to within {TOLERANCE} over {N} samples. Largest discrepancy is {max_error}."
    )
    print()
    raise AssertionError("Energies do not match.")
