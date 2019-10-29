import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch import distributions
from boltzmann import protein
from boltzmann.generative import transforms
from boltzmann import nn
from boltzmann import utils
from simtk import openmm as mm
from simtk.openmm import app
import numpy as np
import mdtraj as md
import os
import shutil
import argparse
from tqdm import tqdm


#
# Command line and logging
#


def parse_args():
    parser = argparse.ArgumentParser(
        prog="train.py", description="Train generative model of molecular conformation."
    )

    path_group = parser.add_argument_group("paths and filenames")
    # Paths and filenames
    path_group.add_argument("--pdb-path", required=True, help="path to pdb file")
    path_group.add_argument("--dcd-path", required=True, help="path to dcd file")
    path_group.add_argument("--output-name", required=True, help="base name for output")
    path_group.add_argument(
        "--overwrite", action="store_true", help="overwrite previous run"
    )
    path_group.set_defaults(overwrite=False)

    # Optimization parameters
    optimizer_group = parser.add_argument_group("optimization parameters")
    optimizer_group.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="number of training iterations (default: %(default)d)",
    )
    optimizer_group.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="size of training batch (default: %(default)d)",
    )
    optimizer_group.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="gradually raise learning rate over first WARMUP_EPOCHS (default: %(default)d)",
    )
    optimizer_group.add_argument(
        "--warmup-factor",
        type=float,
        default=1000,
        help="learning rate starts WARMUP_FACTOR below init-lr (default: %(default)d)",
    )
    optimizer_group.add_argument(
        "--init-lr",
        type=float,
        default=1e-3,
        help="initial learning rate (default: %(default)g)",
    )
    optimizer_group.add_argument(
        "--final-lr",
        type=float,
        default=1e-5,
        help="final learning rate (default: %(default)g)",
    )
    optimizer_group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="strength of weight decay (default: %(default)g)",
    )
    optimizer_group.add_argument(
        "--dropout-fraction",
        type=float,
        default=0.5,
        help="strength of dropout (default: %(default)g)",
    )
    optimizer_group.add_argument(
        "--max-gradient",
        type=float,
        default=100.0,
        help="maximum allowed gradient (default: %(default)g)",
    )
    optimizer_group.add_argument(
        "--log-freq",
        type=int,
        default=10,
        help="how often to update tensorboard (default: %(default)d)",
    )
    optimizer_group.add_argument(
        "--fold-validation",
        type=float,
        default=10.0,
        help="how much data to set aside for training (default: %(default)d)",
    )

    pretrans_group = parser.add_argument_group("pretransformation parameters")
    pretrans_group.add_argument(
        "--pretrans-type",
        default="quad-cdf",
        choices=["quad-cdf", "none"],
        help="pre-transform inputs before neural network (default: %(default)s)",
    )
    pretrans_group.add_argument(
        "--pretrans-epochs",
        type=int,
        default=500,
        help="number of training epochs for pre-transformation layer (default: %(default)d)",
    )
    pretrans_group.add_argument(
        "--pretrans-lr",
        type=float,
        default=1e-2,
        help="learning rate for pretransform training (default: %(default)g)",
    )

    # Network parameters
    network_group = parser.add_argument_group("network parameters")
    network_group.add_argument(
        "--load-network", default=None, help="load previously trained network"
    )
    network_group.add_argument(
        "--model-type",
        default="nsf-coupling",
        choices=[
            "affine-coupling",
            "affine-made",
            "nsf-unconditional",
            "nsf-coupling",
            "nsf-made",
        ],
        help="type of model (default: %(default)s)",
    )
    network_group.add_argument(
        "--coupling-layers",
        type=int,
        default=4,
        help="number of coupling layers (default: %(default)d)",
    )
    network_group.add_argument(
        "--hidden-features",
        type=int,
        default=128,
        help="number of hidden features in each layer (default: %(default)d)",
    )
    network_group.add_argument(
        "--hidden-layers",
        type=int,
        default=2,
        help="number of hidden layers (default: %(default)d)",
    )
    network_group.add_argument(
        "--spline-points",
        type=int,
        default=8,
        help="number of spline points in NSF layers (default: %(default)d)",
    )

    # Loss Function parameters
    loss_group = parser.add_argument_group("loss function parameters")
    loss_group.add_argument(
        "--training-type",
        default="example",
        choices=["example", "energy", "mixed"],
        help="type of loss function ot use (default: %(default)s)",
    )
    loss_group.add_argument(
        "--example-weight",
        type=float,
        default=1.0,
        help="weight for training by example (default: %(default)g)",
    )
    loss_group.add_argument(
        "--energy-weight",
        type=float,
        default=1.0,
        help="weight for training by energy (default: %(default)g)",
    )

    # Energy evaluation parameters
    energy_group = parser.add_argument_group("parameters for energy function")
    energy_group.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help="temperature (default: %(default)g)",
    )
    energy_group.add_argument(
        "--energy-max",
        type=float,
        default=1e20,
        help="maximum energy (default: %(default)g)",
    )
    energy_group.add_argument(
        "--energy-high",
        type=float,
        default=1e10,
        help="log transform energies above this value (default: %(default)g)",
    )

    args = parser.parse_args()
    if args.training_type == "example":
        args.train_example = True
        args.train_energy = False
    elif args.training_type == "energy":
        args.train_example = False
        args.train_energy = True
    elif args.training_type == "mixed":
        args.train_example = True
        args.train_energy = True
    else:
        raise NotImplementedError()

    return args


def setup_writers(args):
    train_writer = SummaryWriter(
        log_dir=f"runs/{args.output_name}_train", purge_step=0, flush_secs=30
    )
    test_writer = SummaryWriter(
        log_dir=f"runs/{args.output_name}_test", purge_step=0, flush_secs=30
    )
    setup_custom_scalars(args, train_writer)
    setup_custom_scalars(args, test_writer)
    return train_writer, test_writer


def setup_custom_scalars(args, writer):
    writer.add_custom_scalars(
        {
            "total_losses": {
                "total_loss": ["Multiline", ["total_loss", "total_loss"]],
                "energy_loss": [
                    "Multiline",
                    ["energy_total_loss", "energy_total_loss"],
                ],
                "example_loss": [
                    "Multiline",
                    ["example_total_loss", "example_total_loss"],
                ],
                "gradient_norm": ["Multiline", ["gradient_norm"]],
            },
            "example_losses": {
                "total": ["Multiline", ["example_total_loss", "example_total_loss"]],
                "ml": ["Multiline", ["example_ml_loss", "example_ml_loss"]],
                "jac": ["Multiline", ["example_jac_loss", "example_jac_loss"]],
            },
            "energy_losses": {
                "total": ["Multiline", ["energy_total_loss", "energy_total_loss"]],
                "ml": ["Multiline", ["energy_kl_loss", "energy_kl_loss"]],
                "jac": ["Multiline", ["energy_jac_loss", "energy_jac_loss"]],
            },
            "energies": {
                "minimum": ["Multiline", ["minimum_energy"]],
                "mean": ["Multiline", ["mean_energy"]],
                "median": ["Multiline", ["median_energy"]],
            },
        }
    )


def write_final_stats(
    args, loss, example_loss, energy_loss, example_train, energy_train
):
    writer = SummaryWriter(
        log_dir=f"results/{args.output_name}_results", purge_step=0, flush_secs=30
    )
    h_params = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "dropout_prob": args.dropout_fraction,
        "weight_decay": args.weight_decay,
        "init_lr": args.init_lr,
        "final_lr": args.final_lr,
        "wamup_epochs": args.warmup_epochs,
        "warmup_factor": args.warmup_factor,
        "max_gradient": args.max_gradient,
        "coupling_layers": args.coupling_layers,
        "model_type": args.model_type,
        "spline_points": args.spline_points,
        "hidden_features": args.hidden_features,
        "hidden_layers": args.hidden_layers,
        "train_example": args.train_example,
        "example_weight": args.example_weight,
        "train_energy": args.train_energy,
        "energy_weight": args.energy_weight,
        "energy_max": args.energy_max,
        "energy_high": args.energy_high,
    }

    metrics = {
        "validation_loss": loss,
        "energy_loss": energy_loss,
        "example_loss": example_loss,
    }
    if args.train_example:
        metrics["example_loss_train"] = example_train
    if args.train_energy:
        metrics["energy_loss_train"] = energy_train

    writer.add_hparams(h_params, metrics)


#
# File input / output
#


def delete_run(name):
    if os.path.exists(f"models/{name}.pkl"):
        os.remove(f"models/{name}.pkl")
    if os.path.exists(f"sample_traj/{name}.pdb"):
        os.remove(f"sample_traj/{name}.pdb")
    if os.path.exists(f"runs/{name}"):
        shutil.rmtree(f"runs/{name}")


def create_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("sample_traj", exist_ok=True)


def load_trajectory(pdb_path, dcd_path):
    print("Loading trajectory")
    t = md.load(args.dcd_path, top=args.pdb_path)
    ind = t.topology.select("backbone")
    t.superpose(t, frame=0, atom_indices=ind)
    return t


def load_network(path, device):
    net = torch.load(path).to(device)
    print(net)
    print_number_trainable_params(net)
    return net


#
# Build network
#


def build_affine_coupling(
    n_dim, n_coupling, hidden_layers, hidden_features, dropout_fraction
):
    layers = []
    for _ in range(n_coupling):
        p = transforms.RandomPermutation(n_dim, 1)
        mask_even = utils.create_alternating_binary_mask(features=n_dim, even=True)
        mask_odd = utils.create_alternating_binary_mask(features=n_dim, even=False)
        t1 = transforms.AffineCouplingTransform(
            mask=mask_even,
            transform_net_create_fn=lambda in_features, out_features: nn.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                num_blocks=hidden_layers,
                dropout_probability=dropout_fraction,
                use_batch_norm=False,
            ),
        )
        t2 = transforms.AffineCouplingTransform(
            mask=mask_odd,
            transform_net_create_fn=lambda in_features, out_features: nn.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                num_blocks=hidden_layers,
                dropout_probability=dropout_fraction,
                use_batch_norm=False,
            ),
        )
        layers.append(p)
        layers.append(t1)
        layers.append(t2)
    return layers


def build_affine_made(n_dim, hidden_layers, hidden_features, dropout_fraction):
    made = transforms.MaskedAffineAutoregressiveTransform(
        n_dim,
        hidden_features=hidden_features,
        num_blocks=hidden_layers,
        dropout_probability=dropout_fraction,
        use_batch_norm=False,
    )
    return [made]


def build_nsf_unconditional(n_dim, spline_points):
    nsf = transforms.PiecewiseQuadraticCDF(
        [n_dim], num_bins=spline_points, tails="linear", tail_bound=5
    )
    return [nsf]


def build_nsf_coupling(
    n_dim, n_coupling, spline_points, hidden_layers, hidden_features, dropout_fraction
):
    layers = []
    for _ in range(n_coupling):
        p = transforms.RandomPermutation(n_dim, 1)
        mask_even = utils.create_alternating_binary_mask(features=n_dim, even=True)
        mask_odd = utils.create_alternating_binary_mask(features=n_dim, even=False)
        t1 = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask_even,
            transform_net_create_fn=lambda in_features, out_features: nn.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                num_blocks=hidden_layers,
                dropout_probability=dropout_fraction,
                use_batch_norm=False,
            ),
            tails="linear",
            tail_bound=5,
            num_bins=spline_points,
            apply_unconditional_transform=False,
        )
        t2 = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask_odd,
            transform_net_create_fn=lambda in_features, out_features: nn.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                num_blocks=hidden_layers,
                dropout_probability=dropout_fraction,
                use_batch_norm=False,
            ),
            tails="linear",
            tail_bound=5,
            num_bins=spline_points,
            apply_unconditional_transform=False,
        )
        layers.append(p)
        layers.append(t1)
        layers.append(t2)
    return layers


def build_nsf_made(
    n_dim, spline_points, hidden_layers, hidden_features, dropout_fraction
):
    made = transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
        features=n_dim,
        hidden_features=hidden_features,
        num_blocks=hidden_layers,
        dropout_probability=dropout_fraction,
        use_batch_norm=False,
        num_bins=spline_points,
        tails="linear",
        tail_bound=5,
    )
    return [made]


def build_network(
    model_type,
    n_dim,
    topology,
    training_data,
    n_coupling,
    spline_points,
    hidden_features,
    hidden_layers,
    dropout_fraction,
    pretrans_type,
    pretrans_epochs,
    pretrans_lr,
    pretrans_batch_size,
    device,
):
    training_data = training_data.to(device)

    print("Creating network")
    layers = []

    # Create the mixed transofrm layer
    pca_block = protein.PCABlock("backbone", True)
    mixed = protein.MixedTransform(n_dim, topology, [pca_block], training_data)
    layers.append(mixed)

    if pretrans_type == "quad-cdf":
        print()
        print("Pre-training unconditional NSF layer")
        print()
        unconditional = build_nsf_unconditional(n_dim - 6, spline_points)[0]
        layers.append(unconditional)
        unconditional_net = transforms.CompositeTransform(layers).to(device)
        pre_train_unconditional_nsf(
            unconditional_net,
            device,
            training_data,
            pretrans_batch_size,
            pretrans_epochs,
            pretrans_lr,
            10,
        )
        print()
        print("Pretraining completed. Freezing weights")
        unconditional.unnormalized_heights.requires_grad_(False)
        unconditional.unnormalized_widths.requires_grad_(False)

    if model_type == "affine-coupling":
        new_layers = build_affine_coupling(
            n_dim - 6, n_coupling, hidden_layers, hidden_features, dropout_fraction
        )
    elif model_type == "affine-made":
        new_layers = build_affine_made(
            n_dim - 6, hidden_layers, hidden_features, dropout_fraction
        )
    elif model_type == "nsf-unconditional":
        new_layers = build_nsf_unconditional(n_dim - 6, spline_points)
    elif model_type == "nsf-coupling":
        new_layers = build_nsf_coupling(
            n_dim - 6,
            n_coupling,
            spline_points,
            hidden_layers,
            hidden_features,
            dropout_fraction,
        )
    elif model_type == "nsf-made":
        new_layers = build_nsf_made(
            n_dim - 6, spline_points, hidden_layers, hidden_features, dropout_fraction
        )
    else:
        raise RuntimeError()

    layers.extend(new_layers)
    net = transforms.CompositeTransform(layers).to(device)
    print()
    print("Network constructed.")
    print_number_trainable_params(net)
    print()
    return net


def print_number_trainable_params(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print()
    print(f"Network has {total_params} trainable parameters")
    print()


#
# Energy function
#


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


#
# Optimizer
#


def setup_optimizer(net, init_lr, weight_decay):
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=init_lr, weight_decay=weight_decay
    )
    return optimizer


def setup_scheduler(optimizer, init_lr, final_lr, epochs, warmup_epochs, warmup_factor):
    anneal = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, final_lr)
    warmup = utils.GradualWarmupScheduler(
        optimizer, warmup_factor, warmup_epochs, after_scheduler=anneal
    )
    return warmup


#
# Loss functions
#


def get_batch_weighted_ml_loss(net, x_batch, example_weight, dist):
    z, z_jac = net.forward(x_batch)

    ll = 0.5 * torch.sum(z ** 2, dim=1) - z_jac
    w = torch.exp(ll - torch.max(ll))
    w_total = torch.sum(w)

    example_ml_loss = (
        torch.sum(w * dist.log_prob(z)) / w_total * example_weight
    )
    example_jac_loss = -torch.sum(w * z_jac) / w_total * example_weight
    example_loss = example_ml_loss + example_jac_loss
    return example_loss, example_ml_loss, example_jac_loss


def get_ml_loss(net, x_batch, example_weight, dist):
    z, z_jac = net.forward(x_batch)

    example_ml_loss = -torch.mean(dist.log_prob(z)) * example_weight
    example_jac_loss = -torch.mean(z_jac) * example_weight
    example_loss = example_ml_loss + example_jac_loss
    return example_loss, example_ml_loss, example_jac_loss


def sample_energy_jac(net, device, energy_evaluator, dist):
    z_batch = dist.sample()
    x, x_jac = net.inverse(z_batch)
    energies = energy_evaluator(x)
    return energies, x_jac


def get_energy_loss(energies, jacobians, energy_weight):
    energy_kl_loss = torch.mean(energies) * energy_weight
    energy_jac_loss = -torch.mean(jacobians) * energy_weight
    energy_loss = energy_kl_loss + energy_jac_loss
    return energy_loss, energy_kl_loss, energy_jac_loss


#
# Training
#


def get_device():
    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    return device


def pre_train_unconditional_nsf(net, device, training_data, batch_size, epochs, lr, out_freq):
    mu = torch.zeros(training_data.shape[-1] - 6, device=device)
    cov = torch.eye(training_data.shape[-1] - 6, device=device)
    dist = distributions.MultivariateNormal(mu, covariance_matrix=cov).expand((batch_size, ))

    indices = np.arange(training_data.shape[0])
    optimizer = setup_optimizer(net, lr, 0.0)
    with tqdm(range(epochs)) as progress:
        for epoch in progress:
            net.train()

            index_batch = np.random.choice(indices, args.batch_size, replace=True)
            x_batch = training_data[index_batch, :]
            loss, _, _ = get_ml_loss(net, x_batch, args.example_weight, dist)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % out_freq == 0:
                progress.set_postfix(loss=f"{loss.item():8.3f}")


def run_training(args, device):
    train_writer, test_writer = setup_writers(args)

    traj = load_trajectory(args.pdb_path, args.dcd_path)
    n_dim = traj.xyz.shape[1] * 3
    training_data_npy = traj.xyz.reshape(-1, n_dim)
    training_data = torch.from_numpy(training_data_npy.astype("float32"))
    print("Trajectory loaded")
    print("Data has size:", training_data.shape)

    if args.load_network:
        net = load_network(f"models/{args.load_network}.pkl", device=device)
    else:
        net = build_network(
            n_dim=n_dim,
            model_type=args.model_type,
            topology=traj.topology,
            training_data=training_data,
            n_coupling=args.coupling_layers,
            spline_points=args.spline_points,
            hidden_features=args.hidden_features,
            hidden_layers=args.hidden_layers,
            dropout_fraction=args.dropout_fraction,
            pretrans_type=args.pretrans_type,
            pretrans_epochs=args.pretrans_epochs,
            pretrans_lr=args.pretrans_lr,
            pretrans_batch_size=args.batch_size,
            device=device,
        )

    optimizer = setup_optimizer(
        net=net,
        init_lr=args.init_lr / args.warmup_factor,
        weight_decay=args.weight_decay,
    )
    scheduler = setup_scheduler(
        optimizer,
        init_lr=args.init_lr,
        final_lr=args.final_lr,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_factor=args.warmup_factor,
    )

    openmm_context = get_openmm_context(args.pdb_path)
    energy_evaluator = get_energy_evaluator(
        openmm_context=openmm_context,
        temperature=args.temperature,
        energy_high=args.energy_high,
        energy_max=args.energy_max,
        device=device,
    )

    # Shuffle the training data
    n = training_data_npy.shape[0]
    n_val = int(n / args.fold_validation)
    np.random.shuffle(training_data_npy)

    # Split the training and validation sets
    val_data = torch.as_tensor(training_data_npy[:n_val, :], device=device)
    train_data = torch.as_tensor(training_data_npy[n_val:, :], device=device)
    indices = np.arange(train_data.shape[0])
    indices_val = np.arange(val_data.shape[0])

    mu = torch.zeros(train_data.shape[-1] - 6, device=device)
    cov = torch.eye(train_data.shape[-1] - 6, device=device)
    dist = distributions.MultivariateNormal(mu, covariance_matrix=cov).expand(
        (args.batch_size, )
    )

    with tqdm(range(args.epochs)) as progress:
        for epoch in progress:
            net.train()

            if args.train_example:
                index_batch = np.random.choice(indices, args.batch_size, replace=True)
                x_batch = train_data[index_batch, :]
                example_loss, example_ml_loss, example_jac_loss = get_ml_loss(
                    net, x_batch, args.example_weight, dist
                )

            if args.train_energy:
                energies, x_jac = sample_energy_jac(
                    net, device, energy_evaluator, dist
                )
                energy_loss, energy_kl_loss, energy_jac_loss = get_energy_loss(
                    energies, x_jac, args.energy_weight
                )

            if args.train_example and args.train_energy:
                loss = example_loss + energy_loss
            elif args.train_example:
                loss = example_loss
            else:
                loss = energy_loss

            optimizer.zero_grad()
            loss.backward()
            gradient_norm = clip_grad_norm_(net.parameters(), args.max_gradient)
            optimizer.step()
            scheduler.step(epoch)

            if epoch % args.log_freq == 0:
                net.eval()

                # Output our training losses
                train_writer.add_scalar("total_loss", loss.item(), epoch)
                train_writer.add_scalar("gradient_norm", gradient_norm, epoch)
                if args.train_example:
                    train_writer.add_scalar(
                        "example_ml_loss", example_ml_loss.item(), epoch
                    )
                    train_writer.add_scalar(
                        "example_jac_loss", example_jac_loss.item(), epoch
                    )
                    train_writer.add_scalar(
                        "example_total_loss", example_loss.item(), epoch
                    )
                if args.train_energy:
                    train_writer.add_scalar(
                        "energy_kl_loss", energy_kl_loss.item(), epoch
                    )
                    train_writer.add_scalar(
                        "energy_jac_loss", energy_jac_loss.item(), epoch
                    )
                    train_writer.add_scalar(
                        "energy_total_loss", energy_loss.item(), epoch
                    )

                # Compute our validation losses
                with torch.no_grad():
                    # Compute the example validation loss
                    index_val = np.random.choice(
                        indices_val, args.batch_size, replace=True
                    )
                    x_val = val_data[index_val, :]
                    example_loss_val, example_ml_loss_val, example_jac_loss_val = get_ml_loss(
                        net, x_val, args.example_weight, dist
                    )

                    # Compute the energy validation loss
                    val_energies, jac_val = sample_energy_jac(
                        net, device, energy_evaluator, dist
                    )
                    energy_loss_val, energy_kl_loss_val, energy_jac_loss_val = get_energy_loss(
                        val_energies, jac_val, args.energy_weight
                    )

                    # Compute the overall validation loss
                    if args.train_example and args.train_energy:
                        loss_val = example_loss_val + energy_loss_val
                    elif args.train_example:
                        loss_val = example_loss_val
                    else:
                        loss_val = energy_loss_val

                    progress.set_postfix(
                        loss=f"{loss.item():8.3f}", val_loss=f"{loss_val.item():8.3f}"
                    )

                    test_writer.add_scalar("total_loss", loss_val.item(), epoch)
                    test_writer.add_scalar(
                        "example_ml_loss", example_ml_loss_val.item(), epoch
                    )
                    test_writer.add_scalar(
                        "example_jac_loss", example_jac_loss_val.item(), epoch
                    )
                    test_writer.add_scalar(
                        "example_total_loss", example_loss_val.item(), epoch
                    )
                    test_writer.add_scalar(
                        "energy_kl_loss", energy_kl_loss_val.item(), epoch
                    )
                    test_writer.add_scalar(
                        "energy_jac_loss", energy_jac_loss_val.item(), epoch
                    )
                    test_writer.add_scalar(
                        "energy_total_loss", energy_loss_val.item(), epoch
                    )
                    test_writer.add_scalar(
                        "mean_energy", torch.mean(val_energies).item(), epoch
                    )
                    test_writer.add_scalar(
                        "median_energy", torch.median(val_energies).item(), epoch
                    )
                    test_writer.add_scalar(
                        "minimum_energy", torch.min(val_energies).item(), epoch
                    )

    # Save our final model
    torch.save(net, f"models/{args.output_name}.pkl")

    # Log our final losses to the console
    print("Final loss:", loss.item())
    if args.train_example:
        print("Final example loss:", example_loss.item())
    if args.train_energy:
        print("Final energy loss:", energy_loss.item())
    print("Final validation loss:", loss_val.item())
    print("Final validation example loss:", example_loss_val.item())
    print("Final validation energy loss:", energy_loss_val.item())

    # Write our final stats
    write_final_stats(
        args,
        loss_val.item(),
        example_loss_val.item(),
        energy_loss_val.item(),
        example_loss.item() if args.train_example else None,
        energy_loss.item() if args.train_energy else None,
    )

    # Generate examples and write trajectory
    net.eval()
    z = torch.normal(0, 1, size=(args.batch_size, n_dim - 6), device=device)
    x, _ = net.inverse(z)
    x = x.cpu().detach().numpy()
    x = x.reshape(args.batch_size, -1, 3)
    traj.unitcell_lengths = None
    traj.unitcell_angles = None
    traj.xyz = x
    traj.save(f"sample_traj/{args.output_name}.pdb")


if __name__ == "__main__":
    args = parse_args()
    if not (args.train_example or args.train_energy):
        raise RuntimeError(
            "You must specify at least one of train_example or train_energy."
        )

    model_path = f"models/{args.output_name}.pkl"
    if os.path.exists(model_path):
        if args.overwrite:
            print(f"Warning: output `{model_path}' already exists. Overwriting anyway.")
        else:
            raise RuntimeError(
                f"Output '{model_path}' already exists. If you're sure use --overwrite."
            )

    # Remove any old data for this run
    delete_run(args.output_name)

    create_dirs()
    device = get_device()
    run_training(args, device)
