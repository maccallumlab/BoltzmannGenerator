import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch import distributions
from boltzmann import protein
from boltzmann.generative import transforms
from boltzmann import nn
from boltzmann import utils
from boltzmann import training
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
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
    subparsers = parser.add_subparsers(dest="action")

    # Common io arguments
    io_parent_parser = argparse.ArgumentParser(add_help=False)
    io_parent_parser.add_argument("--save", required=True, help="basename for output")
    io_parent_parser.add_argument(
        "--overwrite", action="store_true", help="overwrite previous run"
    )
    io_parent_parser.set_defaults(overwrite=False)
    io_parent_parser.add_argument("--pdb-path", required=True, help="path to pdb file")
    io_parent_parser.add_argument(
        "--validation", required=True, help="validation dataset name"
    )

    #
    # Init parameters
    #
    init_parser = subparsers.add_parser(
        "init", help="initialize a new network", parents=[io_parent_parser]
    )

    # Init paths and filenames
    init_parser.add_argument("--dcd-path", required=True, help="path to dcd file")
    init_parser.add_argument(
        "--validation-fraction",
        default=0.05,
        type=float,
        help="fraction of dataset to use for validation (default: %(default)g)",
    )

    # Network parameters
    network_group = init_parser.add_argument_group("network parameters")
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
        default=8,
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
    network_group.add_argument(
        "--dropout-fraction",
        type=float,
        default=0.0,
        help="strength of dropout (default: %(default)g)",
    )
    network_group.add_argument(
        "--ensemble-size",
        type=int,
        default=100_000,
        help="size of configuration ensemble (default: %(default)d)",
    )

    # Pretrainsformation parameters
    pretrans_group = init_parser.add_argument_group("pretransformation parameters")
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
    pretrans_group.add_argument(
        "--pretrans-batch-size",
        type=int,
        default=1024,
        help="batch size for pretransformation training (default: %(default)g)",
    )

    # Noise parameters
    noise_group = init_parser.add_argument_group("noise parameters")
    noise_group.add_argument(
        "--training-noise",
        default=None,
        type=float,
        help="amount of noise to add to training examples (default: automatic)",
    )
    noise_group.add_argument(
        "--min-noise",
        default=0.1,
        type=float,
        help="minimum example noise level for automatic training noise (default: %(default)g)",
    )
    noise_group.add_argument(
        "--max-noise",
        default=0.1,
        type=float,
        help="maximum example noise level for automatic training noise (default: %(default)g)",
    )
    noise_group.add_argument(
        "--n-noise",
        default=10,
        type=int,
        help="number of trial values for automatic training noise (default: %(default)g)",
    )

    #
    # Training Parameters
    #
    train_parser = subparsers.add_parser(
        "train", help="train a network", parents=[io_parent_parser]
    )

    # Training paths
    train_parser.add_argument(
        "--load", required=True, help="basename of network to load"
    )

    # Loss Function parameters
    loss_group = train_parser.add_argument_group("loss function parameters")
    loss_group.add_argument(
        "--example-weight",
        type=float,
        default=1.0,
        help="weight for training by example (default: %(default)g)",
    )
    loss_group.add_argument(
        "--energy-weight",
        type=float,
        default=0.0,
        help="weight for training by energy (default: %(default)g)",
    )

    # Energy evaluation parameters
    energy_group = train_parser.add_argument_group("parameters for energy function")
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

    # Optimization parameters
    optimizer_group = train_parser.add_argument_group("optimization parameters")
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
        default=1e-4,
        help="final learning rate (default: %(default)g)",
    )
    optimizer_group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="strength of weight decay (default: %(default)g)",
    )
    optimizer_group.add_argument(
        "--max-gradient",
        type=float,
        default=1000.0,
        help="maximum allowed gradient (default: %(default)g)",
    )
    optimizer_group.add_argument(
        "--log-freq",
        type=int,
        default=10,
        help="how often to update tensorboard (default: %(default)d)",
    )

    args = parser.parse_args()

    return args


def setup_writer(args):
    writer = SummaryWriter(log_dir=f"runs/{args.save}", purge_step=0, flush_secs=30)
    setup_custom_scalars(args, writer)
    return writer


def setup_custom_scalars(args, writer):
    writer.add_custom_scalars(
        {
            "Sampling": {
                "acceptance rate": ["Multiline", ["acceptance_rate"]],
                "step size": ["Multiline", ["step_size"]],
                "gradient norm": ["Multiline", ["gradient_norm"]],
            },
            "Total Losses (weighted)": {
                "total loss": ["Multiline", ["total_loss"]],
                "energy loss": ["Multiline", ["weighted_energy_total_loss"]],
                "example loss": ["Multiline", ["weighted_example_total_loss"]],
            },
            "Example Losses (unweighted)": {
                "total": [
                    "Multiline",
                    ["example_total_loss", "val_example_total_loss"],
                ],
                "ml": ["Multiline", ["example_ml_loss", "val_example_ml_loss"]],
                "jac": ["Multiline", ["example_jac_loss", "val_example_jac_loss"]],
            },
            "Energy Losses (unweighted)": {
                "total": ["Multiline", ["energy_total_loss"]],
                "ml": ["Multiline", ["energy_kl_loss"]],
                "jac": ["Multiline", ["energy_jac_loss"]],
            },
            "Generative Energies": {
                "minimum": ["Multiline", ["minimum_energy"]],
                "mean": ["Multiline", ["mean_energy"]],
                "median": ["Multiline", ["median_energy"]],
            },
        }
    )


#
# File input / output
#


def delete_run(name):
    if os.path.exists(f"models/{name}.pkl"):
        os.remove(f"models/{name}.pkl")
    if os.path.exists(f"gen_samples/{name}.pdb"):
        os.remove(f"gen_samples/{name}.pdb")
    if os.path.exists(f"ensembles/{name}.dcd"):
        os.remove(f"ensembles/{name}.dcd")
    if os.path.exists(f"runs/{name}"):
        shutil.rmtree(f"runs/{name}")


def create_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("gen_samples", exist_ok=True)
    os.makedirs("ensembles", exist_ok=True)
    os.makedirs("validation", exist_ok=True)


def load_trajectory(pdb_path, dcd_path, align=False):
    t = md.load(dcd_path, top=pdb_path)
    if align:
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
    nsf = transforms.PiecewiseRationalQuadraticCDF(
        [n_dim],
        num_bins=spline_points,
        tails="linear",
        tail_bound=15,
        identity_init=True,
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
            tail_bound=15,
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
            tail_bound=15,
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
        tail_bound=15,
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
    stage1_layers = []

    # Create the mixed transofrm layer
    pca_block = protein.PCABlock("backbone", True)
    mixed = protein.MixedTransform(n_dim, topology, [pca_block], training_data)
    stage1_layers.append(mixed)

    if pretrans_type == "quad-cdf":
        print()
        print("Pre-training unconditional NSF layer")
        print()
        unconditional = build_nsf_unconditional(n_dim - 6, spline_points)[0]
        stage1_layers.append(unconditional)
        unconditional_net = transforms.CompositeTransform(stage1_layers).to(device)
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
        unconditional.unnormalized_derivatives.requires_grad_(False)
        stage1 = unconditional_net
    else:
        stage1 = transforms.CompositeTransform(head_layers).to(device)

    if model_type == "affine-coupling":
        stage2_layers = build_affine_coupling(
            n_dim - 6, n_coupling, hidden_layers, hidden_features, dropout_fraction
        )
    elif model_type == "affine-made":
        stage2_layers = build_affine_made(
            n_dim - 6, hidden_layers, hidden_features, dropout_fraction
        )
    elif model_type == "nsf-unconditional":
        stage2_layers = build_nsf_unconditional(n_dim - 6, spline_points)
    elif model_type == "nsf-coupling":
        stage2_layers = build_nsf_coupling(
            n_dim - 6,
            n_coupling,
            spline_points,
            hidden_layers,
            hidden_features,
            dropout_fraction,
        )
    elif model_type == "nsf-made":
        stage2_layers = build_nsf_made(
            n_dim - 6, spline_points, hidden_layers, hidden_features, dropout_fraction
        )
    else:
        raise RuntimeError()

    stage2 = transforms.CompositeTransform(stage2_layers).to(device)

    net = transforms.TwoStageComposite(stage1, stage2)
    print()
    print("Network constructed.")
    print(net)
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


def get_ml_loss(net, x_batch, example_weight, dist):
    z, z_jac = net.forward(x_batch)

    example_ml_loss = -torch.mean(dist.log_prob(z)) * example_weight
    example_jac_loss = -torch.mean(z_jac) * example_weight
    example_loss = example_ml_loss + example_jac_loss
    return example_loss, example_ml_loss, example_jac_loss


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


def pre_train_unconditional_nsf(
    net, device, training_data, batch_size, epochs, lr, out_freq
):
    mu = torch.zeros(training_data.shape[-1] - 6, device=device)
    cov = torch.eye(training_data.shape[-1] - 6, device=device)
    dist = distributions.MultivariateNormal(mu, covariance_matrix=cov).expand(
        (batch_size,)
    )

    indices = np.arange(training_data.shape[0])
    optimizer = setup_optimizer(net, lr, 0.0)
    with tqdm(range(epochs)) as progress:
        for epoch in progress:
            net.train()

            index_batch = np.random.choice(
                indices, args.pretrans_batch_size, replace=True
            )
            x_batch = training_data[index_batch, :]
            loss, _, _ = get_ml_loss(net, x_batch, 1.0, dist)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % out_freq == 0:
                progress.set_postfix(loss=f"{loss.item():8.3f}")


def train_network(args, device):
    writer = setup_writer(args)

    dcd_path = f"ensembles/{args.load}.dcd"
    traj = load_trajectory(args.pdb_path, dcd_path, align=False)
    traj.unitcell_lengths = None
    traj.unitcell_angles = None

    n_dim = traj.xyz.shape[1] * 3
    ensemble = traj.xyz.reshape(-1, n_dim)
    ensemble = torch.from_numpy(ensemble.astype("float32"))
    print(f"Ensemble has size {ensemble.shape[0]} x {ensemble.shape[1]}.\n")

    validation_dcd_path = f"validation/{args.validation}.dcd"
    valid_traj = load_trajectory(args.pdb_path, validation_dcd_path, align=False)
    n_valid_dim = valid_traj.xyz.shape[1] * 3
    validation_data = valid_traj.xyz.reshape(-1, n_valid_dim)
    validation_data = torch.from_numpy(validation_data.astype("float32")).to(device)
    print(
        f"Validation has size {validation_data.shape[0]} x {validation_data.shape[1]}.\n"
    )

    net = load_network(f"models/{args.load}.pkl", device=device)

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

    trainer = MixedLossTrainer(
        net, device, ensemble, validation_data, args.batch_size, energy_evaluator
    )

    with tqdm(range(args.epochs)) as progress:
        for epoch in progress:
            net.train()

            trainer.compute_training_losses()

            loss = (
                trainer.forward_loss * args.example_weight
                + trainer.inverse_loss * args.energy_weight
            )

            optimizer.zero_grad()
            loss.backward()
            gradient_norm = clip_grad_norm_(net.parameters(), args.max_gradient)
            optimizer.step()
            scheduler.step(epoch)

            validation_step = epoch % args.log_freq == 0
            if validation_step:
                net.eval()
                trainer.compute_validation_losses()
                writer.add_scalar(
                    "val_example_ml_loss", trainer.val_forward_ml.item(), epoch
                )
                writer.add_scalar(
                    "val_example_jac_loss", trainer.val_forward_jac.item(), epoch
                )
                writer.add_scalar(
                    "val_example_total_loss", trainer.val_forward_loss.item(), epoch
                )

                # Output our training losses
                # writer.add_scalar(
                #     "acceptance_rate", trainer.acceptance_probs[-1], epoch
                # )
                # writer.add_scalar("step_size", trainer.step_size, epoch)
                writer.add_scalar("total_loss", loss.item(), epoch)
                writer.add_scalar("gradient_norm", gradient_norm, epoch)
                writer.add_scalar("example_ml_loss", trainer.forward_ml, epoch)
                writer.add_scalar("example_jac_loss", trainer.forward_jac, epoch)
                writer.add_scalar("example_total_loss", trainer.forward_loss, epoch)
                writer.add_scalar(
                    "weighted_example_total_loss",
                    trainer.forward_loss * args.example_weight,
                    epoch,
                )
                writer.add_scalar("energy_kl_loss", trainer.inverse_kl, epoch)
                writer.add_scalar("energy_jac_loss", trainer.inverse_jac, epoch)
                writer.add_scalar("energy_total_loss", trainer.inverse_loss, epoch)
                writer.add_scalar(
                    "weighted_energy_total_loss",
                    trainer.inverse_loss * args.energy_weight,
                    epoch,
                )
                writer.add_scalar("minimum_energy", trainer.min_energy.item(), epoch)
                writer.add_scalar("median_energy", trainer.median_energy.item(), epoch)
                writer.add_scalar("mean_energy", trainer.mean_energy.item(), epoch)

                progress.set_postfix(loss=f"{loss.item():8.3f}")

    # Save our final model
    torch.save(net, f"models/{args.save}.pkl")

    # Save our reservoir
    x = trainer.reservoir.cpu().detach().numpy()
    x = x.reshape(trainer.res_size, -1, 3)
    traj.xyz = x
    traj.save(f"ensembles/{args.save}.dcd")

    # Generate examples and write trajectory
    net.eval()
    z = torch.normal(0, 1, size=(args.batch_size, n_dim - 6), device=device)
    x, _ = net.inverse(z)
    x = x.cpu().detach().numpy()
    x = x.reshape(args.batch_size, -1, 3)
    traj.xyz = x
    traj.save(f"gen_samples/{args.save}.dcd")


def calculate_example_noise(net, training_data, min_noise, max_noise, n_noise):
    # Run all training data through the pretransformation stage of the network
    transformed_data, _ = net.stage1_forward(training_data)
    transformed_data = transformed_data.cpu().detach().numpy()
    np.random.shuffle(transformed_data)
    params = {"bandwidth": np.linspace(min_noise, max_noise, n_noise)}
    grid = GridSearchCV(
        KernelDensity(kernel="gaussian", atol=1e-4, rtol=1e-4),
        params,
        cv=3,
        return_train_score=False,
    )
    grid.fit(transformed_data)
    # Use cross-validation to identify the optimal noise bandwidth.
    return grid.best_params_["bandwidth"]


def init_ensemble(ensemble_size, data):
    if data.shape[0] != ensemble_size:
        print(
            f"Generating ensemble by sampling from {data.shape[0]} to {ensemble_size}.\n"
        )
        sampled = np.random.choice(
            np.arange(data.shape[0]), ensemble_size, replace=True
        )
        ensemble = data[sampled, :]
    else:
        ensemble = data
    return ensemble


def init_network(args, device):
    traj = load_trajectory(args.pdb_path, args.dcd_path, align=True)
    traj.unitcell_lengths = None
    traj.unitcell_angles = None

    n_dim = traj.xyz.shape[1] * 3
    training_data_npy = traj.xyz.reshape(-1, n_dim)
    # Shuffle the training data for later training / test split
    np.random.shuffle(training_data_npy)
    training_data = torch.from_numpy(training_data_npy.astype("float32"))
    print(
        f"Trajectory loaded with size {training_data.shape[0]} x {training_data.shape[1]}"
    )

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
        pretrans_batch_size=args.pretrans_batch_size,
        device=device,
    )

    if args.training_noise is None:
        print(f"Using automatic noise level detection with {args.n_noise} trials.")
        net.example_noise = calculate_example_noise(
            net, training_data, args.min_noise, args.max_noise, args.n_noise
        )
        print(f"Using automatically determined noise level {net.example_noise}.\n")
    else:
        net.example_noise = args.training_noise
        print(f"Using noise level {net.example_noise} specified on command line.\n")

    # We do this just to test if we can.
    openmm_context_ = get_openmm_context(args.pdb_path)

    # Set aside our validation dataset and create our initial ensemble.
    n_valid = int(training_data.shape[0] * args.validation_fraction)
    n_train = training_data.shape[0] - n_valid
    print(
        f"Splitting data into training ({n_train} points) and validation ({n_valid} points) sets.\n"
    )
    validation_data = training_data[:n_valid, :]
    training_data = training_data[n_valid:, :]
    ensemble = init_ensemble(args.ensemble_size, training_data)

    # Save everything
    torch.save(net, f"models/{args.save}.pkl")
    x = ensemble.cpu().detach().numpy()
    x = x.reshape(args.ensemble_size, -1, 3)
    traj.xyz = x
    traj.save(f"ensembles/{args.save}.dcd")
    y = validation_data.cpu().detach().numpy()
    y = y.reshape(n_valid, -1, 3)
    traj.xyz = y
    traj.save(f"validation/{args.validation}.dcd")


class MixedLossTrainer:
    def __init__(
        self, net, device, training_data, validation_data, batch_size, energy_evaluator
    ):
        self.net = net
        self.device = device
        self.training_data = training_data
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.energy_evaluator = energy_evaluator

        self.training_indices = np.arange(self.training_data.shape[0])
        self.validation_indices = np.arange(self.validation_data.shape[0])

        # Setup latent gaussian distribution
        mu = torch.zeros(self.training_data.shape[-1] - 6, device=device)
        cov = torch.eye(self.training_data.shape[-1] - 6, device=device)
        self.latent_distribution = distributions.MultivariateNormal(
            mu, covariance_matrix=cov
        ).expand((self.batch_size,))

        # These statistics are updated during training.
        self.forward_loss = None
        self.forward_ml = None
        self.forward_jac = None
        self.val_forward_loss = None
        self.val_forward_ml = None
        self.val_forward_jac = None
        self.inverse_loss = None
        self.inverse_kl = None
        self.inverse_jac = None
        self.mean_energy = None
        self.median_energy = None
        self.min_energy = None
        self.acceptance_probs = []

    def compute_training_losses(self):
        with torch.no_grad():
            # choose random examples
            example_ind = np.random.choice(
                self.training_indices, size=self.batch_size, replace=True
            )
            x = self.training_data[example_ind, :].to(self.device)
            # transform through stage1
            z_pretrans, _ = self.net.stage1_forward(x)
            # add noise
            z_pretrans = z_pretrans + torch.normal(
                0,
                self.net.example_noise,
                size=z_pretrans.shape,
                device=z_pretrans.device,
            )
            # transform back to x
            x, _ = self.net.stage1_inverse(z_pretrans)
        # transform through full network
        z, z_jac = self.net.forward(x)
        # compute loss
        self.forward_ml = -torch.mean(self.latent_distribution.log_prob(z))
        self.forward_jac = -torch.mean(z_jac)
        self.forward_loss = self.forward_ml + self.forward_jac

        # choose random latent and compute losses
        z_prime = self.latent_distribution.sample().to(self.device)
        x_prime, x_jac_prime = self.net.inverse(z_prime)
        energies = self.energy_evaluator(x_prime)
        self.min_energy = torch.min(energies)
        self.median_energy = torch.median(energies)
        self.mean_energy = torch.mean(energies)
        self.inverse_kl = torch.mean(energies)
        self.inverse_jac = -torch.mean(x_jac_prime)
        self.inverse_loss = self.inverse_kl + self.inverse_jac

    def compute_validation_losses(self):
        with torch.no_grad():
            valid_ind = np.random.choice(
                self.validation_indices, size=self.batch_size, replace=True
            )
            x_valid = self.validation_data[valid_ind, :].to(self.device)
            z_valid, z_jac_valid = self.net.forward(x_valid)
            self.val_forward_ml = -torch.mean(
                self.latent_distribution.log_prob(z_valid)
            )
            self.val_forward_jac = -torch.mean(z_jac_valid)
            self.val_forward_loss = self.val_forward_ml + self.val_forward_jac


if __name__ == "__main__":
    args = parse_args()

    model_path = f"models/{args.save}.pkl"
    if os.path.exists(model_path):
        if args.overwrite:
            print(f"Warning: output `{model_path}' already exists. Overwriting anyway.")
        else:
            raise RuntimeError(
                f"Output '{model_path}' already exists. If you're sure use --overwrite."
            )

    delete_run(args.save)
    create_dirs()
    device = get_device()

    if args.action == "init":
        init_network(args, device)
    elif args.action == "train":
        train_network(args, device)
    else:
        raise RuntimeError(f"Unknown command {args.action}.")
