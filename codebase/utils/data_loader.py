import os
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


def load_data(args):
    loc_max, loc_min, vel_max, vel_min = None, None, None, None
    train_loader, valid_loader, test_loader = None, None, None

    if "kuramoto" in args.suffix:
        train_loader, valid_loader, test_loader = load_ode_data(
            args,
            suffix=args.suffix,
            batch_size=args.batch_size_multiGPU,
            datadir=args.datadir,
        )
    elif "netsim" in args.suffix:
        train_loader, loc_max, loc_min = load_netsim_data(
            batch_size=args.batch_size_multiGPU, datadir=args.datadir
        )
    elif "springs" in args.suffix:
        (
            train_loader,
            valid_loader,
            test_loader,
            loc_max,
            loc_min,
            vel_max,
            vel_min,
        ) = load_springs_data(
            args, args.batch_size_multiGPU, args.suffix, datadir=args.datadir
        )
    else:
        raise NameError("Unknown data to be loaded")

    return train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min


def normalize(x, x_min, x_max):
    return (x - x_min) * 2 / (x_max - x_min) - 1


def remove_unobserved_from_data(loc, vel, edge, args):
    loc = loc[:, :, :, : -args.unobserved]
    vel = vel[:, :, :, : -args.unobserved]
    edge = edge[:, : -args.unobserved, : -args.unobserved]
    return loc, vel, edge


def get_off_diag_idx(num_atoms):
    return np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms],
    )


def data_preparation(
    loc,
    vel,
    edges,
    loc_min,
    loc_max,
    vel_min,
    vel_max,
    off_diag_idx,
    num_atoms,
    temperature=None,
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    # Normalize to [-1, 1]
    loc = normalize(loc, loc_min, loc_max)
    vel = normalize(vel, vel_min, vel_max)

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc = np.transpose(loc, [0, 3, 1, 2])
    vel = np.transpose(vel, [0, 3, 1, 2])
    feat = np.concatenate([loc, vel], axis=3)
    edges = np.reshape(edges, [-1, num_atoms ** 2])
    edges = np.array((edges + 1) / 2, dtype=np.int64)

    feat = torch.FloatTensor(feat)
    edges = torch.LongTensor(edges)

    edges = edges[:, off_diag_idx]

    if temperature is not None:
        dataset = TensorDataset(feat, edges, temperature)
    else:
        dataset = TensorDataset(feat, edges)

    return dataset


def load_springs_data(args, batch_size=1, suffix="", datadir="data"):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    print("Loading data from {}".format(datadir))
    loc_train = np.load(os.path.join(datadir, "loc_train" + suffix + ".npy"))
    vel_train = np.load(os.path.join(datadir, "vel_train" + suffix + ".npy"))
    edges_train = np.load(os.path.join(datadir, "edges_train" + suffix + ".npy"))

    loc_valid = np.load(os.path.join(datadir, "loc_valid" + suffix + ".npy"))
    vel_valid = np.load(os.path.join(datadir, "vel_valid" + suffix + ".npy"))
    edges_valid = np.load(os.path.join(datadir, "edges_valid" + suffix + ".npy"))

    loc_test = np.load(os.path.join(datadir, "loc_test" + suffix + ".npy"))
    vel_test = np.load(os.path.join(datadir, "vel_test" + suffix + ".npy"))
    edges_test = np.load(os.path.join(datadir, "edges_test" + suffix + ".npy"))

    if args.load_temperatures:
        temperatures_train, temperatures_valid, temperatures_test = load_temperatures(
            suffix=suffix, datadir=datadir
        )
    else:
        temperatures_train, temperatures_valid, temperatures_test = None, None, None

    # [num_samples, num_timesteps, num_dims, num_atoms]
    if args.training_samples != 0:
        loc_train = loc_train[: args.training_samples]
        vel_train = vel_train[: args.training_samples]
        edges_train = edges_train[: args.training_samples]

    if args.test_samples != 0:
        loc_test = loc_test[: args.test_samples]
        vel_test = vel_test[: args.test_samples]
        edges_test = edges_test[: args.test_samples]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Exclude self edges
    off_diag_idx = get_off_diag_idx(args.num_atoms)

    train_data = data_preparation(
        loc_train,
        vel_train,
        edges_train,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_train,
    )
    valid_data = data_preparation(
        loc_valid,
        vel_valid,
        edges_valid,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_valid,
    )
    test_data = data_preparation(
        loc_test,
        vel_test,
        edges_test,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_test,
    )
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    return (
        train_data_loader,
        valid_data_loader,
        test_data_loader,
        loc_max,
        loc_min,
        vel_max,
        vel_min,
    )


def load_temperatures(suffix="", datadir="data"):
    temperatures_train = np.load(
        os.path.join(datadir, "temperatures_train" + suffix + ".npy")
    )
    temperatures_valid = np.load(
        os.path.join(datadir, "temperatures_valid" + suffix + ".npy")
    )
    temperatures_test = np.load(
        os.path.join(datadir, "temperatures_test" + suffix + ".npy")
    )

    temperatures_train = torch.FloatTensor(temperatures_train)
    temperatures_valid = torch.FloatTensor(temperatures_valid)
    temperatures_test = torch.FloatTensor(temperatures_test)

    return temperatures_train, temperatures_valid, temperatures_test


def load_ode_data(args, batch_size=1, suffix="", datadir="data"):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    feat_train = np.load(os.path.join(datadir, "feat_train" + suffix + ".npy"))
    edges_train = np.load(os.path.join(datadir, "edges_train" + suffix + ".npy"))
    feat_valid = np.load(os.path.join(datadir, "feat_valid" + suffix + ".npy"))
    edges_valid = np.load(os.path.join(datadir, "edges_valid" + suffix + ".npy"))
    feat_test = np.load(os.path.join(datadir, "feat_test" + suffix + ".npy"))
    edges_test = np.load(os.path.join(datadir, "edges_test" + suffix + ".npy"))

    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]

    if args.training_samples != 0:
        feat_train = feat_train[: args.training_samples]
        edges_train = edges_train[: args.training_samples]

    if args.test_samples != 0:
        feat_test = feat_test[: args.test_samples]
        edges_test = edges_test[: args.test_samples]

    # Reshape to: [num_sims, num_atoms * num_atoms]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    edges_train = edges_train / np.max(edges_train)
    edges_valid = edges_valid / np.max(edges_valid)
    edges_test = edges_test / np.max(edges_test)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = get_off_diag_idx(num_atoms)
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )  # , num_workers=8
    # )
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(
        test_data, batch_size=batch_size
    )  # , num_workers=8) ##THIS

    return train_data_loader, valid_data_loader, test_data_loader


def load_data_for_ncg(datadir, data_index, suffix):
    """Data loading for Neural Granger Causality method (one example at a time)."""
    feat_train = np.load(os.path.join(datadir, "feat_train_small" + suffix + ".npy"))
    edges_train = np.load(os.path.join(datadir, "edges_train_small" + suffix + ".npy"))
    return feat_train[data_index], edges_train[data_index]


def load_netsim_data(batch_size=1, datadir="data"):
    print("Loading data from {}".format(datadir))

    subject_id = [1, 2, 3, 4, 5]

    print("Loading data for subjects ", subject_id)

    loc_train = torch.zeros(len(subject_id), 15, 200)
    edges_train = torch.zeros(len(subject_id), 15, 15)

    for idx, elem in enumerate(subject_id):
        fileName = "sim3_subject_%s.npz" % (elem)
        ld = np.load(os.path.join(datadir, "netsim", fileName))
        loc_train[idx] = torch.FloatTensor(ld["X_np"])
        edges_train[idx] = torch.LongTensor(ld["Gref"])

    # [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = loc_train.unsqueeze(-1)

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    loc_train = normalize(loc_train, loc_min, loc_max)

    # Exclude self edges
    num_atoms = loc_train.shape[1]

    off_diag_idx = get_off_diag_idx(num_atoms)
    edges_train = torch.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = (edges_train + 1) // 2
    edges_train = edges_train[:, off_diag_idx]

    train_data = TensorDataset(loc_train, edges_train)

    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )

    return (train_data_loader, loc_max, loc_min)


def unpack_batches(args, minibatch):
    if args.load_temperatures:
        (data, relations, temperatures) = minibatch
    else:
        (data, relations) = minibatch
        temperatures = None
    if args.cuda:
        data, relations = data.cuda(), relations.cuda()
        if args.load_temperatures:
            temperatures = temperatures.cuda()
    return data, relations, temperatures
