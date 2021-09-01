"""Based on https://github.com/ethanfetaya/NRI (MIT License)."""

import numpy as np
import os
import time
import argparse

from data import kuramoto

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_train",
        type=int,
        default=50000,
        help="Number of training simulations to generate.",
    )
    parser.add_argument(
        "--num_valid",
        type=int,
        default=10000,
        help="Number of validation simulations to generate.",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=10000,
        help="Number of test simulations to generate.",
    )
    parser.add_argument(
        "--length", type=int, default=5000, help="Length of trajectory."
    )
    parser.add_argument(
        "--length_test", type=int, default=10000, help="Length of test set trajectory."
    )
    parser.add_argument(
        "--num_atoms",
        type=int,
        default=5,
        help="Number of atoms (aka time-series) in the simulation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--ode_type", type=str, default="kuramoto", help="Which ODE to use [kuramoto]"
    )
    parser.add_argument('--sample_freq', type=int, default=100,
                        help='How often to sample the trajectory.')
    parser.add_argument('--interaction_strength', type=int, default=1,
                        help='Strength of Interactions between particles')
    parser.add_argument(
        "--undirected",
        action="store_true",
        default=False,
        help="Have symmetric connections",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data",
        help="Where to save generated data.",
    )
    parser.add_argument(
        "--n_save_small",
        type=int,
        default=100,  
        help="Number of training sequences to save separately.",
    )
    args = parser.parse_args()
    print(args)
    return args

def generate_dataset(num_sims, length, sample_freq):
    num_sims = num_sims
    num_timesteps = int((length / float(sample_freq)) - 1)

    t0, t1, dt = 0, int((length / float(sample_freq)) / 10), 0.01
    T = np.arange(t0, t1, dt)

    sim_data_all = []
    edges_all = []
    for i in range(num_sims):
        t = time.time()

        if args.ode_type == "kuramoto":
            sim_data, edges = kuramoto.simulate_kuramoto(
                args.num_atoms, num_timesteps, T, dt, args.undirected
            )
            assert sim_data.shape[2] == 4
        else:
            raise Exception("Invalid args.ode_type")

        sim_data_all.append(sim_data)
        edges_all.append(edges)

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

    data_all = np.array(sim_data_all, dtype=np.float32)
    edges_all = np.array(edges_all, dtype=np.int64)

    return data_all, edges_all


if __name__ == "__main__":

    args = parse_args()
    np.random.seed(args.seed)

    suffix = "_" + args.ode_type

    suffix += str(args.num_atoms)

    if args.undirected:
        suffix += "undir"

    if args.interaction_strength != 1:
        suffix += "_inter" + str(args.interaction_strength)

    print(suffix)

    # NOTE: We first generate all sequences with same length as length_test
    # and then later cut them to required length. Otherwise normalization is
    # messed up (for absolute phase variable).
    print("Generating {} training simulations".format(args.num_train))
    data_train, edges_train = generate_dataset(
        args.num_train, args.length_test, args.sample_freq
    )

    print("Generating {} validation simulations".format(args.num_valid))
    data_valid, edges_valid = generate_dataset(
        args.num_valid, args.length_test, args.sample_freq
    )

    num_timesteps_train = int((args.length / float(args.sample_freq)) - 1)
    data_train = data_train[:, :, :num_timesteps_train, :]
    data_valid = data_valid[:, :, :num_timesteps_train, :]

    # Save 100 training examples as separate block, so we can compare cLSTM +
    # NRI models.
    small_data_train = data_train[:args.n_save_small]
    small_edges_train = edges_train[:args.n_save_small]

    print("Generating {} test simulations".format(args.num_test))
    data_test, edges_test = generate_dataset(
        args.num_test, args.length_test, args.sample_freq
    )

    savepath = os.path.expanduser(args.save_dir)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print("Saving to {}".format(savepath))
    np.save(
        os.path.join(savepath, "feat_train" + suffix + ".npy"),
        data_train,
    )
    np.save(
        os.path.join(savepath, "edges_train" + suffix + ".npy"),
        edges_train,
    )

    np.save(
        os.path.join(savepath, "feat_train_small" + suffix + ".npy"),
        small_data_train,
    )
    np.save(
        os.path.join(savepath, "edges_train_small" + suffix + ".npy"),
        small_edges_train,
    )

    np.save(
        os.path.join(savepath, "feat_valid" + suffix + ".npy"),
        data_valid,
    )
    np.save(
        os.path.join(savepath, "edges_valid" + suffix + ".npy"),
        edges_valid,
    )

    np.save(
        os.path.join(savepath, "feat_test" + suffix + ".npy"),
        data_test,
    )
    np.save(
        os.path.join(savepath, "edges_test" + suffix + ".npy"),
        edges_test,
    )
