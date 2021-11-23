import argparse
import torch
import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--GPU_to_use", type=int, default=None, help="GPU to use for training"
    )

    ############## training hyperparameter ##############
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Number of samples per batch."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0005, help="Initial learning rate."
    )
    parser.add_argument(
        "--lr_decay",
        type=int,
        default=200,
        help="After how epochs to decay LR by a factor of gamma.",
    )
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor.")
    parser.add_argument(
        "--training_samples", type=int, default=0,
        help="If 0 use all data available, otherwise reduce number of samples to given number"
    )
    parser.add_argument(
        "--test_samples", type=int, default=0,
        help="If 0 use all data available, otherwise reduce number of samples to given number"
    )
    parser.add_argument(
        "--prediction_steps",
        type=int,
        default=10,
        metavar="N",
        help="Num steps to predict before re-using teacher forcing.",
    )

    ############## architecture ##############
    parser.add_argument(
        "--encoder_hidden", type=int, default=256, help="Number of hidden units."
    )
    parser.add_argument(
        "--decoder_hidden", type=int, default=256, help="Number of hidden units."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="mlp",
        help="Type of path encoder model (mlp or cnn).",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="mlp",
        help="Type of decoder model (mlp, rnn, or sim).",
    )
    parser.add_argument(
        "--prior",
        type=float,
        default=1,
        help="Weight for sparsity prior (if == 1, uniform prior is applied)",
    )
    parser.add_argument(
        "--edge_types",
        type=int,
        default=2,
        help="Number of different edge-types to model",
    )

    ########### Different variants for variational distribution q ###############
    parser.add_argument(
        "--dont_use_encoder",
        action="store_true",
        default=False,
        help="If true, replace encoder with distribution to be estimated",
    )
    parser.add_argument(
        "--lr_z",
        type=float,
        default=0.1,
        help="Learning rate for distribution estimation.",
    )

    ### global latent temperature ###
    parser.add_argument(
        "--global_temp",
        action="store_true",
        default=False,
        help="Should we model temperature confounding?",
    )
    parser.add_argument(
        "--load_temperatures",
        help="Should we load temperature data?",
        action="store_true",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2,
        help="Middle value of temperature distribution.",
    )
    parser.add_argument(
        "--num_cats",
        type=int,
        default=3,
        help="Number of categories in temperature distribution.",
    )

    ### unobserved time-series ###
    parser.add_argument(
        "--unobserved",
        type=int,
        default=0,
        help="Number of time-series to mask from input.",
    )
    parser.add_argument(
        "--model_unobserved",
        type=int,
        default=0,
        help="If 0, use NRI to infer unobserved particle. "
        "If 1, removes unobserved from data. "
        "If 2, fills empty slot with mean of observed time-series (mean imputation)",
    )
    parser.add_argument(
        "--dont_shuffle_unobserved",
        action="store_true",
        default=False,
        help="If true, always mask out last particle in trajectory. "
        "If false, mask random particle.",
    )
    parser.add_argument(
        "--teacher_forcing",
        type=int,
        default=0,
        help="Factor to determine how much true trajectory of "
        "unobserved particle should be used to learn prediction.",
    )

    ############## loading and saving ##############
    parser.add_argument(
        "--suffix",
        type=str,
        default="_springs5",
        help='Suffix for training data.',
    )
    parser.add_argument(
        "--timesteps", type=int, default=49, help="Number of timesteps in input."
    )
    parser.add_argument(
        "--num_atoms", type=int, default=5, help="Number of time-series in input."
    )
    parser.add_argument(
        "--dims", type=int, default=4, help="Dimensionality of input."
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./data",
        help="Name of directory where data is stored.",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="logs",
        help="Where to save the trained model, leave empty to not save anything.",
    )
    parser.add_argument(
        "--expername",
        type=str,
        default="",
        help="If given, creates a symlinked directory by this name in logdir"
        "linked to the results file in save_folder"
        "(be careful, this can overwrite previous results)",
    )
    parser.add_argument(
        "--sym_save_folder",
        type=str,
        default="../logs",
        help="Name of directory where symlinked named experiment is created."
    )
    parser.add_argument(
        "--load_folder",
        type=str,
        default="",
        help="Where to load pre-trained model if finetuning/evaluating. "
        + "Leave empty to train from scratch",
    )

    ############## fine tuning ##############
    parser.add_argument(
        "--test_time_adapt",
        action="store_true",
        default=False,
        help="Test time adapt q(z) on first half of test sequences.",
    )
    parser.add_argument(
        "--lr_logits",
        type=float,
        default=0.01,
        help="Learning rate for test-time adapting logits.",
    )
    parser.add_argument(
        "--num_tta_steps",
        type=int,
        default=100,
        help="Number of test-time-adaptation steps per batch.",
    )

    ############## almost never change these ##############
    parser.add_argument(
        "--dont_skip_first",
        action="store_true",
        default=False,
        help="If given as argument, do not skip first edge type in decoder, i.e. it represents no-edge.",
    )
    parser.add_argument(
        "--temp", type=float, default=0.5, help="Temperature for Gumbel softmax."
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        default=False,
        help="Uses discrete samples in training forward pass.",
    )
    parser.add_argument(
        "--no_validate", action="store_true", default=False, help="Do not validate results throughout training."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="Disables CUDA training."
    )
    parser.add_argument("--var", type=float, default=5e-7, help="Output variance.")
    parser.add_argument(
        "--encoder_dropout",
        type=float,
        default=0.0,
        help="Dropout rate (1 - keep probability).",
    )
    parser.add_argument(
        "--decoder_dropout",
        type=float,
        default=0.0,
        help="Dropout rate (1 - keep probability).",
    )
    parser.add_argument(
        "--no_factor",
        action="store_true",
        default=False,
        help="Disables factor graph model.",
    )

    args = parser.parse_args()
    args.test = True

    ### Presets for different datasets ###
    if (
        "fixed" in args.suffix
        or "uninfluenced" in args.suffix
        or "influencer" in args.suffix
        or "conf" in args.suffix
    ):
        args.dont_shuffle_unobserved = True

    if "netsim" in args.suffix:
        args.dims = 1
        args.num_atoms = 15
        args.timesteps = 200
        args.no_validate = True
        args.test = False

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.factor = not args.no_factor
    args.validate = not args.no_validate
    args.shuffle_unobserved = not args.dont_shuffle_unobserved
    args.skip_first = not args.dont_skip_first
    args.use_encoder = not args.dont_use_encoder
    args.time = datetime.datetime.now().isoformat()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.type != "cpu":
        if args.GPU_to_use is not None:
            torch.cuda.set_device(args.GPU_to_use)
        torch.cuda.manual_seed(args.seed)
        args.num_GPU = 1  # torch.cuda.device_count()
        args.batch_size_multiGPU = args.batch_size * args.num_GPU
    else:
        args.num_GPU = None
        args.batch_size_multiGPU = args.batch_size

    return args
