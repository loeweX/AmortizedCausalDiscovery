from __future__ import division
from __future__ import print_function

from collections import defaultdict

import time
import numpy as np
import torch

from model.modules import *
from utils import arg_parser, logger, data_loader, forward_pass_and_eval
from model import utils, model_loader


def train():
    best_val_loss = np.inf
    best_epoch = 0

    for epoch in range(args.epochs):
        t_epoch = time.time()
        train_losses = defaultdict(list)

        for batch_idx, minibatch in enumerate(train_loader):

            data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

            optimizer.zero_grad()

            losses, _, _, _ = forward_pass_and_eval.forward_pass_and_eval(
                args,
                encoder,
                decoder,
                data,
                relations,
                rel_rec,
                rel_send,
                args.hard,
                edge_probs=edge_probs,
                log_prior=log_prior,
                temperatures=temperatures,
            )

            loss = losses["loss"]

            loss.backward()
            optimizer.step()

            train_losses = utils.append_losses(train_losses, losses)

        string = logs.result_string("train", epoch, train_losses, t=t_epoch)
        logs.write_to_log_file(string)
        logs.append_train_loss(train_losses)
        scheduler.step()

        if args.validate:
            val_losses = val(epoch)
            val_loss = np.mean(val_losses["loss"])
            if val_loss < best_val_loss:
                print("Best model so far, saving...")
                logs.create_log(
                    args,
                    encoder=encoder,
                    decoder=decoder,
                    optimizer=optimizer,
                    accuracy=np.mean(val_losses["acc"]),
                )
                best_val_loss = val_loss
                best_epoch = epoch
        elif (epoch + 1) % 100 == 0:
            logs.create_log(
                args,
                encoder=encoder,
                decoder=decoder,
                optimizer=optimizer,
                accuracy=np.mean(train_losses["acc"]),
            )

        logs.draw_loss_curves()

    return best_epoch, epoch


def val(epoch):
    t_val = time.time()
    val_losses = defaultdict(list)

    if args.use_encoder:
        encoder.eval()
    decoder.eval()

    for batch_idx, minibatch in enumerate(valid_loader):

        data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

        with torch.no_grad():
            losses, _, _, _ = forward_pass_and_eval.forward_pass_and_eval(
                args,
                encoder,
                decoder,
                data,
                relations,
                rel_rec,
                rel_send,
                True,
                edge_probs=edge_probs,
                log_prior=log_prior,
                testing=True,
                temperatures=temperatures,
            )

        val_losses = utils.append_losses(val_losses, losses)

    string = logs.result_string("validate", epoch, val_losses, t=t_val)
    logs.write_to_log_file(string)
    logs.append_val_loss(val_losses)

    if args.use_encoder:
        encoder.train()
    decoder.train()

    return val_losses


def test(encoder, decoder, epoch):
    args.shuffle_unobserved = False
    # args.prediction_steps = 49
    test_losses = defaultdict(list)

    if args.load_folder == "":
        ## load model that had the best validation performance during training
        if args.use_encoder:
            encoder.load_state_dict(torch.load(args.encoder_file))
        decoder.load_state_dict(torch.load(args.decoder_file))

    if args.use_encoder:
        encoder.eval()
    decoder.eval()

    for batch_idx, minibatch in enumerate(test_loader):

        data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

        with torch.no_grad():
            assert (data.size(2) - args.timesteps) >= args.timesteps

            data_encoder = data[:, :, : args.timesteps, :].contiguous()
            data_decoder = data[:, :, args.timesteps : -1, :].contiguous()

            losses, _, _, _, = forward_pass_and_eval.forward_pass_and_eval(
                args,
                encoder,
                decoder,
                data,
                relations,
                rel_rec,
                rel_send,
                True,
                data_encoder=data_encoder,
                data_decoder=data_decoder,
                edge_probs=edge_probs,
                log_prior=log_prior,
                testing=True,
                temperatures=temperatures,
            )

        test_losses = utils.append_losses(test_losses, losses)

    string = logs.result_string("test", epoch, test_losses)
    logs.write_to_log_file(string)
    logs.append_test_loss(test_losses)

    logs.create_log(
        args,
        decoder=decoder,
        encoder=encoder,
        optimizer=optimizer,
        final_test=True,
        test_losses=test_losses,
    )


if __name__ == "__main__":

    args = arg_parser.parse_args()
    logs = logger.Logger(args)

    if args.GPU_to_use is not None:
        logs.write_to_log_file("Using GPU #" + str(args.GPU_to_use))

    (
        train_loader,
        valid_loader,
        test_loader,
        loc_max,
        loc_min,
        vel_max,
        vel_min,
    ) = data_loader.load_data(args)

    rel_rec, rel_send = utils.create_rel_rec_send(args, args.num_atoms)

    encoder, decoder, optimizer, scheduler, edge_probs = model_loader.load_model(
        args, loc_max, loc_min, vel_max, vel_min
    )

    logs.write_to_log_file(encoder)
    logs.write_to_log_file(decoder)

    if args.prior != 1:
        assert 0 <= args.prior <= 1, "args.prior not in the right range"
        prior = np.array(
            [args.prior]
            + [
                (1 - args.prior) / (args.edge_types - 1)
                for _ in range(args.edge_types - 1)
            ]
        )
        logs.write_to_log_file("Using prior")
        logs.write_to_log_file(prior)
        log_prior = torch.FloatTensor(np.log(prior))
        log_prior = log_prior.unsqueeze(0).unsqueeze(0)

        if args.cuda:
            log_prior = log_prior.cuda()
    else:
        log_prior = None

    if args.global_temp:
        args.categorical_temperature_prior = utils.get_categorical_temperature_prior(
            args.alpha, args.num_cats, to_cuda=args.cuda
        )

    ##Train model
    try:
        if args.test_time_adapt:
            raise KeyboardInterrupt

        best_epoch, epoch = train()

    except KeyboardInterrupt:
        best_epoch, epoch = -1, -1

    print("Optimization Finished!")
    logs.write_to_log_file("Best Epoch: {:04d}".format(best_epoch))

    if args.test:
        test(encoder, decoder, epoch)
