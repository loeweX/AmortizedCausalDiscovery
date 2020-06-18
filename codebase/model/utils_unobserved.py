import torch
import torch.nn.functional as F
from collections import defaultdict

from model import utils


def remove_unobserved(args, data, mask_idx):
    data = torch.cat(
        (data[:, :mask_idx, :, :], data[:, mask_idx + args.unobserved :, :, :],), dim=1,
    )
    return data


def baseline_mean_imputation(args, data_encoder, mask_idx):
    target_unobserved = data_encoder[:, mask_idx, :, :]
    data_encoder = remove_unobserved(args, data_encoder, mask_idx)

    unobserved = torch.mean(data_encoder, dim=1).unsqueeze(1)

    mse_unobserved = F.mse_loss(
        torch.squeeze(unobserved), torch.squeeze(target_unobserved)
    )

    data_encoder = torch.cat(
        (data_encoder[:, :mask_idx, :], unobserved, data_encoder[:, mask_idx:, :],),
        dim=1,
    )

    return data_encoder, unobserved, mse_unobserved


def baseline_remove_unobserved(
    args, data_encoder, data_decoder, mask_idx, relations, predicted_atoms
):
    data_encoder = remove_unobserved(args, data_encoder, mask_idx)
    data_decoder = remove_unobserved(args, data_decoder, mask_idx)

    predicted_atoms -= args.unobserved
    observed_relations_idx = utils.get_observed_relations_idx(args.num_atoms)
    relations = relations[:, observed_relations_idx]

    return data_encoder, data_decoder, predicted_atoms, relations


def add_unobserved_to_data(args, data, unobserved, mask_idx, diff_data_enc_dec):
    if diff_data_enc_dec:
        data = torch.cat(
            (
                data[:, :mask_idx, :],
                torch.unsqueeze(unobserved[:, :, -1, :], 2).repeat(
                    1, 1, args.timesteps, 1
                ),  # start predicting unobserved path from last point predicted
                data[:, mask_idx + 1 :, :],
            ),
            dim=1,
        )
    else:
        data = torch.cat(
            (data[:, :mask_idx, :], unobserved, data[:, mask_idx + 1 :, :],), dim=1,
        )

    return data


def calc_mse_observed(args, output, target, mask_idx):
    output_observed = remove_unobserved(args, output, mask_idx)
    target_observed = remove_unobserved(args, target, mask_idx)
    return F.mse_loss(output_observed, target_observed)


def calc_performance_per_num_influenced(args, relations, output, target, logits, prob, mask_idx, losses):
    if args.model_unobserved == 1:
        num_atoms = args.num_atoms - args.unobserved
    else:
        num_atoms = args.num_atoms

    influenced_idx_relations = list(
        range(num_atoms - 2, num_atoms ** 2, num_atoms - 1)
    )[: num_atoms - 1]
    influenced_idx = relations[:, influenced_idx_relations]

    ## calculate performance based on how many particles are influenced by unobserved one
    total_num_influenced = torch.sum(influenced_idx, 1).tolist()
    if args.model_unobserved != 1 and args.unobserved > 0:
        observed_idx = utils.get_observed_relations_idx(args.num_atoms).astype(int)
        acc_per_sample = utils.edge_accuracy_per_sample(logits[:, observed_idx, :], relations[:, observed_idx])

        output_observed = remove_unobserved(args, output, mask_idx)
        target_observed = remove_unobserved(args, target, mask_idx)
        mse_per_sample = utils.mse_per_sample(output_observed, target_observed)

        auroc_per_num_infl = utils.auroc_per_num_influenced(prob[:, observed_idx, :], relations[:, observed_idx], total_num_influenced)
    else:
        acc_per_sample = utils.edge_accuracy_per_sample(logits, relations)
        mse_per_sample = utils.mse_per_sample(output, target)
        auroc_per_num_infl= utils.auroc_per_num_influenced(prob, relations, total_num_influenced)

    if losses["acc_per_num_influenced"] == 0:
        losses["acc_per_num_influenced"] = defaultdict(list)
        losses["mse_per_num_influenced"] = defaultdict(list)
        losses["auroc_per_num_influenced"] = defaultdict(list)

    for idx, k in enumerate(total_num_influenced):
        losses["acc_per_num_influenced"][k].append(acc_per_sample[idx])
        losses["mse_per_num_influenced"][k].append(mse_per_sample[idx])

    for idx, elem in enumerate(auroc_per_num_infl):
        losses["auroc_per_num_influenced"][idx].append(elem)

    return losses