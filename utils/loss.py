from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from utils.miner import AdaTripletMiner


def distance(x, dist_type, normalize_input=True):
    """
    Args:
        x: embedding tensor
        dist_type: distance metric type
            "cosine" means the negative cosine similarity
            ...
        normalize_input: l2-normalize x or not
    Returns:
        distance matrix, which contains the distance between any two embeddings
    """
    if normalize_input:
        x = F.normalize(x, p=2, dim=1)
    if dist_type == "cosine":
        return -(x @ x.T)
    elif dist_type == "euclidean":
        return torch.cdist(x, x, p=2)
    elif dist_type == "squared_euclidean":
        return ((x.unsqueeze(1) - x.unsqueeze(0)) ** 2).sum(-1)
    else:
        raise NotImplementedError(f"not support: {dist_type}")


# Bit var loss for rotation
def bit_var_loss():
    def F(x):
        return 1 / (1 + torch.exp(-x))

    def loss(Z):
        return torch.mean(F(Z) * (1 - F(Z)))

    return loss


class AdaTripletLoss(nn.Module):
    def __init__(self, args: Namespace, verbose=False):
        super().__init__()
        self.type_of_distance = args.type_of_distance
        self.la = args.la
        self.verbose = verbose
        self.miner = AdaTripletMiner(args)

    def forward(self, logits, labels):
        # indices_tuple = get_all_triplets_indices(labels)
        triplets, an_pairs = self.miner(logits.detach(), labels)

        # get the updated params
        epsilon = self.miner.epsilon
        beta = self.miner.beta

        mat = distance(logits, self.type_of_distance)

        anchor_idx, positive_idx, negative_idx = triplets
        # n_triplets = indices_tuple[0].numel()
        n_triplets = len(anchor_idx)
        if n_triplets > 0:
            ap_dists = mat[anchor_idx, positive_idx]
            an_dists = mat[anchor_idx, negative_idx]
            violation = ap_dists - an_dists + epsilon
            loss_triplet = F.relu(violation).mean()
        else:
            # print("no triplets")
            loss_triplet = 0

        if an_pairs is not None:
            anchor_idx, negative_idx = an_pairs
            n_an_pairs = len(an_pairs[0])
            if n_an_pairs > 0:
                # note: origin code may not smart!
                # f_anc = F.normalize(logits[anchor_idx], p=2, dim=1)
                # f_neg = F.normalize(logits[negative_idx], p=2, dim=1)
                # an = torch.matmul(f_anc.unsqueeze(1), f_neg.unsqueeze(2)).squeeze()
                # loss_neg2 = torch.clamp(an - beta, min=0).mean()

                loss_an = F.relu(-mat[anchor_idx, negative_idx] - beta).mean()
                # print(loss_neg, loss_neg2)
            else:
                # print("no neg_pairs")
                loss_an = 0
        else:
            n_an_pairs = 0
            loss_an = 0

        loss = loss_triplet + self.la * loss_an
        return loss if not self.verbose else (loss, n_triplets, n_an_pairs, epsilon, beta)


if __name__ == "__main__":
    _args = Namespace(
        type_of_distance="cosine",
        type_of_triplets="all",
        epsilon=0.25,
        beta=0,
        k_delta=2,
        k_an=2,
        la=1,
    )
    _criteria = AdaTripletLoss(_args)

    _logits = torch.rand(50, 16)
    _labels = (torch.randn(50, 10) > 0.8).byte()

    print(_criteria(_logits, _labels))
