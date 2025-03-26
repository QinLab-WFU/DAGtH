from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F


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


def get_all_triplets_indices(labels):
    # if batch_size = 7
    # A = sames.unsqueeze(2) -> 7 x 7 x 1
    # B = diffs.unsqueeze(1) -> 7 x 1 x 7
    # A * B -> [7 x 7 x 1->repeat->7] * [7 x 1->repeat->7 x 7]
    #           a11 a11..........a11     b11 b12..........b17 : a11=0 -> pass
    #           a12 a12..........a12     b11 b12..........b17 : if a12=1 & b15=1 -> find triplet: 1=anc 2=pos 5=neg
    #           ...
    #           a77...
    # torch.where([7 x 7 x 7]) -> all 1s in the matrix's 3d index: ([x1,...],[y1,...],[z1,...])
    # so xi, yi, zi is index of anchor, positive and negative in the batch
    if len(labels.shape) == 1:
        sames = labels.unsqueeze(1) == labels.unsqueeze(0)
    else:
        sames = (labels @ labels.T > 0).byte()
    diffs = sames ^ 1
    sames.fill_diagonal_(0)
    # NOTE: gen triplets using AP=1 & AN=0 but lack of PN=0, which will not harm the TripletLoss,
    # because another triplet may use P as A.
    return torch.where(sames.unsqueeze(2) * diffs.unsqueeze(1))


def unique_filter_code(anchor_idx, negative_idx):
    """
    find the unique AN pairs
    1) AN = NA
    2) AN in AP1N, AP2N is same
    """
    an_idxes = torch.stack((anchor_idx, negative_idx)).T
    unique_an_idxes = list(set([(c, b) if c <= b else (b, c) for c, b in an_idxes.tolist()]))
    anc_idxes = torch.tensor([x[0] for x in unique_an_idxes], dtype=torch.int64)
    neg_idxes = torch.tensor([x[1] for x in unique_an_idxes], dtype=torch.int64)
    return anc_idxes, neg_idxes


def unique_filter(anchor_idx, negative_idx):
    x = torch.stack((anchor_idx, negative_idx), dim=1)
    idx = x[:, 0] > x[:, 1]
    x[idx] = x[idx][:, [1, 0]]
    rst = torch.unique(x, dim=0)
    return rst[:, 0], rst[:, 1]


class AdaTripletMiner(nn.Module):
    """
    Returns triplets that violate the margin
    """

    def __init__(self, args: Namespace):
        super().__init__()
        self.type_of_distance = args.type_of_distance
        self.type_of_triplets = args.type_of_triplets

        self.epsilon = args.epsilon
        self.beta = args.beta
        self.k_delta = args.k_delta
        self.k_an = args.k_an
        self.calc_loss_an = args.calc_loss_an

    def update(self, dist, dist_type):
        # dist = dist.cpu().numpy()
        if dist_type == "ap_an":
            # Eq. (7): ε(t) = μΔ(t)/K_Δ
            # self.epsilon = F.relu(dist.mean() / self.k_delta)
            self.epsilon = torch.clip(dist.mean() / self.k_delta, 0, 0.5)
        elif dist_type == "an":
            # Eq. (8): β(t) = 1 + (μ_an(t)-1)/K_an
            self.beta = 1 + (-dist.mean() - 1) / self.k_an
        else:
            raise NotImplementedError(f"not support: {dist_type}")

    def forward(self, logits, labels):
        anchor_idx, positive_idx, negative_idx = get_all_triplets_indices(labels)
        # mat = distance(logits, self.type_of_distance)
        mat = distance(logits.detach(), self.type_of_distance)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        delta = an_dist - ap_dist

        # 1) mining an pairs
        # self.update(an_dist, "an")
        if self.calc_loss_an:
            idxes = unique_filter(anchor_idx, negative_idx)
            an_dist_unique = mat[idxes[0], idxes[1]]
            self.update(an_dist_unique, "an")  # will update beta
            # print("beta", self.beta)

            threshold_condition = -an_dist_unique >= self.beta
            an_pairs = (idxes[0][threshold_condition], idxes[1][threshold_condition])
        else:
            an_pairs = None

        # 2) mining triplets
        self.update(delta, "ap_an")  # will update epsilon
        # print("epsilon", self.epsilon)

        if self.type_of_triplets == "easy":
            threshold_condition = delta > self.epsilon
        else:
            threshold_condition = delta <= self.epsilon
            if self.type_of_triplets == "hard":
                threshold_condition &= delta <= 0
            elif self.type_of_triplets == "semi-hard":
                threshold_condition &= delta > 0
            else:
                pass  # here is "all"

        triplets = (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )

        return triplets, an_pairs


def get_all_quadruplets(labels, need_jk=False):
    """
    get quadruplets like (A, P1, P2, N)
    just fit AP1, AP2, AN & P1!=P2
    note P1P2 may negative, P1N, P2N may positive, just leave another quadruplet to constrain the margin
    Args:
        multi-hot labels
    Returns:
        quadruplets: shape is n_quadruplets x 4
        Sjk: see "C. Multi-Label Based Hashing" of paper
    """
    sames = (labels @ labels.T > 0).byte()
    diffs = sames ^ 1
    sames.fill_diagonal_(0)

    # mining anchor, positive1, positive2
    I, J, K = torch.where(
        sames.unsqueeze(2) * sames.unsqueeze(1) * torch.triu(1 - torch.eye(sames.shape[0], device=labels.device))
    )

    if I.numel() == 0:
        # print("I is None")
        return None

    # finding negatives & gen quadruplets
    N = diffs[I].nonzero()
    if N.numel() == 0:
        # print("N is None")
        return None
    idx = N[:, 0]
    quadruplets = torch.hstack((I[idx].unsqueeze(1), J[idx].unsqueeze(1), K[idx].unsqueeze(1), N[:, 1].unsqueeze(1)))
    # assert (sames[J[idx], K[idx]] == (labels @ labels.T > 0).byte()[J[idx], K[idx]]).all()
    return quadruplets if not need_jk else (quadruplets, sames[J[idx], K[idx]])


class QuadrupletMarginMiner(nn.Module):
    """
    Returns quadruplets that violate the margin
    """

    def __init__(self, margin, type_of_distance, type_of_quadruplets, what_is_hard):
        super().__init__()
        self.margin = margin
        self.type_of_distance = type_of_distance
        self.type_of_quadruplets = type_of_quadruplets
        self.what_is_hard = what_is_hard

    def forward(self, logits, labels):
        quadruplets = get_all_quadruplets(labels)
        if quadruplets is None:
            return None
        # mat = distance(logits, self.type_of_distance)
        mat = distance(logits.detach(), self.type_of_distance)
        I, J, K, N = quadruplets[:, 0], quadruplets[:, 1], quadruplets[:, 2], quadruplets[:, 3]
        ij_dists = mat[I, J]
        ik_dists = mat[I, K]
        in_dists = mat[I, N]
        margin1 = in_dists - ij_dists
        margin2 = in_dists - ik_dists
        # print("margin1:", margin1 <= self.margin)
        # print("margin2:", margin2 <= self.margin)
        # violation1 = ij_dists - in_dists + self.margin
        # violation2 = ik_dists - in_dists + self.margin
        # print("violation:", nn.functional.relu(violation1)+nn.functional.relu(violation2))

        if self.what_is_hard == "one":
            opt = lambda x, y: x | y
        elif self.what_is_hard == "all":
            opt = lambda x, y: x & y
        else:
            raise NotImplementedError(f"not support: {self.what_is_hard}")

        if self.type_of_quadruplets == "easy":
            threshold_condition = opt(margin1 > self.margin, margin2 > self.margin)
        else:
            threshold_condition = opt(margin1 <= self.margin, margin2 <= self.margin)
            if self.type_of_quadruplets == "hard":
                threshold_condition &= opt(margin1 <= 0, margin2 <= 0)
            elif self.type_of_quadruplets == "semi-hard":
                threshold_condition &= opt(margin1 > 0, margin2 > 0)
            else:
                pass  # here is "all"
        if not threshold_condition.any():
            return None
        return quadruplets[threshold_condition]


class AdaQuadrupletMiner(nn.Module):
    """
    Returns quadruplets that violate the margin
    """

    def __init__(self, args: Namespace):
        super().__init__()
        self.type_of_distance = args.type_of_distance
        self.type_of_quadruplets = args.type_of_quadruplets
        self.what_is_hard = args.what_is_hard

        self.epsilon = args.epsilon
        self.k_delta = args.k_delta

    def update(self, dist, dist_type):
        # dist = dist.cpu().numpy()
        if dist_type == "ap_an":
            # Eq. (7): ε(t) = μΔ(t)/K_Δ
            self.epsilon = nn.functional.relu(dist.mean() / self.k_delta)
        else:
            raise NotImplementedError(f"not support: {dist_type}")

    def forward(self, logits, labels):
        quadruplets = get_all_quadruplets(labels)
        if quadruplets is None:
            return None

        # mat = distance(logits, self.type_of_distance)
        mat = distance(logits.detach(), self.type_of_distance)
        I, J, K, N = quadruplets[:, 0], quadruplets[:, 1], quadruplets[:, 2], quadruplets[:, 3]
        ij_dists = mat[I, J]
        ik_dists = mat[I, K]
        in_dists = mat[I, N]
        margin1 = in_dists - ij_dists
        margin2 = in_dists - ik_dists

        delta = torch.cat((margin1, margin2))
        self.update(delta, "ap_an")

        if self.what_is_hard == "one":
            opt = lambda x, y: x | y
        elif self.what_is_hard == "all":
            opt = lambda x, y: x & y
        else:
            raise NotImplementedError(f"not support: {self.what_is_hard}")

        if self.type_of_quadruplets == "easy":
            threshold_condition = opt(margin1 > self.epsilon, margin2 > self.epsilon)
        else:
            threshold_condition = opt(margin1 <= self.epsilon, margin2 <= self.epsilon)
            if self.type_of_quadruplets == "hard":
                threshold_condition &= opt(margin1 <= 0, margin2 <= 0)
            elif self.type_of_quadruplets == "semi-hard":
                threshold_condition &= opt(margin1 > 0, margin2 > 0)
            else:
                pass  # here is "all"

        if not threshold_condition.any():
            return None

        return quadruplets[threshold_condition]


if __name__ == "__main__":
    _args = Namespace(
        type_of_distance="cosine",
        type_of_triplets="all",
        epsilon=0.25,
        beta=0,
        k_delta=2,
        k_an=2,
    )
    _miner = AdaTripletMiner(_args)

    _logits = torch.randn(5, 16)
    _labels = (torch.randn(5, 10) > 0.8).byte()

    print(_miner(_logits, _labels))
