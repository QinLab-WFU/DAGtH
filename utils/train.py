import json
import os
import time
from copy import deepcopy

import torch
from loguru import logger
from timm.utils import AverageMeter, random_seed

from BaseLine.network import build_model
from BaseLine.utils import build_optimizer
from _data import build_loader, get_topk, get_class_num
from _utils import prediction, mean_average_precision, calc_learnable_params, init
from config import get_config
from loss import AdaTripletLoss


def train_epoch(args, dataloader, net, criterion, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["n_triplets", "n_an_pairs", "epsilon", "beta", "loss", "mAP"]:
        stat_meters[x] = AverageMeter()

    net.train()
    for images, labels, _ in dataloader:
        images, labels = images.cuda(), labels.cuda()
        logits = net(images)

        loss, n_triplets, n_an_pairs, epsilon, beta = criterion(logits, labels)

        stat_meters["n_triplets"].update(n_triplets)
        stat_meters["n_an_pairs"].update(n_an_pairs)
        stat_meters["epsilon"].update(epsilon)
        stat_meters["beta"].update(beta)

        if n_triplets + n_an_pairs == 0:
            continue

        stat_meters["loss"].update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to check overfitting
        q_cnt = labels.shape[0] // 10
        map_k = mean_average_precision(logits[:q_cnt], logits[q_cnt:], labels[:q_cnt], labels[q_cnt:])
        stat_meters["mAP"].update(map_k)

    toc = time.time()
    stat_str = ""
    for x in stat_meters.keys():
        stat_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"

    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{stat_str}"
    )


def train_val(args, train_loader, query_loader, dbase_loader, logger):
    # setup net
    net = build_model(args, True)
    logger.info(f"number of net's params: {calc_learnable_params(net)}")

    # setup criterion
    criterion = AdaTripletLoss(args, True)

    # setup optimizer
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training process
    best_map = 0.0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, criterion, optimizer, epoch)

        # we monitor MAP@R validation accuracy every 5 epochs
        # and use early stopping patience of 10 to get the best params of model
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            qB, qL = prediction(net, query_loader)
            rB, rL = prediction(net, dbase_loader)
            map_k = mean_average_precision(qB, rB, qL, rL, args.topk)
            del qB, qL, rB, rL
            logger.info(
                f"[Evaluating][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][best-mAP@{args.topk}:{best_map:.4f}][mAP@{args.topk}:{map_k:.4f}][count:{0 if map_k > best_map else (count + 1)}]"
            )

            if map_k > best_map:
                best_map = map_k
                best_epoch = epoch
                best_checkpoint = deepcopy(net.state_dict())
                count = 0
            else:
                count += 1
                if count == 10:
                    logger.info(
                        f"without improvement, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}"
                    )
                    torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logger.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_map


def prepare_loaders(args, bl_func):
    train_loader, query_loader, dbase_loader = (
        bl_func(
            args.data_dir,
            args.dataset,
            "train",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            drop_last=True,
        ),
        bl_func(args.data_dir, args.dataset, "query", None, batch_size=args.batch_size, num_workers=args.n_workers),
        bl_func(args.data_dir, args.dataset, "dbase", None, batch_size=args.batch_size, num_workers=args.n_workers),
    )
    return train_loader, query_loader, dbase_loader


def main():
    init("0")
    args = get_config()

    dummy_logger_id = None
    rst = []
    for dataset in ["cifar", "nuswide", "flickr", "coco"]:
        # for dataset in ["nuswide", "coco"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = prepare_loaders(args, build_loader)

        # for hash_bit in [16, 32, 64, 128]:
        for hash_bit in [32]:
            random_seed()
            print(f"processing hash-bit: {hash_bit}")
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(fname.endswith(".pkl") for fname in os.listdir(args.save_dir)):
                raise Exception(f"*.pkl exists in {args.save_dir}")

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", rotation="500 MB", level="INFO")

            with open(f"{args.save_dir}/config.json", "w+") as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            best_epoch, best_map = train_val(args, train_loader, query_loader, dbase_loader, logger)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})
    for x in rst:
        print(
            f"[dataset:{x['dataset']}][bits:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
        )


if __name__ == "__main__":
    main()
