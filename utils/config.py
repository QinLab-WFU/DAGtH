import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description=os.path.basename(os.path.dirname(__file__)))

    parser.add_argument("--dataset", type=str, default="cifar", help="cifar/nuswide/flickr/coco")
    parser.add_argument("--backbone", type=str, default="resnet50", help="see network.py")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--n_workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--optimizer", type=str, default="adamw", help="adam/rmsprop/adamw/sgd")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--data_dir", type=str, default="../_datasets", help="directory to dataset")
    parser.add_argument("--save_dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n_classes", type=int, default=10, help="number of dataset classes")
    parser.add_argument("--n_bits", type=int, default=16, help="length of hashing binary")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")

    parser.add_argument("--type_of_distance", type=str, default="cosine", help="cosine/euclidean")
    parser.add_argument("--type_of_triplets", type=str, default="all", help="all/semi-hard/hard")

    parser.add_argument("--epsilon", type=float, default=0.25, help="hyper-parameter ε, aka. margin")
    parser.add_argument("--beta", type=float, default=0, help="hyper-parameter β")
    parser.add_argument("--k_delta", type=float, default=2, help="K_Δ of Eq. (7) in paper")
    parser.add_argument("--k_an", type=float, default=2, help="K_an of Eq. (8) in paper")
    parser.add_argument("--la", type=float, default=1, help="loss_w_lambda")

    parser.add_argument("--calc_loss_an", type=bool, default=False, help="calculate L_an?")

    return parser.parse_args()
