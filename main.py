import os
import random
import shutil
import argparse
import builtins

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from train import train
from ssit import build_model
from data import build_dataset
from funcs import print_config, is_main


parser = argparse.ArgumentParser()
# base setting
parser.add_argument('--arch', '--architecture', type=str, default='ViT-S-p16', help='network architecture, shoud be in archs in vits.py')
parser.add_argument('--data-index', type=str, default='./data_index/pretraing_dataset.pkl', help='pickle file with lesion predicted results')
parser.add_argument('--save-path', type=str, default='./checkpoints', help='path to save checkpoints')
parser.add_argument('--record-path', type=str, default=None, help='path to save log')
parser.add_argument('--pretrained', action='store_true', help='load pretrained parameters in ImageNet')
parser.add_argument('--device', type=str, default='cuda', help='only support cuda')
parser.add_argument('--seed', type=int, default=-1, help='random seed for reproducibilty. Set to -1 to disable.')
parser.add_argument('--resume', action='store_true', help='resume training from the latest checkpoint')

# DDP setting
parser.add_argument('--distributed', action='store_true', help='distributed training')
parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes for distributed training')
parser.add_argument('--n-gpus', type=int, default=None, help='number of gpus per node')
parser.add_argument('--addr', type=str, default='127.0.0.1', help='master address')
parser.add_argument('--port', type=str, default='28888', help='master port')
parser.add_argument('--rank', type=int, default=0, help='rank of current process')

# training setting
parser.add_argument('--input-size', type=int, default=224, help='input size')
parser.add_argument('--start-epoch', type=int, default=0, help='start epoch for training')
parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
parser.add_argument('--warmup-epochs', type=int, default=40, help='number of warmup epochs')
parser.add_argument('--mask-ratio', type=float, default=0.25, help='ratio of masked pixels')
parser.add_argument('--disable-progress', action='store_true', help='do not show progress bar')
parser.add_argument('--ss', type=float, default=10, help='weight of saliency segmentation loss')
parser.add_argument('--cl', type=float, default=1, help='weight of contrastive learning loss')
parser.add_argument('--saliency-threshold', type=float, default=0.5, help='threshold for saliency map')
parser.add_argument('--batch-size', type=int, default=512, help='total training batch size')
parser.add_argument('--optimizer', type=str, default='ADAMW', help='SGD / ADAM / ADAMW')
parser.add_argument('--moco-m', type=float, default=0.99, help='momentum for moco')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature for moco')
parser.add_argument('--learning-rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
parser.add_argument('--weight-decay', type=float, default=0.1, help='weight decay for SGD and ADAM')
parser.add_argument('--num-workers', type=int, default=32, help='total number of workers')
parser.add_argument('--save-interval', type=int, default=20, help='number of interval to store model and checkpoint')
parser.add_argument('--pool-mode', type=str, default='max', help="'max' / 'avg', pooling mode for saliency map patch")


def main():
    # print configuration
    args = parser.parse_args()
    print_config(args)

    # create folder
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # create logger
    record_path = args.record_path
    if record_path is None:
        record_path = os.path.join(save_path, 'log')

    n_gpus = args.n_gpus if args.n_gpus else torch.cuda.device_count()
    if args.distributed:
        args.world_size = n_gpus * args.nodes
        os.environ['MASTER_ADDR'] = args.addr
        os.environ['MASTER_PORT'] = args.port
        mp.spawn(worker, nprocs=n_gpus, args=(n_gpus, args))


def worker(gpu, n_gpus, args):
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.gpu = gpu
        args.rank = args.rank * n_gpus + gpu
        dist.init_process_group(
            backend=args.backend,
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank
        )
        torch.distributed.barrier()

        args.batch_size = int(args.batch_size / args.world_size)
        args.num_workers = int((args.num_workers + n_gpus - 1) / n_gpus)

        # suppress printing
        if args.gpu != 0 or args.rank != 0:
            def print_pass(*args):
                pass
            builtins.print = print_pass

    if args.seed >= 0:
        set_random_seed(args.seed + args.rank)

    model = build_model(args)
    train_dataset = build_dataset(args)
    logger = SummaryWriter(args.record_path) if is_main(args) else None
    scaler = torch.cuda.amp.GradScaler()

    train(
        args=args,
        model=model,
        train_dataset=train_dataset,
        logger=logger,
        scaler=scaler
    )
    torch.distributed.barrier()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
