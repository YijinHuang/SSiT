import os
import math
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.distributed import all_reduce, ReduceOp

from funcs import is_main, to_devices, print_msg


def train(args, model, train_dataset, logger=None, scaler=None):
    optimizer = initialize_optimizer(args, model)
    train_sampler = initialize_sampler(args, train_dataset) if args.distributed == True else None
    train_loader = initialize_dataloader(args, train_dataset, train_sampler)

    if args.resume:
        resume(args, model, optimizer, scaler)

    # start training
    model.train()
    avg_cl_loss = 0
    avg_ss_loss = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        epoch_cl_loss = 0
        epoch_ss_loss = 0
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        progress = enumerate(train_loader)
        if is_main(args) and not args.disable_progress:
            progress = tqdm(progress)
        for step, train_data in progress:
            scheduler_step = epoch + step / len(train_loader)
            lr = adjust_learning_rate(args, optimizer, scheduler_step)
            moco_m = adjust_moco_momentum(args, scheduler_step)
            ss = adjust_lambda_ss(args, scheduler_step) if args.ss_decay else args.ss

            X1, X2, M1, M2 = train_data
            X1, X2, M1, M2 = to_devices(args, X1, X2, M1, M2)

            # forward
            with torch.cuda.amp.autocast(True):
                cl_loss, ss_loss = model(X1, X2, M1, M2, moco_m)
                loss = args.cl * cl_loss + ss * ss_loss

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            scaler.step(optimizer)
            scaler.update()

            if args.distributed:
                all_reduce(cl_loss, ReduceOp.AVG)
                all_reduce(ss_loss, ReduceOp.AVG)

            # metrics
            if is_main(args):
                epoch_cl_loss += cl_loss
                epoch_ss_loss += ss_loss
                avg_cl_loss = epoch_cl_loss / (step + 1)
                avg_ss_loss = epoch_ss_loss / (step + 1)

                message = '[{}] epoch: {}/{}, cl loss: {:.6f}, ss loss: {:.6f}, lr: {:.6f}, moco_m: {:.6f}'.format(
                    current_time, epoch + 1, args.epochs, avg_cl_loss, avg_ss_loss, lr, moco_m)
                if not args.disable_progress:
                    progress.set_description(message)

        if is_main(args) and args.disable_progress:
            print(message)

        if is_main(args) and (epoch + 1) % args.save_interval == 0 and (epoch + 1) < args.epochs:
            save_checkpoint(args, epoch, model, optimizer, scaler)

        # record
        if is_main(args) and logger:
            logger.add_scalar('contrastive loss', avg_cl_loss, epoch)
            logger.add_scalar('saliency segmentation loss', avg_ss_loss, epoch)
            logger.add_scalar('learning rate', lr, epoch)
            logger.add_scalar('moco momentum', moco_m, epoch)

    # save final model
    if is_main(args):
        save_checkpoint(args, epoch, model, optimizer, scaler)
        if logger:
            logger.close()


# define data loader
def initialize_dataloader(args, train_dataset, train_sampler):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    return train_loader


# define optmizer
def initialize_optimizer(args, model):
    optimizer_strategy = args.optimizer
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    momentum = args.momentum

    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAMW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer


def initialize_sampler(args, train_dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank
    )
    return train_sampler


def adjust_learning_rate(args, optimizer, epoch):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.learning_rate * epoch / args.warmup_epochs
    else:
        lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(args, epoch):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


def adjust_lambda_ss(args, epoch):
    """Adjust moco momentum based on current epoch"""
    ss = args.ss * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    return ss


def save_checkpoint(args, epoch, model, optimizer, scaler):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scaler': scaler.state_dict(),
    }
    model = model.module if args.distributed else model

    torch.save(checkpoint, os.path.join(args.save_path, 'checkpoint.pt'))
    torch.save(model, os.path.join(args.save_path, 'epoch_{}.pt'.format(epoch + 1)))
    print_msg('Saved checkpoint to {}'.format(args.save_path))


def resume(args, model, optimizer, scaler):
    checkpoint_path = os.path.join(args.save_path, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        print_msg('Loading checkpoint {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        print_msg('Loaded checkpoint {} from epoch {}'.format(checkpoint_path, checkpoint['epoch']))
    else:
        print_msg('No checkpoint found at {}'.format(checkpoint_path))