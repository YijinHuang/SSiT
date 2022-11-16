import os
import time
import random
import argparse

import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from funcs import *
from vits import archs


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, help='ddr / aptos2019 / messidor2')
parser.add_argument('--arch', type=str, default='ViT-S-p16', help='model architecture')
parser.add_argument('--data-path', type=str, help='dataset folder')
parser.add_argument('--save-path', type=str, default='./eval_checkpoints', help='save path')
parser.add_argument('--log-path', type=str, default='./log', help='log path')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
parser.add_argument('--checkpoint-key', type=str, default='base_encoder', help='base_encoder / momentum_encoder')
parser.add_argument('--linear', action='store_true', help='use linear eval')
parser.add_argument('--num-classes', type=int, default=5, help='number of classes')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device', type=str, default='cuda', help='device')

parser.add_argument('--epochs', type=int, default=25, help='number of epochs')
parser.add_argument('--input-size', type=int, default=384, help='input size')
parser.add_argument('--learning-rate', type=float, default=0.00002, help='learning rate')
parser.add_argument('--criterion', type=str, default='mse', help='mse / ce')
parser.add_argument('--optimizer', type=str, default='ADAM', help='optimizer')
parser.add_argument('--weight-decay', type=float, default=0.00001, help='weight decay')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--kappa-prior', action='store_true', help='use kappa as best model indicator')
parser.add_argument('--eval-interval', type=int, default=1, help='the epoch interval of evaluating model on val dataset')
parser.add_argument('--save-interval', type=int, default=5, help='the epoch interval of saving model')
parser.add_argument('--disable-progress', action='store_true', help='disable progress bar')


def main():
    args = parser.parse_args()

    save_path = args.save_path
    log_path = args.log_path
    if log_path is None:
        log_path = os.path.join(save_path, 'log')
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(log_path)

    set_random_seed(args.seed)
    model = generate_model(args)
    train_dataset, test_dataset, val_dataset = generate_dataset(args)
    estimator = Estimator(args.criterion, args.num_classes)
    scaler = torch.cuda.amp.GradScaler()
    train(
        args=args,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,
        logger=logger,
        scaler=scaler
    )

    # test
    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(save_path, 'best_validation_weights.pt')
    evaluate(args, model, checkpoint, test_dataset, estimator)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, 'final_weights.pt')
    evaluate(args, model, checkpoint, test_dataset, estimator)


def train(args, model, train_dataset, val_dataset, estimator, logger=None, scaler=None):
    device = args.device
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    loss_function = nn.MSELoss() if args.criterion == 'mse' else nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # start training
    model.train()
    max_indicator = 0
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        estimator.reset()
        progress = tqdm(enumerate(train_loader)) if not args.disable_progress else enumerate(train_loader)
        for step, train_data in progress:
            X, y = train_data
            X, y = X.to(device), y.to(device).float()

            if scaler is not None:
                with torch.cuda.amp.autocast(True):
                    y_pred, _ = model(X)
                    y_pred = y_pred.squeeze() if args.criterion == 'mse' else y_pred
                    loss = loss_function(y_pred, y)

                optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1)

                scaler.step(optimizer)
                scaler.update()
            else:
                y_pred = model(X)
                y_pred = y_pred.squeeze() if args.criterion == 'mse' else y_pred
                loss = loss_function(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            estimator.update(y_pred, y)
            avg_acc = estimator.get_accuracy(6)
            avg_kappa = estimator.get_kappa(6)

            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            message = '[{}] epoch: [{} / {}], loss: {:.6f}, acc: {:.4f}, kappa: {:.4f}'.format(current_time, epoch + 1, args.epochs, avg_loss, avg_acc, avg_kappa)
            if not args.disable_progress:
                progress.set_description(message)

        if args.disable_progress:
            print(message)

        # validation performance
        if epoch % args.eval_interval == 0:
            eval(model, val_loader, estimator, device)
            acc = estimator.get_accuracy(6)
            kappa = estimator.get_kappa(6)
            print('validation accuracy: {}, kappa: {}'.format(acc, kappa))
            if logger:
                logger.add_scalar('validation accuracy', acc, epoch)
                logger.add_scalar('validation kappa', kappa, epoch)

            # save model
            indicator = kappa if args.kappa_prior else acc
            if indicator > max_indicator:
                torch.save(
                    model.state_dict(), 
                    os.path.join(args.save_path, 'best_validation_weights.pt')
                )
                max_indicator = indicator
                print_msg('Best in validation set. Model save at {}'.format(args.save_path))

        if epoch % args.save_interval == 0:
            torch.save(
                model.state_dict(), 
                os.path.join(args.save_path, 'epoch_{}.pt'.format(epoch))
            )

        # update learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler:
            lr_scheduler.step()

        # record
        if logger:
            logger.add_scalar('training loss', avg_loss, epoch)
            logger.add_scalar('training accuracy', avg_acc, epoch)
            logger.add_scalar('training kappa', avg_kappa, epoch)
            logger.add_scalar('learning rate', curr_lr, epoch)

    # save final model
    torch.save(
        model.state_dict(), 
        os.path.join(args.save_path, 'final_weights.pt')
    )

    if logger:
        logger.close()


def evaluate(args, model, checkpoint, test_dataset, estimator):
    weights = torch.load(checkpoint)
    model.load_state_dict(weights, strict=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    print('Running on Test set...')
    eval(model, test_loader, estimator, args.device)

    print('========================================')
    print('Finished! test acc: {}'.format(estimator.get_accuracy(6)))
    print('Confusion Matrix:')
    print(estimator.conf_mat)
    print('quadratic kappa: {}'.format(estimator.get_kappa(6)))
    print('========================================')


def eval(model, dataloader, estimator, device):
    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    for test_data in dataloader:
        X, y = test_data
        X, y = X.to(device), y.to(device).float()

        y_pred, _ = model(X)
        estimator.update(y_pred, y)

    model.train()
    torch.set_grad_enabled(True)


def generate_model(args):
    assert args.arch in archs.keys(), 'Not implemented architecture.'
    out_features = 1 if args.criterion == 'mse' else args.num_classes
    model = archs[args.arch](
        num_classes=out_features,
        img_size=args.input_size,
        feat_concat=True
    )

    linear_key = 'head'
    checkpoint_key = args.checkpoint_key
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, checkpoint_key, linear_key)
    else:
        print_msg('No checkpoint provided. Training from scratch.')

    if args.linear:
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['%s.weight' % linear_key, '%s.bias' % linear_key]:
                param.requires_grad = False
        # init the fc layer
        nn.init.normal_(getattr(model, linear_key).weight, mean=0.0, std=0.01 )
        nn.init.constant_(getattr(model, linear_key).bias, 0)

    model = model.to(args.device)
    return model


def generate_dataset(args):
    train_transform, test_transform = data_transforms(args)
    train_path = os.path.join(args.data_path, 'train')
    test_path = os.path.join(args.data_path, 'test')
    val_path = os.path.join(args.data_path, 'val')

    train_dataset = datasets.ImageFolder(train_path, train_transform, loader=pil_loader)
    test_dataset = datasets.ImageFolder(test_path, test_transform, loader=pil_loader)
    val_dataset = datasets.ImageFolder(val_path, test_transform, loader=pil_loader)

    dataset = train_dataset, test_dataset, val_dataset

    print_dataset_info(dataset)
    return dataset


def data_transforms(args):
    mean, std = get_dataset_stats(args.dataset)
    augmentations = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(
            size=(args.input_size, args.input_size),
            scale=(0.87, 1.15),
            ratio=(0.7, 1.3)
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.1
        ),
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ]

    normalization = [
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    train_preprocess = transforms.Compose([
        *augmentations,
        *normalization
    ])

    test_preprocess = transforms.Compose(normalization)
    return train_preprocess, test_preprocess


class Estimator():
    def __init__(self, criterion, num_classes, thresholds=None):
        self.criterion = criterion
        self.num_classes = num_classes
        self.thresholds = [-0.5 + i for i in range(num_classes)] if not thresholds else thresholds

        self.reset()  # intitialization

    def update(self, predictions, targets):
        targets = targets.cpu()
        predictions = predictions.cpu()
        predictions = self.to_prediction(predictions)

        # update metrics
        self.num_samples += len(predictions)
        self.correct += (predictions == targets).sum().item()
        for i, p in enumerate(predictions):
            self.conf_mat[int(targets[i])][int(p.item())] += 1

    def get_accuracy(self, digits=-1):
        acc = self.correct / self.num_samples
        acc = acc if digits == -1 else round(acc, digits)
        return acc

    def get_kappa(self, digits=-1):
        kappa = quadratic_weighted_kappa(self.conf_mat)
        kappa = kappa if digits == -1 else round(kappa, digits)
        return kappa

    def reset(self):
        self.correct = 0
        self.num_samples = 0
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def to_prediction(self, predictions):
        if self.criterion == 'ce':
            predictions = torch.tensor(
                [torch.argmax(p) for p in predictions]
            ).long()
        elif self.criterion == 'mse':
            predictions = torch.tensor(
                [self.classify(p.item()) for p in predictions]
            ).float()
        else:
            raise NotImplementedError('Not implemented criterion.')

        return predictions

    def classify(self, predict):
        thresholds = self.thresholds
        predict = max(predict, thresholds[0])
        for i in reversed(range(len(thresholds))):
            if predict >= thresholds[i]:
                return i


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
