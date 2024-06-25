import os
import time
import math
import random
import pickle
import argparse

import torch
import cv2 as cv
import numpy as np
import torch.nn as nn
import albumentations as A
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists

from funcs import *
from vits import archs


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, help='drive / idrid')
parser.add_argument('--arch', type=str, default='ViT-S-p16', help='model architecture')
parser.add_argument('--data-index', type=str, help='dataset index')
parser.add_argument('--save-path', type=str, default='./eval_checkpoints', help='save path')
parser.add_argument('--log-path', type=str, default=None, help='log path')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
parser.add_argument('--checkpoint-key', type=str, default='base_encoder', help='base_encoder / momentum_encoder')
parser.add_argument('--linear', action='store_true', help='use linear eval')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device', type=str, default='cuda', help='device')

parser.add_argument('--iterations', type=int, default=2000, help='number of iterations')
parser.add_argument('--warmup-iterations', type=int, default=200, help='number of warmup iterations')
parser.add_argument('--input-size', type=int, default=512, help='input size')
parser.add_argument('--patch-size', type=int, default=256, help='input size')
parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--optimizer', type=str, default='ADAM', help='optimizer')
parser.add_argument('--weight-decay', type=float, default=0.000001, help='weight decay')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--kappa-prior', action='store_true', help='use kappa as best model indicator')
parser.add_argument('--eval-interval', type=int, default=50, help='the epoch interval of evaluating model on val dataset')
parser.add_argument('--save-interval', type=int, default=500, help='the epoch interval of saving model')
parser.add_argument('--disable-progress', action='store_true', help='disable progress bar')
parser.add_argument('--ce-weight', type=float, default=10, help='weight of cross entropy loss')


def main():
    args = parser.parse_args()

    log_path = args.log_path
    if log_path is None:
        log_path = os.path.join(args.save_path, 'log')
    os.makedirs(args.save_path, exist_ok=True)
    logger = SummaryWriter(log_path)

    set_random_seed(args.seed)
    model = generate_model(args)
    train_dataset, test_dataset, val_dataset = generate_dataset(args)
    estimator = Estimator()
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
    checkpoint = os.path.join(args.save_path, 'best_validation_weights.pt')
    evaluate(args, model, checkpoint, test_dataset, estimator)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(args.save_path, 'final_weights.pt')
    evaluate(args, model, checkpoint, test_dataset, estimator)


def train(args, model, train_dataset, val_dataset, estimator, logger=None, scaler=None):
    device = args.device
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    dice_loss = smp.losses.DiceLoss('binary')
    bce_loss = nn.BCEWithLogitsLoss()
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
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # start training
    model.train()
    max_indicator = 0
    avg_loss, avg_dice = 0, 0
    cum_loss = 0
    data_iter = iter(train_loader)
    for step in range(args.iterations):
        lr = adjust_learning_rate(args, optimizer, step)
        X, y, data_iter = load_sample(train_loader, data_iter)
        X, y = X.to(device), y.to(device).long()

        if scaler is not None:
            with torch.cuda.amp.autocast(True):
                y_pred = model(X).squeeze()
                loss = dice_loss(y_pred, y) + args.ce_weight * bce_loss(y_pred, y.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(X).squeeze()
            loss = dice_loss(y_pred, y) + args.ce_weight * bce_loss(y_pred, y.float())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        # metrics
        cum_loss += loss.item()
        avg_loss = cum_loss / (step + 1)
        estimator.update(y_pred, y)
        
        if (step+1) % 10 == 0:
            avg_dice = estimator.get_dice(6)
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            message = '[{}] step: [{} / {}], loss: {:.6f}, dice: {:.4f}, lr: {:.6f}'.format(current_time, step + 1, args.iterations, avg_loss, avg_dice, lr)
            print(message)
            estimator.reset()
            cum_loss = 0

        # validation performance
        if (step+1) % args.eval_interval == 0:
            eval(args, model, val_loader, estimator, device)
            dice = estimator.get_dice(6)
            print('validation dice: {}'.format(dice))
            if logger:
                logger.add_scalar('validation dice', dice, step)

            # save model
            indicator = dice
            if indicator > max_indicator:
                torch.save(
                    model.state_dict(), 
                    os.path.join(args.save_path, 'best_validation_weights.pt')
                )
                max_indicator = indicator
                print_msg('Best in validation set. Model save at {}'.format(args.save_path))
            
        if (step+1) % args.save_interval == 0:
            torch.save(
                model.state_dict(), 
                os.path.join(args.save_path, 'step_{}.pt'.format(step))
            )

        # record
        if logger:
            logger.add_scalar('training loss', avg_loss, step)
            logger.add_scalar('training dice', avg_dice, step)
            logger.add_scalar('learning rate', lr, step)

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
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    print('Running on Test set...')
    eval(args, model, test_loader, estimator, args.device)

    print('========================================')
    print('Finished! Dice: {}'.format(estimator.get_dice(6)))
    print('========================================')


def eval(args, model, dataloader, estimator, device):
    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    for test_data in dataloader:
        X, y = test_data
        X, y = X.to(device), y.to(device).float()

        X = patchify(X, kernel_size=args.patch_size, stride=args.patch_size // 2)
        y_pred = model(X)
        y_pred = unpatchify(y_pred, kernel_size=args.patch_size, stride=args.patch_size // 2, target_shape=y.shape)

        estimator.update(y_pred, y)

    model.train()
    torch.set_grad_enabled(True)


def generate_model(args):
    assert args.arch in archs.keys(), 'Not implemented architecture.'
    encoder = archs[args.arch](
        img_size=args.patch_size
    )

    linear_key = 'head'
    checkpoint_key = args.checkpoint_key
    if args.checkpoint:
        load_checkpoint(encoder, args.checkpoint, checkpoint_key, linear_key)
    else:
        print_msg('No checkpoint provided. Training from scratch.')

    if args.linear:
        for name, param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()

    model = Segmentor(encoder)
    model = model.to(args.device)
    return model


def generate_dataset(args):
    train_transform, test_transform = data_transforms(args)
    datasets = pickle.load(open(args.data_index, 'rb'))
    print(datasets)

    train_dataset = SegmentationDataset(datasets['train'], train_transform, loader=cv_loader)
    test_dataset = SegmentationDataset(datasets['test'], test_transform, loader=cv_loader)
    val_dataset = SegmentationDataset(datasets['val'], test_transform, loader=cv_loader)

    dataset = train_dataset, test_dataset, val_dataset

    print('train dataset: {}'.format(len(train_dataset)))
    print('test dataset: {}'.format(len(test_dataset)))
    print('val dataset: {}'.format(len(val_dataset)))
    return dataset


def data_transforms(args):
    mean, std = get_dataset_stats(args.dataset)
    train_preprocess = A.Compose([
        A.Resize(args.input_size, args.input_size),
        CropNonEmptyMaskIfExists(args.patch_size, args.patch_size),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    test_preprocess = A.Compose([
        A.Resize(args.input_size, args.input_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    return train_preprocess, test_preprocess


class Segmentor(nn.Module):
    def __init__(self, encoder):
        super(Segmentor, self).__init__()

        self.encoder = encoder
        patch_size = encoder.patch_size
        hidden_dim = encoder.head.weight.shape[1]
        self.segmentor = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=(patch_size ** 2),
                kernel_size=1,
            ),
            nn.PixelShuffle(upscale_factor=patch_size),
        )

    def forward(self, x):
        _, f = self.encoder(x)
        f = f[:, 1:]

        B, L, C = f.shape
        H = W = int(L ** 0.5)
        f = f.permute(0, 2, 1).reshape(B, C, H, W)
        y = self.segmentor(f)
        return y.squeeze()


class Estimator():
    def __init__(self):
        self.reset()  # intitialization

    def update(self, predictions, targets):
        targets = targets.cpu()
        predictions = self.predict(predictions)

        self.num_samples += targets.shape[0]
        self.dice += sum(self.compute_dice(targets, predictions))

    def compute_dice(self, targets, predictions):
        dices = []
        for i in range(targets.shape[0]):
            target = targets[i]
            prediction = predictions[i]
            tp = torch.sum(target * prediction)
            fp = torch.sum(prediction) - tp
            fn = torch.sum(target) - tp
            dice = (2 * tp) / (2 * tp + fp + fn)
            dices.append(dice.item())
        return dices
    
    def predict(self, predictions):
        predictions = torch.sigmoid(predictions)
        predictions = predictions > 0.5
        return predictions.cpu()

    def get_dice(self, digits=-1):
        score = self.dice / self.num_samples
        score = score if digits == -1 else round(score, digits)
        return score

    def reset(self):
        self.dice = 0
        self.num_samples = 0


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, loader=None):
        self.dataset = dataset
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        image = self.loader(image)
        mask = cv.imread(mask).astype(np.uint8)
        mask = mask[:, :, 2] if len(mask.shape) == 3 else mask
        mask = mask / 255

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.dataset)

def cv_loader(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_sample(data_loader, data_iter):
    try:
        X, y = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        X, y = next(data_iter)
    return X, y, data_iter


def patchify(x, kernel_size, stride):
    pad_size = kernel_size - stride
    x = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    x = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    x = x.permute(0, 2, 3, 1, 4, 5)
    x = x.contiguous().view(-1, *x.shape[3:])
    return x


def unpatchify(x, kernel_size, stride, target_shape, num_channel=1):
    B, H, W = target_shape
    C = num_channel
    pad_size = kernel_size - stride

    x = x.contiguous().view(B, -1, C, kernel_size, kernel_size)
    x = x.contiguous().view(B, -1, C * kernel_size * kernel_size)
    x = x.permute(0, 2, 1)
    x = F.fold(x, output_size=(H, W), kernel_size=kernel_size, padding=pad_size, stride=stride)
    x = x / (kernel_size / stride) ** 2
    return x


def adjust_learning_rate(args, optimizer, step):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if step < args.warmup_iterations:
        lr = args.learning_rate * step / args.warmup_iterations
    else:
        lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * (step - args.warmup_iterations) / (args.iterations - args.warmup_iterations)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



if __name__ == '__main__':
    main()

