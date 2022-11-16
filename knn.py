# =======================================================================
# Based on https://github.com/facebookresearch/dino/blob/main/eval_knn.py
# =======================================================================

import os
import argparse

import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms as pth_transforms

from vits import archs
from funcs import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset to evaluate (ddr / messidor2 / aptos2019).')
parser.add_argument('--arch', default='ViT-S-p16', type=str, help='Architecture (support only ViT).')
parser.add_argument('--data-path', type=str, help='Path to the fundus dataset.')
parser.add_argument('--checkpoint', default='', type=str, help="Path to pretrained weights to evaluate.")
parser.add_argument('--input-size', default=384, type=int, help='Input size of the model.')
parser.add_argument('--batch-size', default=16, type=int, help='Per-GPU batch-size')
parser.add_argument('--nb-knn', default=[5, 10, 20], nargs='+', type=int,
    help='Number of NN to use. 20 is usually working the best.')
parser.add_argument('--temperature', default=0.07, type=float,
    help='Temperature used in the voting coefficient')
parser.add_argument('--network', default='', type=str, help="network")
parser.add_argument('--use-cuda', default=True,
    help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
parser.add_argument('--patch-size', default=16, type=int, help='Patch resolution of the model.')
parser.add_argument("--checkpoint-key", default="base_encoder", type=str,
    help='Key to use in the checkpoint')
parser.add_argument('--dump-features', default=None,
    help='Path where to save computed features, empty for no saving')
parser.add_argument('--load-features', default=None, help="""If the features have
    already been computed, where to find them.""")
parser.add_argument('--num-workers', default=12, type=int, help='Number of data loading workers per GPU.')
parser.add_argument('--device', default='cuda', type=str, help='Device to use.')


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    mean, std = get_dataset_stats(args.dataset)
    transform = pth_transforms.Compose([
        pth_transforms.Resize((args.input_size, args.input_size)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])

    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "test"), transform=transform)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} test imgs.")

    # ============ building network ... ============
    model = archs[args.arch](
        pretrained=False,
        num_classes=1,
        img_size=args.input_size,
    )

    linear_key = 'head'
    checkpoint_key = args.checkpoint_key
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, checkpoint_key, linear_key)
    else:
        print_msg('No checkpoint provided. Training from scratch.')

    model = model.to(args.device)
    model.eval()

    for _, param in model.named_parameters():
        param.requires_grad = False

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print("Extracting features for test set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)

    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    # save features and labels
    if args.dump_features:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    features = None
    for samples, index in data_loader:
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = multi_scale(samples, model)
        else:
            feats = model.forward_features(samples).clone()[:,0]

        # init storage feature matrix
        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # update storage feature matrix
        if use_cuda:
            features.index_copy_(0, index, feats)
        else:
            features.index_copy_(0, index.cpu(), feats.cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=5):
    top1, total = 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 50
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)

    conf_mat = torch.zeros(num_classes, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)

        # kappa 
        tgt = targets.data.view(-1, 1)
        for i, p in enumerate(predictions.narrow(1, 0, 1)):
            conf_mat[int(tgt[i])][int(p.item())] += 1

    conf_mat = conf_mat.cpu().numpy()
    top1 = top1 * 100.0 / total
    kappa = quadratic_weighted_kappa(conf_mat)
    f1 = conf_mat[1][1] / (conf_mat[1][1] + 0.5 * (conf_mat[0][1] + conf_mat[1][0]))
    return top1, kappa, f1


def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model.forward_features(inp)[:,0].clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return 


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    args = parser.parse_args()

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    train_features = train_features.cuda()
    test_features = test_features.cuda()
    train_labels = train_labels.cuda()
    test_labels = test_labels.cuda()

    print("Features are ready!\nStart the k-NN classification.")
    print("Evaluating on {}:".format(args.dataset))
    for k in args.nb_knn:
        acc, kappa, _ = knn_classifier(train_features, train_labels,
            test_features, test_labels, k, args.temperature)
        print(f"{k}-NN classifier result: Acc: {acc}, Kappa: {kappa}")
