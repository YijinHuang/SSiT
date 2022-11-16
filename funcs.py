import torch
import numpy as np
from vits import resize_pos_embed


def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)


def print_config(args):
    print('=================================')
    for key, value in args.__dict__.items():
        print('{}: {}'.format(key, value))
    print('=================================')


def print_dataset_info(datasets):
    train_dataset, test_dataset, val_dataset = datasets
    print('=========================')
    print('Dataset Loaded.')
    print('Categories:\t{}'.format(len(train_dataset.classes)))
    print('Training:\t{}'.format(len(train_dataset)))
    print('Validation:\t{}'.format(len(val_dataset)))
    print('Test:\t\t{}'.format(len(test_dataset)))
    print('=========================')


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def is_main(args):
    return (not args.distributed) or args.rank == 0


def to_devices(args, *tensors):
    if args.distributed:
        return [tensor.cuda(args.gpu) for tensor in tensors]
    else:
        return [tensor.to(args.device) for tensor in tensors]


def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()
    return (observed - expected) / (1 - expected)


def load_checkpoint(model, checkpoint_path, checkpoint_key, linear_key):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint.state_dict()
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith(checkpoint_key) and not k.startswith('%s.%s' % (checkpoint_key, linear_key)):
            # remove prefix
            state_dict[k[len("%s." % checkpoint_key):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # position embedding
    pos_embed_w = state_dict['pos_embed']
    pos_embed_w = resize_pos_embed(pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    state_dict['pos_embed'] = pos_embed_w

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"%s.weight" % linear_key, "%s.bias" % linear_key}
    print_msg('Load weights form {}'.format(checkpoint_path))


def get_dataset_stats(dataset):
    # mean and std from train set
    dataset_stats = {
        'ddr': (
            [0.423737496137619, 0.2609460651874542, 0.128403902053833], 
            [0.29482534527778625, 0.20167365670204163, 0.13668020069599152]
        ),
        'aptos2019': (
            [0.46100369095802307, 0.246780663728714, 0.07989078760147095],
            [0.24873991310596466, 0.13842609524726868, 0.08025242388248444]
        ),
        'messidor2': (
            [0.48436370491981506, 0.2238118201494217, 0.07583174854516983], 
            [0.2939208149909973, 0.14721707999706268, 0.06350880116224289]
        )
    }
    if dataset in dataset_stats.keys():
        mean, std = dataset_stats[dataset]
    else:
        raise NotImplementedError(
            'Not implemented dataset: {}. '
            'Please specify the dataset name [--dataset ddr / aptos2019 / messidor2]. '
            'If you are training on the customized dataset, '
            'please add the mean and std of your dataset in dataset_stats in funcs.py.'.format(dataset)
        )
    return mean, std
