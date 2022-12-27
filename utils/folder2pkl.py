import os
import glob
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--output-folder', type=str, default='../dataset/', help='path to save dataset index')
parser.add_argument('--image-folder', type=str, help='path to image folder')
parser.add_argument('--saliency-folder', type=str, help='path to saliency folder')


def folder2pkl():
    args = parser.parse_args()
    image_folder = args.image_folder
    saliency_folder = args.saliency_folder
    output_folder = args.output_folder

    # sort images by name
    image_paths = sorted(glob.glob(os.path.join(image_folder, '**', '*.*'), recursive=True))
    saliency_paths = sorted(glob.glob(os.path.join(saliency_folder, '**', '*.*'), recursive=True))
    assert len(image_paths) == len(saliency_paths), 'image number and saliency number are not equal'

    # absolute path
    image_paths = [os.path.abspath(path) for path in image_paths]
    saliency_paths = [os.path.abspath(path) for path in saliency_paths]

    dataset = list(zip(image_paths, saliency_paths))
    for image_path, saliency_path in dataset:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        saliency_id = os.path.splitext(os.path.basename(saliency_path))[0]
        assert image_id == saliency_id, 'image name {} and saliency name {} do not match'.format(image_id, saliency_id)

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'pretraining_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    folder2pkl()
