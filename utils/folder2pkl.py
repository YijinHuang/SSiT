import os
import glob
import pickle
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--output-file', type=str, default='../data_index/pretraining_dataset.pkl', help='path to save dataset index')
parser.add_argument('--image-folder', type=str, help='path to image folder')
parser.add_argument('--saliency-folder', type=str, help='path to saliency folder')


def folder2pkl():
    args = parser.parse_args()
    image_folder = Path(args.image_folder)
    saliency_folder = Path(args.saliency_folder)
    output_file = Path(args.output_file)

    # sort images by name
    image_paths = sorted(image_folder.glob('**/*.*'), key=lambda x: x.stem)
    saliency_paths = sorted(saliency_folder.glob('**/*.*'), key=lambda x: x.stem)
    assert len(image_paths) == len(saliency_paths), 'Number of images and number of saliency maps are not equal'

    # absolute path
    image_paths = [path.absolute() for path in image_paths]
    saliency_paths = [path.absolute() for path in saliency_paths]

    dataset = list(zip(image_paths, saliency_paths))
    for image_path, saliency_path in dataset:
        image_id = image_path.stem
        saliency_id = saliency_path.stem
        assert image_id == saliency_id, 'Image id {} and saliency id {} do not match'.format(image_id, saliency_id)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    folder2pkl()
