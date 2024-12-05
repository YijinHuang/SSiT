import os
import argparse
import cv2 as cv
import numpy as np

from pathlib import Path
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_process', type=int, default=8, help='number of processes')
parser.add_argument('--saliency-model', type=str, default='fine_grained', help='saliency model (fine_grained / spectral_residual)')
parser.add_argument('--image-folder', type=str, help='path to image folder')
parser.add_argument('--output-folder', type=str, help='path to save saliency map')
parser.add_argument('--visualize-folder', type=str, default='', help='path to save saliency map visualizeualization')


circle = np.zeros((512, 512))
circle = cv.circle(circle, (256, 256), 240, 1, -1)


def saliency_detect(i, saliency_model, src_path, output_path, visualize_path):
    image = cv.imread(str(src_path))
    image = preprocess(image)

    if saliency_model == 'fine_grained':
        saliency = cv.saliency.StaticSaliencyFineGrained_create()
    elif saliency_model == 'spectral_residual':
        saliency = cv.saliency.StaticSaliencySpectralResidual_create()
    else:
        raise ValueError('Unknown saliency model: {}'.format(saliency_model))

    (_, raw_saliencyMap) = saliency.computeSaliency(image)
    raw_saliencyMap *= circle

    np.save(output_path, raw_saliencyMap)

    if visualize_path:
        int_saliencyMap = (raw_saliencyMap * 255).astype("uint8")
        cv.imwrite(str(visualize_path), int_saliencyMap)

    if i % 500 == 0:
        print('Processed {} images'.format(i))


def main():
    args = parser.parse_args()
    image_folder = Path(args.image_folder)
    output_folder = Path(args.output_folder)
    visualize_folder = Path(args.visualize_folder)

    i = 0
    res = []
    pool = Pool(processes=args.num_process)
    print('Loading tasks...')
    for folder, _, imgs in os.walk(args.image_folder):
        folder = Path(folder)
        subfolders = folder.relative_to(image_folder)
        output_subfolder = output_folder.joinpath(subfolders)
        output_subfolder.mkdir(parents=True, exist_ok=True)

        if args.visualize_folder:
            visualize_subfolder = visualize_folder.joinpath(subfolders)
            visualize_subfolder.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            i += 1
            src_path = folder.joinpath(img)
            output_path = output_subfolder.joinpath(img).with_suffix('.npy')
            visualize_path = visualize_subfolder.joinpath(img) if args.visualize_folder else ''
            res.append(pool.apply_async(saliency_detect, args=(i, args.saliency_model, src_path, output_path, visualize_path)))

    print('Waiting for all subprocesses done...')
    for re in res:
        re.get()
    pool.close()
    pool.join()
    print('All subprocesses done.')


def preprocess(img):
    scale = 512
    mask = np.zeros(img.shape)
    cv.circle(mask, (int(img.shape[1]/2), int(img.shape[0]/2)),
                int(scale/2*0.98), (1, 1, 1), -1, 8, 0)
    weighted_img = cv.addWeighted(img, 4, cv.GaussianBlur(img, (0, 0), scale/30), -4, 128)
    processed_img = weighted_img * mask + 128 * (1 - mask)

    # To reproduce the saliency map used in the paper,
    # we simulated the processing of saving the processed image in jpeg format and then reading it.
    # These codes can be removed if error or performance degradation is observed.
    processed_img = processed_img.astype(np.uint8)
    _, jpeg = cv.imencode('.jpeg', processed_img)
    processed_img = cv.imdecode(jpeg, cv.IMREAD_COLOR)

    return processed_img


if __name__ == '__main__':
    main()
