import os
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_process', type=int, default=8, help='number of processes')
parser.add_argument('--image-folder', type=str, help='path to image folder')
parser.add_argument('--output-folder', type=str, help='path to save saliency map')
parser.add_argument('--visualize-folder', type=str, default='', help='path to save saliency map visualizeualization')


circle = np.zeros((512, 512))
circle = cv.circle(circle, (256, 256), 240, 1, -1)


def saliency(i, src_path, output_path, visualize_path):
    image = cv.imread(src_path)
    image = preprocess(image)

    saliency = cv.saliency.StaticSaliencyFineGrained_create()
    (_, raw_saliencyMap) = saliency.computeSaliency(image)
    raw_saliencyMap *= circle

    np.save(output_path, raw_saliencyMap)

    if visualize_path:
        int_saliencyMap = (raw_saliencyMap * 255).astype("uint8")
        cv.imwrite(visualize_path, int_saliencyMap)

    if i % 500 == 0:
        print('Processed {} images'.format(i))


def main():
    args = parser.parse_args()

    i = 0
    pool = Pool(processes=args.num_process)
    print('Start processing...')
    for folder, _, imgs in os.walk(args.image_folder):
        output_folder = folder.replace(args.image_folder, args.output_folder)
        os.makedirs(output_folder, exist_ok=True)

        if args.visualize_folder:
            visualize_folder = folder.replace(args.image_folder, args.visualize_folder)
            os.makedirs(visualize_folder, exist_ok=True)
        for img in tqdm(imgs):
            i += 1
            src_path = os.path.join(folder, img)
            output_path = os.path.join(output_folder, os.path.splitext(img)[0]) + '.npy'
            visualize_path = os.path.join(visualize_folder, img) if args.visualize_folder else ''
            pool.apply_async(saliency, args=(i, src_path, output_path, visualize_path))

    print('Waiting for all subprocesses done...')
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