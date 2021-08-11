#!/usr/bin/python3

from typing import Tuple, Dict
from pathlib import Path
import numpy as np
import cv2
from argparse import ArgumentParser
from .common import load


def mean_std(annot: Dict, annot_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """ Get the mean and standard deviation of all images for each channel.

    Args:
        annot (Dict): annotation dictionary
        annot_path (Path): Path object pointing the annotation file

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean,std) tuple. 
    """
    mean = []
    sq_mean = []
    for img_ann in annot['images']:
        # joined absolute path to the image file
        img_path = annot_path.parent.joinpath(Path(img_ann['file_name']))

        # open image file and get the mean and standard deviation
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        _mean, _std = cv2.meanStdDev(img)
        _sq_mean = _std*_std + _mean*_mean
        mean.append(_mean)
        sq_mean.append(_sq_mean)
    mean = np.mean(np.concatenate(mean, axis=1), axis=1)
    sq_mean = np.mean(np.concatenate(sq_mean, axis=1), axis=1)
    std = np.sqrt(sq_mean - mean*mean)
    return mean, std


if __name__ == '__main__':
    parser = ArgumentParser(
        'resize all images and annotations with given size')
    parser.add_argument('--annotation', type=str,
                        help='path to the annotation file')
    args = parser.parse_args()

    annot_path = Path(args.annotation).absolute().resolve()
    annot = load(annot_path)
    mean, std = mean_std(annot, annot_path)
    print("mean: {}".format(mean))
    print("std: {}".format(std))
