#!/usr/bin/python3

from typing import Dict
from pathlib import Path
import cv2
from argparse import ArgumentParser
from .common import load


def inspect(annot: Dict, annot_path: Path):
    """inspect the shape and dtype of the image

    Args:
        annot (Dict): annotation dictionary
        annot_path (Path): Path object pointing the annotation file

    """
    for img_ann in annot['images']:
        # joined absolute path to the image file
        img_path = annot_path.parent.joinpath(Path(img_ann['file_name']))

        # open image file and get the mean and standard deviation
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        print("shape:{} , type: {}, file: {}".format(
            img.shape, img.dtype, img_path.name))


if __name__ == '__main__':
    parser = ArgumentParser(
        'resize all images and annotations with given size')
    parser.add_argument('--annotation', type=str,
                        help='path to the annotation file')

    args = parser.parse_args()

    annot_path = Path(args.annotation).absolute().resolve()
    annot = load(annot_path)
    inspect(annot, annot_path)
