#! /usr/bin/python3

import os
from typing import Callable
from pathlib import Path
from argparse import ArgumentParser
import cv2
import numpy as np
from .common import load, ImageSequenceIter, make_dim_3, is_image


def idx_ext(img_path: Path) -> int:
    idx = -1
    if not is_image(img_path):
        return idx

    # check for patterns
    patterns = ["rgb_image_rect_color", "depth_to_rgb_image_rect"]
    for i, p in enumerate(patterns):
        if p in img_path.name:
            idx = i
            break
    return idx


def seed_ext(img_path: Path) -> str:
    patterns = ["rgb_image_rect_color", "depth_to_rgb_image_rect"]
    seed = img_path.name
    for i, p in enumerate(patterns):
        seed = seed.replace(p, '')
    return seed


def concatenate(root_path: Path, seed_extractor: Callable[[Path], str], idx_extractor: Callable[[Path], int], remove_old: bool = True) -> None:
    """Concatanate images along the channel axis. All images should be inside the same directory

    Args:
        root_path (Path): Path to the directory containing all images.
        seed_extractor (Callable[[Path], str]): seed string extracting function using the path. images with the same seed will be concatanated.
        idx_extractor (Callable[[Path], int]): index extracting function using the path. images are concatanated in this index order. 
        remove_old (bool, optional): if this is True, remove all used images. image name with the index 0 will be the result of concatanate. Defaults to True.
    """
    im_iter = ImageSequenceIter(root_path, seed_extractor, idx_extractor)
    for imgs in im_iter:
        img_path = [root_path.parent.joinpath(
            Path(imgs[i])) for i in range(len(imgs))]
        arrs = [make_dim_3(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
                for path in img_path]
        new_img = np.concatenate(arrs, axis=2)
        cv2.imwrite(str(img_path[0]), new_img)
        if remove_old:
            for path in img_path[1:]:
                os.remove(str(path))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--annotation', type=str,
                        help='path to the annotation file')
    args = parser.parse_args()

    annot_path = Path(args.annotation).absolute().resolve()
    annot = load(annot_path)
    concatenate(annot_path.parent)
