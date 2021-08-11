#! /usr/bin/python3

from typing import Dict, Tuple, Union
from pathlib import Path
from argparse import ArgumentParser
import cv2
from .common import yes_no, save, load


def _get_size(img_size: Tuple[int, int], target_size: Tuple[Union[int, None], Union[int, None]]) -> Tuple[int, int]:
    """ Calculate the size of output image

    Args:
        img_size (Tuple): source image size. (width, heigth) format.
        target_size (Tuple): output image size. if one of width and height is None, missing one is automatically calculated preserving aspect ratio.

    Returns:
        Tuple: output image size
    """
    # img size in h, w, c order. output is in (w, h) order
    w, h = img_size[:2]
    new_w, new_h = target_size
    if (new_w is not None) and (new_h is not None):
        return (int(new_w), int(new_h))

    if (new_w is None) and (new_h is not None):
        return (int(new_h * w / h), int(new_h))

    if (new_w is not None) and (new_h is None):
        return (int(new_w), int(new_w * h / w))


def resize_image(annot: Dict, annot_path: Path, target_size: Tuple[Union[int, None], Union[int, None]]) -> Dict:
    """Resize the image both in the annotation and image file.

    Args:
        annot (Dict): coco annotation dictionary
        annot_path (Path): Path object pointing the annotation file
        target_size (Tuple): target image size tuple. (width, heigth) format

    Returns:
        Dict: image resized coco annotation dictionary
    """
    size_dict = {}
    for img_ann in annot['images']:
        # joined absolute path to the image file
        img_path = annot_path.parent.joinpath(Path(img_ann['file_name']))

        # get the size of the images from annotations
        old_w, old_h = img_ann['width'], img_ann['height']
        new_w, new_h = _get_size((old_w, old_h), target_size)

        # change size value in the annotation
        img_ann['width'], img_ann['height'] = new_w, new_h

        # record the size of images for iterating the annot['annotations' ]
        size_dict[img_ann['id']] = [old_w, old_h, new_w, new_h]

        # change the size of image in the file
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, dsize=(new_w, new_h))
        cv2.imwrite(str(img_path), img)

    for ann in annot['annotations']:
        ow, oh, nw, nh = size_dict[ann['image_id']]
        mult = nw/ow, nh/oh
        # resize segmentation
        for seg in ann['segmentation']:
            for i, val in enumerate(seg):
                seg[i] = int(val * mult[i % 2])
        # resize bounding box
        for i, val in enumerate(ann['bbox']):
            ann['bbox'][i] = int(val * mult[i % 2])
        # calculate area
        ann['area'] = ann['area'] * (nw * nh) / (ow * oh)


if __name__ == '__main__':
    parser = ArgumentParser(
        'resize all images and annotations with given size')
    parser.add_argument('--annotation', type=str,
                        help='path to the annotation file')
    parser.add_argument(
        '--width', type=int,
        help='resize to the given width preserving original aspect ratio. \
            if both the width and height are given, aspect ratio is ignored.')
    parser.add_argument(
        '--height', type=int,
        help='resize to the given height preserving original aspect ratio. \
            if both the width and height are given, aspect ratio is ignored.')
    args = parser.parse_args()

    if not yes_no("[NOTICE] This process can result in permanent data loss. Did you make a copy?"):
        exit()

    annot_path = Path(args.annotation).absolute().resolve()
    annot = load(annot_path)
    annot = resize_image(annot, annot_path, (args.width, args.height))
    # save
    save(annot, annot_path)
