#! /usr/bin/python3

from math import sqrt
from typing import Dict, Tuple, Union
from pathlib import Path
from argparse import ArgumentParser
import cv2
from common import yes_no, save, load
import numpy as np


def _get_size(
    img_size: Tuple[int, int], target_size: Tuple[Union[int, None], Union[int, None]]
) -> Tuple[int, int]:
    """Calculate the size of output image

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


def relocate(
    annot: Dict,
    color_img_dir_path: Path,
    depth_img_dir_path: Path,
    target_size: Tuple[Union[int, None], Union[int, None]],
) -> Dict:
    """Resize the image both in the annotation and image file.

    Args:
        annot (Dict): coco annotation dictionary
        annot_path (Path): Path object pointing the annotation file
        target_size (Tuple): target image size tuple. (width, heigth) format

    Returns:
        Dict: image resized coco annotation dictionary
    """
    size_dict = {}
    for img_ann in annot["images"]:
        # joined absolute path to the image file
        color_img_path = Path(color_img_dir_path).joinpath(Path(img_ann["file_name"]))
        depth_img_path = Path(depth_img_dir_path).joinpath(Path(img_ann["file_name"]))

        # change the size of image in the file
        color_img = cv2.imread(str(color_img_path), cv2.IMREAD_UNCHANGED)
        depth_img = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)

        # get the size of the images from annotations
        old_w, old_h = color_img.shape[1], color_img.shape[0]
        new_w, new_h = _get_size((old_w, old_h), target_size)

        # record the size of images for iterating the annot['annotations' ]
        size_dict[img_ann["id"]] = [old_w, old_h, new_w, new_h]

        if (old_w == new_w) and (old_h == new_h):
            continue

        # pad image
        if new_w < old_w:
            new_color_img = np.zeros_like(color_img)
            new_depth_img = np.zeros_like(depth_img)
            x_lb = int((old_w - new_w) / 2)
            x_ub = x_lb + new_w
            y_lb = int((old_h - new_h) / 2)
            y_ub = y_lb + new_h
            new_color_img[y_lb:y_ub, x_lb:x_ub] = cv2.resize(
                color_img, dsize=(new_w, new_h)
            )
            new_depth_img[y_lb:y_ub, x_lb:x_ub] = cv2.resize(
                depth_img, dsize=(new_w, new_h)
            )

            # cv2.imshow("new_color_img", new_color_img)
            # cv2.imshow("new_depth_img", new_depth_img)
            # cv2.waitKey(0)
        # crop image
        elif new_w > old_w:
            new_color_img = np.zeros_like(color_img)
            new_depth_img = np.zeros_like(depth_img)
            x_lb = int((new_w - old_w) / 2)
            x_ub = x_lb + old_w
            y_lb = int((new_h - old_h) / 2)
            y_ub = y_lb + old_h
            new_color_img = cv2.resize(color_img, dsize=(new_w, new_h))[
                y_lb:y_ub, x_lb:x_ub
            ]
            new_depth_img = cv2.resize(depth_img, dsize=(new_w, new_h))[
                y_lb:y_ub, x_lb:x_ub
            ]
            # cv2.imshow("new_color_img", new_color_img)
            # cv2.imshow("new_depth_img", new_depth_img)
            # cv2.waitKey(0)

        new_depth_img = new_depth_img / sqrt(new_w / old_w * new_h / old_h)
        new_depth_img = new_depth_img.astype(np.uint16)

        cv2.imwrite(str(color_img_path), new_color_img)
        cv2.imwrite(str(depth_img_path), new_depth_img)

    for ann in annot["annotations"]:
        ow, oh, nw, nh = size_dict[ann["image_id"]]
        mult = nw / ow, nh / oh
        osz = (ow, oh)
        # pad image
        if nw < ow:
            osz = (ow, oh)
            # resize segmentation
            for seg in ann["segmentation"]:
                for i, val in enumerate(seg):
                    seg[i] = int(
                        (val - (0.5 * osz[i % 2])) * mult[i % 2] + (0.5 * osz[i % 2])
                    )

            # resize bounding box
            ann["bbox"][0] = int((ann["bbox"][0] - (0.5 * ow)) * mult[0] + (0.5 * ow))
            ann["bbox"][1] = int((ann["bbox"][1] - (0.5 * oh)) * mult[1] + (0.5 * oh))
            ann["bbox"][2] = int(ann["bbox"][2] * mult[0])
            ann["bbox"][3] = int(ann["bbox"][3] * mult[1])
            # calculate area
            ann["area"] = ann["area"] * (nw * nh) / (ow * oh)

        # crop image
        elif nw > ow:
            osz = (ow, oh)
            lb = (int((nw - ow) / 2), int((nh - oh) / 2))
            ub = (lb[0] + ow, lb[1] + oh)

            # resize segmentation
            for seg in ann["segmentation"]:
                for i, val in enumerate(seg):
                    seg[i] = min(
                        max(
                            int(
                                (val - (0.5 * osz[i % 2])) * mult[i % 2]
                                + (0.5 * osz[i % 2])
                            ),
                            0,
                        ),
                        osz[i % 2] - 1,
                    )

            # resize bounding box
            # for i, val in enumerate(ann["bbox"]):
            #     ann["bbox"][i] = int(val * mult[i % 2])

            xl = ann["bbox"][0]
            yl = ann["bbox"][1]
            xu = ann["bbox"][2] + xl
            yu = ann["bbox"][3] + yl
            xl = min(max(int((xl - (0.5 * ow)) * mult[0] + (0.5 * ow)), 0), ow - 1)
            yl = min(max(int((yl - (0.5 * oh)) * mult[1] + (0.5 * oh)), 0), oh - 1)
            xu = min(max(int((xu - (0.5 * ow)) * mult[0] + (0.5 * ow)), 0), ow - 1)
            yu = min(max(int((yu - (0.5 * oh)) * mult[1] + (0.5 * oh)), 0), oh - 1)
            ann["bbox"][0] = xl
            ann["bbox"][1] = yl
            ann["bbox"][2] = xu - xl
            ann["bbox"][3] = yu - yl
            # calculate area
            ann["area"] = ann["area"] * (nw * nh) / (ow * oh)
    return annot


if __name__ == "__main__":
    parser = ArgumentParser("resize all images and annotations with given size")
    parser.add_argument("--annotation", type=str, help="path to the annotation file")
    parser.add_argument(
        "--color_img_dir", type=str, help="path to the color image directory"
    )
    parser.add_argument(
        "--depth_img_dir", type=str, help="path to the depth image directory"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="resize to the given width preserving original aspect ratio. \
            if both the width and height are given, aspect ratio is ignored.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="resize to the given height preserving original aspect ratio. \
            if both the width and height are given, aspect ratio is ignored.",
    )
    args = parser.parse_args()

    # if not yes_no(
    #     "[NOTICE] This process can result in permanent data loss. Did you make a copy?"
    # ):
    #     exit()

    annot_path = Path(args.annotation).absolute().resolve()
    annot = load(annot_path)
    annot = relocate(
        annot, args.color_img_dir, args.depth_img_dir, (args.width, args.height)
    )
    # save
    save(annot, annot_path)
