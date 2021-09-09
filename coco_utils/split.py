#!/usr/bin/python3

from typing import Dict
from pathlib import Path
import cv2
from argparse import ArgumentParser
from .common import load, save


def split(annot: Dict):
    """inspect the shape and dtype of the image

    Args:
        annot (Dict): annotation dictionary
        annot_path (Path): Path object pointing the annotation file

    """

    # copy all except annotations

    out = {}
    img_ids = {}
    keys = ["train", "val", "test"]
    ratio = [0.7, 0.2, 0.1]
    for k in keys:
        out[k] = {
            k: v for k, v in annot.items() if (k != "annotations") and (k != "images")
        }

    # calculage number of annotations (train : validatiaon : test  = 7:2:1)
    num_annot = len(annot["images"])
    num_data = [int(r * num_annot) for r in ratio]
    num_data[-1] = num_annot - sum(num_data[:-1])
    # make num_data to a range
    num_data.insert(0, 0)
    ranges = [sum(num_data[: i + 1]) for i in range(len(num_data))]

    for i, k in enumerate(keys):
        # collect images for each data
        out[k]["images"] = [ann for ann in annot["images"][ranges[i] : ranges[i + 1]]]

        # generate image id map
        img_ids[k] = [img["id"] for img in out[k]["images"]]

        # collect annotations for each data
        out[k]["annotations"] = [
            ann for ann in annot["annotations"] if ann["image_id"] in img_ids[k]
        ]

    return tuple(out.values())


if __name__ == "__main__":
    parser = ArgumentParser("split annotation into train, val, test")
    parser.add_argument(
        "-a", "--annotation", type=str, help="path to the annotation file"
    )

    args = parser.parse_args()

    annot_path = Path(args.annotation).absolute().resolve()
    annot = load(annot_path)
    train, val, test = split(annot)

    save(train, annot_path.with_name("train.json"))
    save(val, annot_path.with_name("val.json"))
    save(test, annot_path.with_name("test.json"))
