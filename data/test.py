#! /usr/bin/env python3
from os import PathLike
from typing import Dict
import pandas as pd
import json
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import numpy as np
import cv2
from math import sqrt
import random


def save(annot_dict: Dict, path: PathLike) -> None:
    """save coco annotation dictionary

    Args:
        annot_dict (Dict): annotation dictionary
        path (PathLike): target path to the annotation as .json file
    """
    opath = Path(path)
    opath.parent.mkdir(parents=True, exist_ok=True)
    with opath.open("w") as f:
        json.dump(annot_dict, f)


def load(path: PathLike) -> Dict:
    """load coco annotation json file

    Args:
        path (PathLike): path to the source json file

    Returns:
        Dict: parsed json object as dictionary
    """
    ipath = Path(path).absolute().resolve()
    with ipath.open("r") as f:
        return json.load(f)


def dict_to_pandas(raw):
    cats = pd.DataFrame(raw["categories"]).loc[:, ["id", "name", "supercategory"]]
    imgs = pd.DataFrame(raw["images"]).loc[:, ["id", "width", "height", "file_name"]]
    annots = pd.DataFrame(raw["annotations"]).loc[
        :, ["id", "image_id", "category_id", "segmentation", "area", "bbox", "iscrowd"]
    ]
    return cats, imgs, annots


def pandas_to_dict(cats, imgs, annots):
    return {
        "categories": list(cats.T.to_dict().values()),
        "images": list(imgs.T.to_dict().values()),
        "annotations": list(annots.T.to_dict().values()),
    }


def instance_statistics(cats: pd.DataFrame, annots: pd.DataFrame):
    annots["category"] = annots["category_id"].apply(lambda x: cats.loc[x - 1, "name"])
    print(annots["category"].value_counts())
    annots.drop(columns=["category"])


def image_statistics(imgs: pd.DataFrame):
    print(
        imgs[["height", "width"]]
        .apply(lambda x: f"{x[1]}x{x[0]}", axis=1)
        .value_counts()
    )


def statistics(cats, imgs, annots):
    image_statistics(imgs)
    instance_statistics(cats, annots)


def show(ann_path, img_dir: PathLike):
    coco = COCO(ann_path)
    img_dir = Path(img_dir)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    imgs = coco.loadImgs(imgIds)

    for img in imgs:
        print(img)
        I = Image.open(img_dir / img["file_name"])
        plt.axis("off")
        plt.title(img["file_name"])
        plt.imshow(I)
        annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns, draw_bbox=True)
        plt.show()


def split(cats, images, annots, num_images):
    imgs0 = images.loc[: (num_images - 1), :]
    annots0 = annots[annots["image_id"].isin(imgs0.loc[:, "id"])]
    imgs1 = images.loc[num_images:, :]
    annots1 = annots[annots["image_id"].isin(imgs1.loc[:, "id"])]
    imgs1.loc[:, "id"] = imgs1.loc[:, "id"] - num_images
    annots1.loc[:, "image_id"] = annots1.loc[:, "image_id"] - num_images
    return (cats, imgs0, annots0), (cats, imgs1, annots1)


def sample(cats, images: pd.DataFrame, annots: pd.DataFrame, num_images):
    imgs0 = images.sample(n=num_images).sort_values(["id"], ascending=True)
    annots0 = annots[annots["image_id"].isin(imgs0.loc[:, "id"])]
    g = imgs0.groupby("id")
    id_map = {idx: i + 1 for i, idx in enumerate(g.grouper.levels[0])}
    imgs0 = imgs0.replace({"id": id_map})
    annots0 = annots0.replace({"image_id": id_map})
    annots0["id"] = pd.factorize(annots0["id"])[0] + 1
    return cats, imgs0, annots0


def merge(cats0, images0, annots0, cats1, images1, annots1):
    pass


def random_sample():
    info_path = "/home/ohilho/Downloads/unloader_rgbd_20210930/unloader_rgbd/info/test_1680.json"
    img_dir = "/home/ohilho/Downloads/unloader_rgbd_20210930/unloader_rgbd/test_images/color_test_1680"
    raw = load(info_path)
    cats, imgs, annots = dict_to_pandas(raw)
    print("======= Source Data ========")
    # instance_statistics(cats, annots)
    # image_statistics(imgs)
    show(info_path, img_dir)
    exit(0)
    for i in range(1, 8):
        num_data = i * 100
        out_path = "/home/ohilho/Documents/train_rsample{}.json".format(num_data)
        ds = sample(cats, imgs, annots, num_data)
        d = pandas_to_dict(*ds)
        save(d, out_path)
        print("======= Random Sampled Data : {} ========".format(num_data))
        print("destination : {}".format(out_path))
        statistics(*ds)
    exit(0)


def crop(width, height):
    info_path = (
        "/home/ohilho/Downloads/unloader_rgbd_20210930/unloader_rgbd/info/test.json"
    )
    color_img_dir = "/home/ohilho/Downloads/unloader_rgbd_20210930/unloader_rgbd/color"
    depth_img_dir = "/home/ohilho/Downloads/unloader_rgbd_20210930/unloader_rgbd/depth"
    raw = load(info_path)
    cats, imgs, annots = dict_to_pandas(raw)
    print("======= Source Data ========")
    # show(info_path, color_img_dir)

    for idx, row in imgs.iterrows():
        # read image
        img = cv2.imread(
            str(Path(color_img_dir) / row["file_name"]), cv2.IMREAD_UNCHANGED
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # make boundary
        x_lb = int((row["width"] - width) * 0.5)
        x_ub = x_lb + width
        y_lb = int((row["height"] - height) * 0.5)
        y_ub = y_lb + height

        plt.figure("original")
        plt.axis("off")
        plt.title(row["file_name"])
        plt.imshow(img)

        plt.figure("crop")
        plt.axis("off")
        plt.title(row["file_name"])
        plt.imshow(img[y_lb:y_ub, x_lb:x_ub])

        plt.show()
        for b in annots.loc[annots["image_id"] == row["id"], "bbox"]:
            print(b)

    # for i in range(1, 8):
    #     num_data = i * 100
    #     out_path = "/home/ohilho/Documents/train_rsample{}.json".format(num_data)
    #     ds = sample(cats, imgs, annots, num_data)
    #     d = pandas_to_dict(*ds)
    #     save(d, out_path)
    #     print("======= Random Sampled Data : {} ========".format(num_data))
    #     print("destination : {}".format(out_path))
    #     statistics(*ds)


if __name__ == "__main__":
    # crop(1080, 720)
    # random_sample()
    color_img_dir = "/home/ohilho/repos/dataset/unloader_rgbd/unloader_rgbd_stc/color"
    annot = "/home/ohilho/repos/dataset/unloader_rgbd/unloader_rgbd_stc/info/train.json"
    show(annot, color_img_dir)
