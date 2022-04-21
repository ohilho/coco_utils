#!/usr/bin/python3

from typing import Dict
from pathlib import Path
import cv2
from argparse import ArgumentParser
from common import load
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle


class DepthInfoInspector:
    def __init__(self) -> None:
        self.coco = None
        self.obj_mean_depths = []

    def load_annotation(self, annot_path: str):
        self.coco = COCO(annot_path)
        self.annot = load(annot_path)

    def inspect(self, depth_img_dir: str):

        # get ids
        catIds = self.coco.getCatIds()
        imgIds = self.coco.getImgIds()
        annIds = self.coco.getAnnIds()

        imgs = self.coco.loadImgs(imgIds)
        anns = self.coco.loadAnns(annIds)

        obj_mean_depths = []
        for img in imgs:
            anns_in_img = self.coco.imgToAnns[img["id"]]
            img_mat = cv2.imread(
                str(Path(depth_img_dir).joinpath(img["file_name"])),
                cv2.IMREAD_UNCHANGED,
            )
            for ann in anns_in_img:
                mask = self.coco.annToMask(ann)
                inst_only = img_mat * mask
                nz_count = np.count_nonzero(inst_only)
                nz_sum = np.sum(inst_only)
                if nz_count == 0:
                    continue
                mean = nz_sum / nz_count
                obj_mean_depths.append(mean)

        self.obj_mean_depths = np.array(obj_mean_depths)
        mean = np.mean(self.obj_mean_depths)
        median = np.median(self.obj_mean_depths)
        std = np.std(self.obj_mean_depths)
        print("mean: {}, median: {}, std: {}".format(mean, median, std))

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.obj_mean_depths, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.obj_mean_depths = pickle.load(f)

    def hist(self, title):
        plt.hist(self.obj_mean_depths, bins=range(0, 5000, 50))
        plt.show()

    def get_kde(self, x):
        estimator = stats.gaussian_kde(self.obj_mean_depths)
        return estimator(x)

    def kde(self, title):
        x = np.array(range(0, 5000, 10))
        plt.plot(x, self.get_kde(x))
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser("resize all images and annotations with given size")
    parser.add_argument("--annotation", type=str, help="path to the annotation file")
    parser.add_argument("--depth_img_dir", type=str, help="path to the depth_img_dir")
    args = parser.parse_args()

    dset_dir = "/home/ohilho/Documents/unloader_rgbd_20210930/unloader_rgbd"

    all_data = [
        {
            "title": "training dataset",
            "annot": "/home/ohilho/Documents/unloader_rgbd/unloader_rgbd_stc/info/train.json",
            "img_dir": "/home/ohilho/Documents/unloader_rgbd/unloader_rgbd_stc/depth",
        }
    ]
    for i in range(5):
        title = "lab test dataset {}".format(i + 1)
        img_dir = "/home/ohilho/Documents/unloader_rgbd/unloader_rgbd_lab/test_{}/depth".format(
            i
        )
        annot = "/home/ohilho/Documents/unloader_rgbd/unloader_rgbd_lab/test_{}/info/test.json".format(
            i
        )
        all_data.append(
            {
                "title": title,
                "annot": annot,
                "img_dir": img_dir,
            }
        )

    for i in range(7):
        title = "aug test dataset {}".format(i + 1)
        img_dir = "/home/ohilho/Documents/unloader_rgbd/unloader_rgbd_augmented/test_{}/depth".format(
            i
        )
        annot = "/home/ohilho/Documents/unloader_rgbd/unloader_rgbd_augmented/test_{}/info/test.json".format(
            i
        )
        all_data.append(
            {
                "title": title,
                "annot": annot,
                "img_dir": img_dir,
            }
        )
    x = np.array(range(0, 5000, 10))
    for d in all_data:
        di = DepthInfoInspector()
        di.load_annotation(d["annot"])
        di.inspect(d["img_dir"])
        di.save(d["title"] + "_data.pkl")
        d["kde"] = di.get_kde(x)

    all_kde = np.stack([d["kde"] for d in all_data], -1)
    plt.plot(x, all_kde)
    plt.title(
        "Labeled Instance Depth Distributions for Test Datasets and Training Dataset"
    )
    plt.legend([d["title"] for d in all_data])
    plt.xlabel("mean depth for each labeled instance [ mm ]")
    plt.ylabel("density")
    plt.show()
    exit(0)
