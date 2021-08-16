#! /usr/bin/python3
from os import PathLike
from pathlib import Path
import numpy as np
from typing import Callable, Dict, Union
import json


################################################################
# coco instance segmentation dataset structure
# {
#   "info":{
#       "contributor": str,
#       "date_created": "YYYY-MM-DD HH:MM:SS.SUBSEC",
#       "description": str,
#       "url": "https://github.com/usr/repo",
#       "version": "0.0.0",
#       "year": 2021
#   },
#   "license":[
#       {
#         "id": 1,
#         "name": "Attribution-NonCommercial-ShareAlike License",
#         "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
#       }
#   ],
#   "categories":[
#       {
#           "id": 1,
#           "name": "sack",
#           "supercategory": "sack"
#       }, ...
#   ],
#   "images":[
#       {
#           "file_name": str,
#           "height": int,
#           "id": int,
#           "width": int
#       }, ...
#   ]
#   "annotations":[
#       {
#           "area": 35418.5,
#           "bbox": [int(x),int(y),int(width),int(height)],
#           "category_id": 3,
#           "id": 0,
#           "image_id": 0,
#           "iscrowd": 0,
#           "segmentation": [
#             [
#               x0,y0,x1,y1, int, int, ...
#             ]
#           ]
#       }
#   ]
# }
################################################################

def save(annot_dict: Dict, path: PathLike) -> None:
    """save coco annotation dictionary

    Args:
        annot_dict (Dict): annotation dictionary
        path (PathLike): target path to the annotation as .json file
    """
    opath = Path(path)
    opath.parent.mkdir(parents=True, exist_ok=True)
    with opath.open('w') as f:
        json.dump(annot_dict, f)


def load(path: PathLike) -> Dict:
    """load coco annotation json file

    Args:
        path (PathLike): path to the source json file

    Returns:
        Dict: parsed json object as dictionary
    """
    ipath = Path(path).absolute().resolve()
    with ipath.open('r') as f:
        return json.load(f)


def yes_no(message: str, default='n') -> bool:
    options = {
        'y': True, 'n': False, 'yes': True, 'no': False
    }
    print(message + " [y/n], default = '{}'".format(default))
    ans = input().lower()
    if ans not in options.keys():
        print("wrong answer. Choose default: '{}'".format(default))
    return options[ans]


def make_dim_3(arr: np.ndarray) -> np.ndarray:
    dim = len(arr.shape)
    # grayscale image case. shape is (h,w) make it (h,w,c)
    if dim == 2:
        return np.expand_dims(arr, dim)
    return arr


class ImageSequenceIter:
    """Image sequence iterator. This iterator iterates the files inside the given path.
    This iterator need two callables: seed_extractor and index_extractor.
    seed_extractor extracts the seed string from the path. index_extractor extracts the index from the path. 
    Files with the same seed is considered as one sequence. index is the sequence number of the image in the sequence. 
    if num_idx is given, you will iterate over the sequences which has the length of 'num_idx'.
    """

    def __init__(self, path: PathLike, seed_extractor: Callable[[Path], str], index_extractor: Callable[[Path], int], num_idx: Union[None, int] = None) -> None:
        self.path = Path(path)
        correlated = {}
        for p in self.path.iterdir():
            idx = index_extractor(p)
            # don't count non-contributing file
            if idx < 0:
                continue

            # extract seed from path
            seed = seed_extractor(p)

            # create key in the dictionary if there is no key named as seed.
            if seed not in correlated.keys():
                correlated[seed] = {}

            # put the value
            correlated[seed][idx] = p

        self.seq_list = list(correlated.values())

        if num_idx != None:
            self.seq_list = [p for p in self.seq_list if len(
                self.seq_list) == num_idx]

    def __iter__(self):
        return iter(self.seq_list)


def is_image(path: Path) -> bool:
    # check if this is allowed image file
    extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    return path.suffix in extensions
