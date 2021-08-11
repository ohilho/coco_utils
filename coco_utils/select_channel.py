#! /usr/bin/python3

from typing import Dict, List
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
import cv2
from .common import save, load


def _str_to_channel_idx(channel: str) -> List[int]:
    """Convert the channel string into list of channel index.
    for example, 'bgr' is converted into [0,1,2], 'rgba' is converted into [2,1,0,3].

    Args:
        channel (str): channel string. this is combination of 'b','g','r',and 'a'. each of character apears only one time.

    Returns:
        List: list of index of the channels
    """
    idx_map = {'B': 0, 'G': 1, 'R': 2, 'A': 3,
               'b': 0, 'g': 1, 'r': 2, 'a': 3}
    return np.array([idx_map[c] for c in channel])


def select_channel(annot: Dict, annot_path: Path, channel: str) -> None:
    """Create a new image containing only selected channels in selected order.

    Args:
        annot (Dict): coco annotation dictionary
        annot_path (Path): Path object pointing the annotation file
        channel (str): channel string. for example, bgr means channel order in blue, green, and red. b,g,r,a is allowed.

    """
    for img_ann in annot['images']:
        # joined absolute path to the image file
        img_path = annot_path.parent.joinpath(Path(img_ann['file_name']))

        # change the size of image in the file
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img = img[:, :, _str_to_channel_idx(channel)]
        cv2.imwrite(str(img_path), img)


if __name__ == '__main__':
    parser = ArgumentParser(
        'resize all images and annotations with given size')
    parser.add_argument('--annotation', type=str, required=True,
                        help='path to the annotation file')
    parser.add_argument('--channel', type=str, required=True,
                        help='channel string. B, G, R, A is allowed. e.g. BGA')
    args = parser.parse_args()

    annot_path = Path(args.annotation).absolute().resolve()
    annot = load(annot_path)
    annot = select_channel(annot, annot_path, args.channel)
    # save
    save(annot, annot_path)
