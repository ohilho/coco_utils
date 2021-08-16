#! /usr/bin/python3
from abc import abstractproperty
from argparse import ArgumentParser
from coco_utils.common import yes_no, is_image
from pathlib import Path
import coco_utils as cu


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


def name_gen(img_path: Path) -> Path:
    new_name = img_path.name.replace(
        "rgb_image_rect_color", "bgrd16uc4_rect_d_to_rgb")
    return img_path.with_name(new_name)


if __name__ == '__main__':
    parser = ArgumentParser('remove annotations about lost files.')
    parser.add_argument('--annotation', type=str,
                        help='path to the annotation file')
    parser.add_argument('-y', '--yes', action='store_true')
    args = parser.parse_args()

    if not args.yes:
        if not yes_no("This procedure may lose your data permanently. Did you make a copy?"):
            exit(-1)

    annot_path = Path(args.annotation).absolute().resolve()

    annot = cu.load(annot_path)
    annot = cu.refine(annot, annot_path)
    cu.select_channel(annot, annot_path, "bgr")
    cu.type_cast(annot, annot_path, "uint16")
    cu.concatenate(annot_path.parent, seed_ext,
                   idx_ext, remove_old=True, num_idx=2)
    annot = cu.rename(annot, annot_path, name_gen)

    # show result
    cu.inspect(annot, annot_path)
    mean, std = cu.mean_std(annot, annot_path)
    print("mean : {} , std: {}".format(mean, std))

    # save
    cu.save(annot, annot_path)
