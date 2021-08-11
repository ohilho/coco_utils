#! /usr/bin/python3

from typing import Dict
from pathlib import Path
from argparse import ArgumentParser
from .common import save, load


def refine(annot: Dict, annot_path: Path) -> Dict:
    """ Remove all images and annotations about missing file.

    Args:
        annot (Dict): annotation dictionary
        annot_path (Path): Path object pointing the annotation file

    Returns:
        Dict: refined annotation dictionary
    """
    missing_ids = []
    for img_ann in annot['images']:
        # joined absolute path to the image file
        img_path = annot_path.parent.joinpath(Path(img_ann['file_name']))
        # nothing to do about this
        if img_path.exists():
            continue
        # these annotations refers missing files
        missing_ids.append(img_ann['id'])

    # refine annotations
    annot['images'] = [img_ann for img_ann in annot['images']
                       if img_ann['id'] not in missing_ids]
    annot['annotations'] = [ann for ann in annot['annotations']
                            if ann['image_id'] not in missing_ids]
    return annot


if __name__ == '__main__':
    parser = ArgumentParser('remove annotations about lost files.')
    parser.add_argument('--annotation', type=str,
                        help='path to the annotation file')
    args = parser.parse_args()

    annot_path = Path(args.annotation).absolute().resolve()
    annot = load(annot_path)
    annot = refine(annot, annot_path)
    # save
    save(annot, annot_path)
