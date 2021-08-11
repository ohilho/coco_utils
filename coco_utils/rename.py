#! /usr/bin/python3

from typing import Callable, Dict
from pathlib import Path
from argparse import ArgumentParser
from .common import save, load


def rename(annot: Dict, annot_path: Path, name_gen: Callable[[Path], Path], annot_only: bool = False) -> Dict:
    """ Remove all images and annotations about missing file. 

    Args:
        annot (Dict): annotation dictionary
        annot_path (Path): Path object pointing the annotation file
        name_gen (function): new name generating function using old name. name_gen(annot['images'][i]['file_name']) should be available. 

    Returns:
        Dict: annotation dictionary with new image file names
    """
    missing_ids = []
    for img_ann in annot['images']:
        # joined absolute path to the image file
        img_path = annot_path.parent.joinpath(Path(img_ann['file_name']))
        new_name = name_gen(img_path)
        img_ann['file_name'] = str(new_name.relative_to(annot_path.parent))
        if not annot_only:
            img_path.rename(new_name)
    return annot


def _test_name_gen(p: Path):
    return p.with_name("{}.png".format(p.name))


if __name__ == '__main__':
    print("this is not command line executable. look inside the code L34.")
    parser = ArgumentParser('remove annotations about lost files.')
    parser.add_argument('--annotation', type=str,
                        help='path to the annotation file')
    args = parser.parse_args()

    annot_path = Path(args.annotation).absolute().resolve()
    annot = load(annot_path)
    annot = rename(annot, annot_path, _test_name_gen)
    # save
    save(annot, annot_path)
