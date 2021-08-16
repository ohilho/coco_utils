#!/usr/bin/python3

from typing import Dict, List
from copy import deepcopy


def _offset(ids: List) -> int:
    zero_base_offset = 1 if 0 in ids else 0
    return max(ids) + zero_base_offset


def merge(annot0: Dict, annot1: Dict, info: Dict) -> Dict:
    # deep copy annotations
    ann0 = deepcopy(annot0)
    ann1 = deepcopy(annot1)
    # these keys will merged with id offset
    dset_keys = ['license', 'images', 'annotations', 'categories']
    # calculate offsets
    offsets = {k: _offset([ann['id'] for ann in ann0[k]]) for k in dset_keys}

    # merge
    for key in dset_keys:
        for ann in ann1[key]:
            ann['id'] += offsets[key]
            ann0[key].append(ann)

    ann0['info'] = info
    return ann0
