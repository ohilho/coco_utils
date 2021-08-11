#!/usr/bin/python3

from typing import Dict
from pathlib import Path
import numpy as np
import cv2
from argparse import ArgumentParser
from .common import load


def _str2type(type_str: str) -> np.dtype:
    """Convert string into numpy type. for example, 'int8' into np.int8, 'float32' into np.float32, etc.

    Args:
        type_str (str): type name. 'int8','int16','int32','int64','uint8','uint16','uint32','uint64','float32', and 'float64' is allowed.

    Returns:
        np.dtype: corresponding numpy data type.
    """
    str2type_dict = {'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64,
                     'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
                     'float32': np.float32, 'float64': np.float64}
    return str2type_dict[type_str]


def _scale(src_arr: np.ndarray, dst_type: np.dtype) -> np.ndarray:
    """Scale the value in the array considering the type max value so that each pixel's (value/max_value) can be conserved.
    For example, when the src_array.dtype is uint8, and the dst_type is uint16, each pixel value in the image is 255 times scaled up.

    Args:
        src_arr (np.ndarray): The numpy array which is going to be scaled. 
        dst_type (np.dtype): output type

    Returns:
        np.ndarray: scaled numpy array
    """
    # I'll not use np.iinfo.max. 64bit type max value +1 may overflow.
    type_log_scale = {np.int8: 8, np.int16: 16, np.int32: 32, np.int64: 64,
                      np.uint8: 8, np.uint16: 16, np.uint32: 32, np.uint64: 64,
                      np.float32: 0, np.float64: 0,
                      np.dtype('int8'): 8, np.dtype('int16'): 16, np.dtype('int32'): 32, np.dtype('int64'): 64,
                      np.dtype('uint8'): 8, np.dtype('uint16'): 16, np.dtype('uint32'): 32, np.dtype('uint64'): 64,
                      np.dtype('float32'): 0, np.dtype('float64'): 0
                      }
    src_log = type_log_scale[src_arr.dtype]
    dst_log = type_log_scale[dst_type]
    # instead of dividing, I use subtraction with log scale.
    log_scale = dst_log - src_log
    if log_scale == 0:
        return src_arr
    log_abs = abs(log_scale)
    scale_abs = pow(2, log_abs)
    if log_scale < 0:
        return src_arr/scale_abs
    else:
        return src_arr*scale_abs


def type_cast(annot: Dict, annot_path: Path, output_type: str) -> None:
    """Covert the data type of image conserving the (value/max_value) ratio for each pixel.
        For example, 8uc1 ->16uc1, 32c2 -> float64, ...

    Args:
        annot (Dict): annotation dictionary
        annot_path (Path): Path object pointing the annotation file
        output_type (str): output image pixel data type. 'int8','int16','int32','int64','uint8','uint16','uint32','uint64','float32', and 'float64' is allowed.

    """
    for img_ann in annot['images']:
        # joined absolute path to the image file
        img_path = annot_path.parent.joinpath(Path(img_ann['file_name']))

        # open image file and get the mean and standard deviation
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        otype = _str2type(output_type)
        img = _scale(img, otype)
        img = img.astype(otype)
        cv2.imwrite(str(img_path), img)


if __name__ == '__main__':
    parser = ArgumentParser(
        'resize all images and annotations with given size')
    parser.add_argument('--annotation', type=str,
                        help='path to the annotation file')
    parser.add_argument('--output_type', type=str,
                        help='one of the [int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64]')
    args = parser.parse_args()

    annot_path = Path(args.annotation).absolute().resolve()
    annot = load(annot_path)
    type_cast(annot, annot_path, args.output_type)
