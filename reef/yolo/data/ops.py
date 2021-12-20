import os
from pathlib import Path
from typing import Union, List

import numpy as np


def read_labels(path: Union[Path, str]) -> List[List[float]]:
    """Return all labels given path

    Return: list(list(number))
        [
            [cls, x, y, w, h],
            ...
        ]
    """
    result = []

    if not Path(path).exists():
        return result

    for labels in open(path, "r").readlines():
        # remove trailing \n and split by space
        labels = labels.rstrip().split()
        # filter empty spaces and cast to integers
        labels = list(map(float, filter(lambda x: x != '', labels)))
        # cast cls to integer
        labels[0] = int(labels[0])

        result.append(labels)

    return result


def unnormalize(
        bbox: Union[List, np.array],
        width: int,
        height: int
):
    r = np.array([width, height, width, height])
    return (np.array(bbox) * r).round(0).astype(int)


def img2label_paths(img_paths: list) -> list:
    """Define label paths as a function of image paths

    Replace /images/ to /labels/
    """
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
