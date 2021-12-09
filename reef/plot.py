from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

from copy import copy


def add_bbox(
        ax: Axes,
        bounding_boxes: list
) -> Axes:
    """Draw bounding boxes on image
    """
    for bbox in bounding_boxes:
        rect = patches.Rectangle(
            (bbox['x'], bbox['y']),
            bbox['width'],
            bbox['height'],
            facecolor='none',
            edgecolor='red'
        )
        ax.add_patch(rect)

    return ax


def add_image(
        ax: Axes,
        image: Image.Image
) -> Axes:
    """Add image on axes
    """
    ax.imshow(image)
    return ax


def plot_sample(
        ax: Axes,
        image: Image.Image,
        bounding_box: list,
) -> Axes:
    """Add image and bounding boxes on images simultaneously
    """
    add_image(ax, image)
    add_bbox(ax, bounding_box)
    return ax


def stack_images(
        paths: list,
        bounding_boxes: list,
        axis: str = 'width'
) -> Tuple[Image.Image, list]:
    """Plot a stack of images and their bounding boxes in one ax.
    If axis equals to 'width' then images will be in a row.
    Otherwise in a column
    """
    # prevent changes from original dataframe
    bounding_boxes = copy(bounding_boxes)

    # open images and get their sizes
    images = [Image.open(p) for p in paths]
    widths, heights = zip(*(image.size for image in images))

    # compute sum along given axis and max of opposite dimension for new image
    total_dim_size = sum(widths) if axis == 'width' else sum(heights)
    max_dim_size = max(heights) if axis == 'width' else max(widths)
    total_shape = (total_dim_size, max_dim_size) if axis == 'width' else (max_dim_size, total_dim_size)

    # initialize new image and bounding boxes
    new_image = Image.new("RGB", total_shape)
    new_bounding_boxes = []

    offset = 0
    dim_stack = 'x' if axis == 'width' else 'y'
    dim_sizes = widths if axis == 'width' else heights

    # iterate each image and increment offset along given dimension
    for image, ds, img_bounding_boxes in zip(images, dim_sizes, bounding_boxes):
        paste_place = (offset, 0) if axis == 'width' else (0, offset)
        new_image.paste(image, paste_place)
        for bbox in img_bounding_boxes:
            bbox = copy(bbox)
            bbox[dim_stack] = offset + bbox[dim_stack]
            new_bounding_boxes.append(bbox)
        offset += ds

    # return new image and bounding boxes in form suitable for plot_sample function
    return new_image, new_bounding_boxes


def plot_stack(
        paths: list,
        bounding_boxes: list,
        axis: str = 'width',
        base_figsize: tuple = (4, 4)
) -> Tuple[Figure, Axes]:
    """Plot stacked images along given axis

    Parameters:
        paths: list[str] - list of paths to images
        bounding_boxes: list[list[dict]] - list of images bounding boxes
        axis: str - 'width' if stack in row, otherwise stack in col
        base_figsize: tuple - size of one image
    Returns:
        fig: Figure, ax: Axes
    """
    n = len(paths)
    stacked_image, stacked_bbox = stack_images(
        paths,
        bounding_boxes,
        axis
    )
    figsize = (base_figsize[0] * n, base_figsize[1]) if axis == 'width' else (base_figsize[0], 4 * base_figsize[1])
    fig, ax = plt.subplots(figsize=figsize)

    plot_sample(ax, stacked_image, stacked_bbox)
    return fig, ax
