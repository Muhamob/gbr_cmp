import os
from pathlib import Path
from typing import Union, List, Optional

import torch
from sahi.slicing import get_slice_bboxes
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv

from reef.yolo.data.ops import read_labels, unnormalize, img2label_paths
from utils.augmentations import Albumentations, random_perspective, augment_hsv, letterbox
from utils.general import xywhn2xyxy, xyxy2xywhn

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes


class LoadImagesAndLabels(Dataset):
    """Train only dataset"""

    def __init__(
            self,
            path: Union[str, Path],
            slice_height: int = 360,
            slice_width: int = 640,
            overlap_threshold: float = 0.75,
            img_size: int = 640,
            batch_size: int = 16,
            augment: int = False,
            hyp: dict = None,
            stride: int = 32,
            random_state: int = None,
            prefix: str = '',
            debug: bool = True
    ):
        # Path to txt with images
        self.path = path

        # slicing
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_threshold = overlap_threshold
        self.slicer = Slicer(
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_threshold=self.overlap_threshold,
            random_state=random_state
        )

        # net input image size (max size)
        self.img_size = img_size
        self.filler = Filler(width=self.img_size, height=self.img_size)

        # augmentation
        self.augment = augment
        self.mosaic = self.augment  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.albumentations = Albumentations() if augment else None

        # Hyperparameters
        self.hyp = hyp

        # Grid parameters
        self.stride = stride  # max size of cell

        # Logging paramters
        self.prefix = prefix
        self.debug = debug

        # Sampling parameters
        self.batch_size = batch_size
        self.random_state = random_state
        self.rs = np.random.RandomState(self.random_state)

        # images, label_paths and labels
        self.img_paths, self.label_paths, self.labels = [], [], []
        self.load_images_and_labels()

    def load_images_and_labels(self):
        # read lines with images paths
        self.img_paths = list(map(lambda x: x.rstrip(), open(self.path).readlines()))
        self.label_paths = img2label_paths(self.img_paths)
        self.labels = list(map(read_labels, self.label_paths))

        assert len(self.labels) == len(self.img_paths) == len(self.label_paths), "Lengths doesn't match"

        return self

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # get image and labels
        img_path = self.img_paths[index]
        labels = self.labels[index]

        # read image
        image = cv.imread(img_path)
        image_height, image_width, image_channels = image.shape

        # crop random area and all labels in this area
        image, labels, box = self.slicer.slice_random(image, labels)
        labels = np.array(labels)

        if len(labels):  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], self.slice_width, self.slice_height, padw=0, padh=0)

        if self.augment:
            image, labels = random_perspective(
                image,
                labels,
                degrees=self.hyp['degrees'],
                translate=self.hyp['translate'],
                scale=self.hyp['scale'],
                shear=self.hyp['shear'],
                perspective=self.hyp['perspective']
            )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(
                labels[:, 1:5],
                w=image.shape[1],
                h=image.shape[0],
                clip=True,
                eps=1E-3
            )

        if self.augment:
            # Albumentations
            image, labels = self.albumentations(image, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(
                image,
                hgain=self.hyp['hsv_h'],
                sgain=self.hyp['hsv_s'],
                vgain=self.hyp['hsv_v']
            )

            # Flip up-down
            if self.rs.random() < self.hyp['flipud']:
                image = np.flipud(image)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if self.rs.random() < self.hyp['fliplr']:
                image = np.fliplr(image)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        # Pad image to stride stride
        image, labels = self.filler.transform(image, labels)
        labels = np.array(labels)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image)

        return torch.from_numpy(image), labels_out, img_path, box

    @staticmethod
    def collate_fn(batch):
        imgs, labels, paths, boxes = zip(*batch)  # transposed
        for i, label in enumerate(labels):
            label[:, 0] = i  # add target image index for build_targets()
        return torch.stack(imgs, 0), torch.cat(labels, 0), paths, boxes


class Slicer:
    def __init__(
            self,
            slice_width: int,
            slice_height: int,
            overlap_threshold: float = 1.0,
            random_state: Optional[int] = None
    ):
        self.slice_width = slice_width
        self.slice_height = slice_height

        self.overlap_threshold = overlap_threshold

        self.rs = np.random.RandomState(random_state)

    def slice_grid(
            self,
            image: np.array,
            labels: List[List]
    ) -> (np.array, List[List], List[int]):
        height, width = image.shape[:2]

        image_boxes = get_slice_bboxes(
            image_height=height,
            image_width=width,
            slice_width=self.slice_width,
            slice_height=self.slice_height,
            overlap_height_ratio=0,
            overlap_width_ratio=0
        )

        for box in image_boxes:
            transformed_image = self.transform_image(image, box)
            transformed_labels = self.transform_labels(labels, box, width, height)

            yield transformed_image, transformed_labels, box

    def slice_grid_random(
            self,
            image: np.array,
            labels: List[List]
    ) -> (np.array, List[List], List[int]):
        height, width = image.shape[:2]

        image_boxes = get_slice_bboxes(
            image_height=height,
            image_width=width,
            slice_width=self.slice_width,
            slice_height=self.slice_height,
            overlap_height_ratio=0,
            overlap_width_ratio=0
        )

        box = image_boxes[self.rs.randint(len(image_boxes))]
        transformed_image = self.transform_image(image, box)
        transformed_labels = self.transform_labels(labels, box, width, height)

        return transformed_image, transformed_labels, box

    def slice_random(
            self,
            image: np.array,
            labels: List[List]
    ) -> (np.array, List[List], List[int]):
        height, width = image.shape[:2]

        x_offset = self.rs.randint(width - self.slice_width)
        y_offset = self.rs.randint(height - self.slice_height)

        box = [x_offset, y_offset, x_offset + self.slice_width, y_offset + self.slice_height]

        transformed_image = self.transform_image(image, box)
        transformed_labels = self.transform_labels(labels, box, width, height)

        return transformed_image, transformed_labels, box

    @staticmethod
    def transform_image(
            image: np.array,
            box: list
    ) -> np.array:
        x_min, y_min, x_max, y_max = box
        return image[y_min:y_max, x_min:x_max]

    def calc_overlap(
            self,
            label: List[float]
    ):
        x_c, y_c, w, h = label
        x_overlap = max(0, min(1, x_c + w / 2) - max(0, x_c - w / 2))
        y_overlap = max(0, min(1, y_c + h / 2) - max(0, y_c - h / 2))

        return x_overlap * y_overlap / (w * h)

    def transform_labels(
            self,
            labels: List[List],
            box: List[int],
            image_width: int,
            image_height: int
    ) -> List[List]:
        x_min, y_min, x_max, y_max = box

        result_labels = []
        for c, *label in labels:
            x, y, w, h = unnormalize(label, image_width, image_height)
            # shift
            x, y = x - x_min, y - y_min
            # scale
            label = [
                c,
                x / self.slice_width,
                y / self.slice_height,
                w / self.slice_width,
                h / self.slice_height
            ]

            # calc overlap ratio
            overlap_ratio = self.calc_overlap(label[1:])
            if overlap_ratio < self.overlap_threshold: continue

            # check that center of box in image
            is_box_center_in_image = (
                (label[1] < 0 or label[1] > 1)
                or
                (label[2] < 0 or label[2] > 1)
            )
            if is_box_center_in_image: continue

            result_labels.append(label)

        return result_labels


class Filler:
    def __init__(
            self,
            width: int,
            height: int,
            fill_value: tuple = (114, 114, 114),
            auto: bool = True,
            scale_fill: bool = False,
            scale_up: bool = True,
            stride: int = 32
    ):
        # output image size
        self.width = width
        self.height = height

        # fill parameters
        self.fill_value = fill_value
        self.auto = auto
        self.scale_fill = scale_fill
        self.scale_up = scale_up
        self.stride = stride

    def fill(self, image: np.array):
        """
        image size: (height, width, 3)
        """
        image, ratio, (dw, dh) = letterbox(
            im=image,
            new_shape=(self.height, self.width),
            color=self.fill_value,
            scaleFill=self.scale_fill,
            scaleup=self.scale_up,
            stride=self.stride
        )

        return image, ratio, (dw, dh)

    def transform(
            self,
            image: np.array,
            labels: List[int]
    ):
        # original image size
        height, width = image.shape[:2]

        # fill image
        image, ratio, (dw, dh) = self.fill(image)

        # transformed image size
        t_height, t_width = image.shape[:2]

        results_labels = []
        for c, *label in labels:
            x, y, w, h = unnormalize(label, width=width, height=height)

            # shift center of box
            x *= ratio[0]
            y *= ratio[1]

            x += dw
            y += dh

            results_labels.append([
                c,
                x / t_width,
                y / t_height,
                w / t_width,
                h / t_height
            ])

        return image, results_labels
