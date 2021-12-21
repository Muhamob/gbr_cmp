from typing import Optional

import albumentations as A
import numpy as np

from utils.augmentations import random_perspective, augment_hsv
from utils.general import xywhn2xyxy, xyxy2xywhn


class Augmentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(
            self,
            hyp: dict,
            slice_width: int,
            slice_height: int,
            random_state: Optional[int] = None
    ):
        self.hyp = hyp

        self.slice_width = slice_width
        self.slice_height = slice_height

        self.random_state = random_state
        self.rs = np.random.RandomState(self.random_state)

        self.transforms = [
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.0)
        ]

        self.transform_with_bbox = A.Compose(
            self.transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            )
        )
        self.transform_without_bbox = A.Compose(
            self.transforms
        )

    def _random_perspective(self, im, labels):
        # apply random perspective
        if len(labels):  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(
                labels[:, 1:],
                self.slice_width,
                self.slice_height,
                padw=0,
                padh=0
            )

        im, labels = random_perspective(
            im,
            labels,
            degrees=self.hyp['degrees'],
            translate=self.hyp['translate'],
            scale=self.hyp['scale'],
            shear=self.hyp['shear'],
            perspective=self.hyp['perspective']
        )

        if len(labels):
            labels[:, 1:5] = xyxy2xywhn(
                labels[:, 1:5],
                w=im.shape[1],
                h=im.shape[0],
                clip=True,
                eps=1E-3
            )

        return im,  labels

    def __call__(self, im, labels, p=1.0):
        # random perspective
        im, labels = self._random_perspective(im, labels)

        # apply albumentations augmentations
        if self.rs.random() < p:
            if len(labels) > 0:
                new = self.transform_with_bbox(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
                im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
            else:
                new = self.transform_without_bbox(image=im)
                im = new['image']

        # HSV color-space
        augment_hsv(
            im,
            hgain=self.hyp['hsv_h'],
            sgain=self.hyp['hsv_s'],
            vgain=self.hyp['hsv_v']
        )

        # Flip up-down
        if self.rs.random() < self.hyp['flipud']:
            im = np.flipud(im)
            if len(labels):
                labels[:, 2] = 1 - labels[:, 2]

        # Flip left-right
        if self.rs.random() < self.hyp['fliplr']:
            im = np.fliplr(im)
            if len(labels):
                labels[:, 1] = 1 - labels[:, 1]

        return im, labels
