def coco2yolo(image_height, image_width, bboxes):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """

    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    # normolizinig
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / image_height

    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]] / 2

    return bboxes


def yolo2coco(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]

    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    # denormalizing
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    # converstion (xmid, ymid) => (xmin, ymin)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2

    return bboxes
