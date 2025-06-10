import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np
from PIL import Image
import torch


def imgaug_worker_init_fn(worker_id):
    ia.seed(np.random.get_state()[1][0] + worker_id)


def collate_fn(batch):
    images = [i["image"] for i in batch]
    targets = [i["target"] for i in batch]
    item_ids = [i["item_id"] for i in batch]
    extents = [i["extent"] for i in batch]
    # specimens = [i["specimen"] for i in batch]

    return {
        "image": torch.stack(images),
        "target": list(targets),
        "item_id": list(item_ids),
        "extent": list(extents),
        # "specimen": list(specimens),
    }


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential(
            [
                iaa.Rotate((-45, 45)),
                iaa.Fliplr(0.5),
                iaa.flip.Flipud(p=0.5),
                iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.0, 0.1))),
                iaa.Sometimes(0.3, iaa.MultiplyBrightness(mul=(0.95, 1.05))),
                iaa.Sometimes(0.3, iaa.Crop(percent=(0, 0.1))),
                iaa.RemoveCBAsByOutOfImageFraction(0.9),
                iaa.ClipCBAsToImagePlanes(),
            ]
        )

    def __call__(self, img, bbs, labels):
        """
        The labels have to be bundled into the BoundingBox object so that, if
        a BoundingBox gets thrown away by augmentation (e.g., it's
        out-of-bounds after a crop or rotate), the corresponding label is
        also removed.
        """
        img = np.array(img)
        bbs = ia.BoundingBoxesOnImage(
            [
                ia.BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3], label=label)
                for bb, label in zip(bbs, labels)
            ],
            shape=img.shape,
        )

        img_aug, bbs_aug = self.aug(image=img, bounding_boxes=bbs)
        labels_aug = [bb.label for bb in bbs_aug.bounding_boxes]
        bbs_aug = [(bb.x1, bb.y1, bb.x2, bb.y2) for bb in bbs_aug.bounding_boxes]

        return Image.fromarray(img_aug), bbs_aug, labels_aug
