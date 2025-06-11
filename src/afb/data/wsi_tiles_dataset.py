import logging
import time
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from afb.utils.geometry import PixelBox
from afb.utils.bbox import XYXYBox

logger = logging.getLogger(__name__)


class WSITilesDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transforms=None,
    ):
        """
        Args:
            root_dir: path to the directory containing the dataset.
                Should have subdirs 'images' and 'labels' as well as a
                'data_conf.yaml' as generated in build_data.py
            transforms: (optional) a function/transform to apply on the image and target.
        """
        self.transforms = transforms

        # This could get slow for very large datasets but for ~1/2 million
        # tiles it's only a couple sec, prob ok for afb. Debug logging in
        # case it becomes a bigger problem
        logger.debug(f"Globbing for images in {root_dir}...")
        start_time = time.time()
        self.img_names = sorted(list((root_dir / "images").rglob("*.png")))
        logger.debug(f"...took {time.time() - start_time} seconds.")
        logger.debug(f"Globbing for labels in {root_dir}...")
        start_time = time.time()
        self.label_names = sorted(list((root_dir / "labels").rglob("*.txt")))
        logger.debug(f"...took {time.time() - start_time} seconds.")

        assert (
            len(self.img_names) == len(self.label_names)
        ), f"Number of images and labels should be the same. Saw {len(self.img_names)} images and {len(self.label_names)} labels."
        logger.debug("Checking for image/label filename mismatches...")
        start_time = time.time()
        for im, label in zip(self.img_names, self.label_names):
            assert (
                im.stem == label.stem
            ), f"Image/label filename mismatch: {im} vs {label}"
        logger.debug(f"...took {time.time() - start_time} seconds.")

        # with open(root_dir / 'data_conf.yaml') as fh:
        #     data_conf = yaml.safe_load(fh)
        # self.patch_size = data_conf['patch_properties']['patch_size']
        # lab_ids = [single['lab_id'] for single in data_conf['lab_ids']]
        # spec_lookup = {lab_id:Specimen(LabID(lab_id)) for lab_id in lab_ids}

        # useful to have extent & item_id & specimen for each patch available
        # w/o having to load the image via __getitem__
        self.extents = []
        self.item_ids = []
        # self.specimens = []
        logger.debug("Precomputing extents & item_ids...")
        start_time = time.time()
        for img_name in self.img_names:
            lab_id, item_id, left, top, width, height = img_name.stem.split("_")
            left, top, width, height = int(left), int(top), int(width), int(height)
            self.extents.append(
                PixelBox(left=left, top=top, width=width, height=height)
            )
            # item_id = ImageID.instantiate_type(item_id)
            self.item_ids.append(item_id)
            # TODO: add assertions that specimen has the needed metadata fields?
            # Or just pydantify Specimen to guarantee this?
            # assert 'clsi_m48' in metadata, f"Metadata for train/val should always have clsi_m48, item_id: {item_id}, metadata: {metadata}"
            # self.specimens.append(spec_lookup[item_id.get_lab_id()])
        logger.debug(
            f"Precomputing extents & item_ids took {time.time() - start_time} seconds."
        )
        logger.debug("WSITilesDataset initialized")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Load images
        img_path = self.img_names[idx]
        img = Image.open(img_path).convert("RGB")

        # Load labels
        label_path = self.label_names[idx]
        with open(label_path, "r") as f:
            annotations = f.readlines()

        # useful for debugging, possibly unnecessary otherwise?
        # although it only costs ~few hundred ns & could save us from
        # some subtle & nasty bugs
        assert Path(img_path).stem == Path(label_path).stem

        boxes, labels = [], []
        for anno in annotations:
            anno = anno.strip().split()
            label = anno[0]  # leave as str b/c imgaug expects str, convert after
            cx, cy, w, h = [float(x) for x in anno[1:]]
            box = XYXYBox.from_normalized_center_box(
                center_x=cx, center_y=cy, width=w, height=h, img_size=img.size
            )
            boxes.append(box.to_list())
            labels.append(label)

        if self.transforms:
            img, boxes, labels = self.transforms(img, boxes, labels)

        labels = torch.as_tensor([int(label) for label in labels], dtype=torch.int64)
        if not boxes:  # If no boxes, return the right size tensor
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        assert len(labels) == len(boxes)

        img = pil_to_tensor(img).to(dtype=torch.float32)
        img /= 255.0

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return {
            "image": img,
            "target": target,
            "item_id": self.item_ids[idx],
            "extent": self.extents[idx],
            # 'specimen': self.specimens[idx],
        }
