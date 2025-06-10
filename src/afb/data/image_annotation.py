from enum import Enum

import torch


class AFBLabel(str, Enum):
    AFB = "afb"
    NON_AFB = "non_afb"
    MIMIC = "mimic"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, label: str):
        if label is None:
            return None
        label = label.lower()
        if label == "afb" or label.startswith("afb_"):
            return cls.AFB
        elif label.startswith("non-afb_"):
            return cls.NON_AFB
        elif label.startswith("afb mimic_"):
            return cls.MIMIC
        elif label.startswith("unknown_"):
            return cls.UNKNOWN
        elif label.startswith("ls_seen_"):
            return None
        elif label in ["keypoint", "roi"]:
            return None
        else:
            raise ValueError(f"Unknown label {label}")

    @classmethod
    def get_class_names(cls) -> list:
        # NOTE APPEND ONLY
        #
        # don't sort or re-order this list,
        # append only to maintain backwards compatibility
        # with models trained on order of the class names
        return [
            "background",
            cls.AFB.upper(),
            cls.NON_AFB.upper(),
            cls.MIMIC.upper(),
            cls.UNKNOWN.upper(),
        ]

    def index(self) -> int:
        return self.get_class_names().index(self.value.upper())

    @classmethod
    def from_index(cls, index: int):
        return cls.get_class_names()[index]


def filter_bboxes_by_objclass(target, target_idx):
    """
    Assumes target is a dict containing ground truth or predicted
    bboxes for a single patch. It must at least contain the key
    'labels', a tensor of length N, the number of bboxes.

    This function drops all bboxes whose label is not equal to target_idx,
    including from other keys target may contain.
    """
    inds_to_keep = torch.nonzero(target["labels"] == target_idx).flatten()
    return {k: torch.index_select(target[k], 0, inds_to_keep) for k in target.keys()}
