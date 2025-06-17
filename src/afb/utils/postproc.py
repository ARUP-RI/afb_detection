import pandas as pd
import numpy as np
import torch
from torchvision.ops import batched_nms
from tqdm import tqdm

from afb.utils.geometry import compute_effective_area

from afb.data.m48_scale import density_per_1000x
from afb.utils.gpus import choose_single_gpu_if_available

from afb.data.image_annotation import AFBLabel
from afb.data.schema.pandas import ITEM_SCHEMA
from afb.data.metadata import MetaData


def _postproc1wsi(item_id, preds, tiles_info, gpu, min_box_microns):
    resized_um_per_px = 0.2878
    min_box_px = min_box_microns / resized_um_per_px
    preds = preds[(preds.x2 - preds.x1) >= min_box_px]
    preds = preds[(preds.y2 - preds.y1) >= min_box_px]

    while True:
        try:
            preds = preds.sort_values(by="confidence", ascending=False)
            # only do nms on gpu if we have lots of boxes
            device = gpu if preds.shape[0] > 10000 else torch.device("cpu")
            boxes = torch.as_tensor(
                preds[["x1", "y1", "x2", "y2"]].values, device=device
            )
            scores = torch.as_tensor(preds.confidence.values, device=device)
            labels = torch.as_tensor(
                preds.afb_label.values, device=device, dtype=torch.int64
            )
            keep_indices = batched_nms(boxes, scores, labels, iou_threshold=0.4).to(
                device="cpu"
            )
            break
        except RuntimeError:
            # cuda nms crashes if there's too many boxes (of order ~1/2
            # million or so), but if there's that many predictions on one
            # wsi, we probably don't need to use them all! So throw out
            # the lower scoring half and try nms again
            preds = preds.head(preds.shape[0] // 2)
    preds = preds.iloc[keep_indices]
    afb_count = preds[preds.afb_label == AFBLabel.AFB.index()].shape[0]

    item_patches = len(tiles_info[item_id])
    patches_pixel_area = compute_effective_area(tiles_info[item_id])
    scan_meta = MetaData.scan_lookup(item_id=item_id)
    orig_um_per_px_x, orig_um_per_px_y = scan_meta.mpp_x, scan_meta.mpp_y
    assert np.isclose(orig_um_per_px_x, orig_um_per_px_y, rtol=1e-3)
    um_per_px = 0.5 * (orig_um_per_px_x + orig_um_per_px_y)

    mm_per_px = 1e-3 * um_per_px
    patches_area_mm = patches_pixel_area * mm_per_px**2
    specimen = MetaData.specimen_lookup(item_id=item_id)
    item_meta = {
        "item_id": item_id,
        "density": density_per_1000x(afb_count, patches_area_mm),
        "afb_count": afb_count,
        "non_afb_count": preds[preds.afb_label == AFBLabel.NON_AFB.index()].shape[0],
        "mimic_count": preds[preds.afb_label == AFBLabel.MIMIC.index()].shape[0],
        "unk_count": preds[preds.afb_label == AFBLabel.UNKNOWN.index()].shape[0],
        "ao_label": specimen.ao_pos,
        "wsi_positive": specimen.afb_positive,
        "total_patches": item_patches,
        "patch_area_mm": patches_area_mm,
    }
    return preds, item_meta


def post_proc_preds(obj_preds, tiles_info, device=None, min_box_microns=2):
    """
    Assuming obj_preds and tiles_info are as output by obj_det_predict in
    BaseLightningModule.
    Note I'm assuming min_box_size is in microns, NOT pixels!
    """
    if device is None:
        device = choose_single_gpu_if_available()
    all_obj_preds = []
    all_item_meta = []
    print(f"obj_preds: {obj_preds}")
    wrapped_iter = tqdm(obj_preds.groupby("item_id", sort=False))
    for item_id, group_df in wrapped_iter:
        # here we could do multiprocessing to parallelize over item_ids, or
        # we can use a gpu for any wsi with a lot of predictions, but doing
        # both would be very tricky. I chose single-thread w/ gpu - it won't
        # be blazing fast but doing nms on gpu should avoid the occasional
        # slide(s) that take >10 min to do nms on cpu
        wrapped_iter.set_description(f"post-processing wsi {item_id}:")
        bbox_preds, item_meta = _postproc1wsi(
            item_id,
            group_df,
            tiles_info,
            device,
            min_box_microns,
        )
        print(f"item_meta: {item_meta}")
        all_obj_preds.append(bbox_preds)
        all_item_meta.append(item_meta)

    all_obj_preds = pd.concat(all_obj_preds)
    all_item_meta = pd.DataFrame(all_item_meta, columns=ITEM_SCHEMA)
    print(f"all_item_meta: {all_item_meta}")
    return all_obj_preds, all_item_meta
