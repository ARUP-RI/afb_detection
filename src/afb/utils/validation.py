import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou
import seaborn as sns

from afb.utils.geometry import PixelBox

from afb.utils.bbox import Vec2D, XYXYBox

from afb.data.m48_scale import M48Scale, density_per_1000x
from afb.data.image_annotation import AFBLabel, filter_bboxes_by_objclass
from afb.data.metadata import MetaData

from afb.config import BOX_IOU_THRESH


def boxes_tp_fp_fn(targ_boxes, pred_boxes):
    """
    Given gt and predicted bboxes (torch tensors of size N x 4 and
    M x 4, where N is the number of gt boxes and M is the # of pred boxes),
    compute which pred boxes correctly detected a gt box (true pos), which
    pred boxes missed all gt boxes (false pos), and which gt boxes were
    missed (false neg), and return these three categories as
    tensors of bboxes along w/ the indices in targ_boxes and pred_boxes
    at which they occur.
    Note the notion of true positive here matches the soft, not hard,
    notion in box_hits_misses above.
    Also assumes all targ_boxes & pred_boxes have SAME CLASS LABEL,
    and relies on caller to do this filtering!
    """
    # convenient fall-thru defaults: assume all fp & fn, no true pos,
    # although note fp and fn tensors may be empty. This handles all
    # 3 cases where either targ_boxes and/or pred_boxes contain zero boxes
    fn_boxes = targ_boxes
    fp_boxes = pred_boxes
    tp_boxes = torch.tensor([]).reshape((0, 4))
    fn_inds = torch.arange(len(fn_boxes))
    fp_inds = torch.arange(len(fp_boxes))
    tp_inds = torch.arange(len(tp_boxes))
    # the only case where we might have true positives
    if len(targ_boxes) > 0 and len(pred_boxes) > 0:
        detections = box_iou(targ_boxes, pred_boxes)
        detections = detections > BOX_IOU_THRESH

        gt_misses = ~(detections.max(axis=1).values)
        fn_inds = torch.nonzero(gt_misses).flatten()
        fn_boxes = torch.index_select(targ_boxes, 0, fn_inds)

        pred_hits = detections.max(axis=0).values
        tp_inds = torch.nonzero(pred_hits).flatten()
        fp_inds = torch.nonzero(~pred_hits).flatten()
        tp_boxes = torch.index_select(pred_boxes, 0, tp_inds)
        fp_boxes = torch.index_select(pred_boxes, 0, fp_inds)
    return tp_inds, tp_boxes, fp_inds, fp_boxes, fn_inds, fn_boxes


def density_vs_threshold(bbox_predictions, item_predictions):
    density_threshold_data = []
    thresholds = np.linspace(0, 1, 101)
    # item_predictions = coerce_pandas_item_ids(item_predictions, coerce_type=ImageID)

    # sort fails on class ImageID b/c __lt__ op isn't implemented or meaningful
    for item_id, df in bbox_predictions.groupby("item_id", sort=False):
        # item_id = ImageID.instantiate_type(item_id)
        item_preds = item_predictions[item_predictions.item_id == item_id]
        assert item_preds.shape[0] == 1
        item_preds = item_preds.iloc[0]
        patch_area_mm = item_preds.patch_area_mm
        specimen = MetaData.specimen_lookup(item_id=item_id)
        afb_positive = specimen.afb_positive
        if afb_positive:
            # values in specimen.ao_pos aren't reliable, use clsi_m48 instead,
            # & use str as catch-all for bool or None for easier plotting
            ao_call = (
                str(specimen.clsi_m48 > 0) if specimen.clsi_m48 is not None else "None"
            )
        else:
            # if specimen is negative, I don't care to distinguish ao neg & ao none
            ao_call = "False"

        df = df[df.afb_label == AFBLabel.AFB.index()]
        for t in thresholds:
            detection_count = df[df.confidence > t].shape[0]
            density = density_per_1000x(detection_count, patch_area_mm)
            categorization = M48Scale.logarithmic_from_density(density)

            density_threshold_data.append(
                {
                    "item_id": item_id,
                    "gt_m48": str(specimen.clsi_m48),
                    "ao_pos": ao_call,
                    "afb_pos": afb_positive,
                    "threshold": t,
                    "density": density,
                    "pred_m48": categorization.value,
                }
            )

    return pd.DataFrame(density_threshold_data)


def draw_categorization_vs_threshold(density_threshold_data, output_dir, agg=True):
    plt.figure()
    # add a bokeh option w/ hovertool for easier EDA of marginal cases?

    if agg:
        sns.lineplot(
            x="threshold", y="density", hue="gt_m48", data=density_threshold_data
        )
    else:
        # need str ids b/c (I suspect?) lineplot is doing a sorted groupby
        # density_threshold_data = coerce_pandas_item_ids(
        #     density_threshold_data,
        #     coerce_type=str
        # )
        sns.lineplot(
            x="threshold",
            y="density",
            hue="afb_pos",
            hue_order=[False, True],
            style="ao_pos",
            style_order=["True", "None", "False"],
            data=density_threshold_data,
            units="item_id",
            estimator=None,
            lw=0.5,
        )
    plt.yscale("log")
    category_thresholds = [0, 0.01, 0.1, 1, 10]
    category_text = ["+/-", "1+", "2+", "3+", "4+"]
    for thresh, thresh_text in zip(category_thresholds, category_text):
        plt.axhline(y=thresh, color=(0, 0, 0, 0.2), linestyle="--", linewidth=1)
        # plt.text(1, thresh * 1.1, thresh_text, color='gray', fontsize='x-small')
    plt.yticks(ticks=category_thresholds, labels=category_text)
    legend = plt.legend()

    label_values = {
        "4": "4+",
        "3": "3+",
        "2": "2+",
        "1": "1+",
        "0": "+/-",
        "-1": "No AFB",
        "False": "Negative",
        "True": "Positive",
    }

    # print('-', legend.texts)
    plt.ylim(0, 100)

    for t in legend.texts:
        t.set_text(label_values.get(t.get_text(), t.get_text()))

    plt.title("Estimated density and categorization vs. confidence threshold")
    if output_dir is not None:
        plt.savefig(output_dir / "categorization_vs_threshold.png")


def densities_at_threshold(density_threshold_data, confidence_thresh=0):
    """
    Takes output of density_vs_threshold and filters down to one row for each item
    at the confidence score threshold closest to provided confidence_thresh.
    """
    # make local copy to avoid altering orig df
    dens_thresh_data = density_threshold_data.copy()
    n_items = len(dens_thresh_data.item_id.unique())
    # first find closest threshold in dens_thresh_data to given confidence_thresh
    confs = np.asarray(dens_thresh_data["threshold"].unique())
    diffs = np.abs(confs - confidence_thresh)
    idx = diffs.argmin()
    pick_thresh = confs[idx]
    dens_thresh_data = dens_thresh_data[
        np.isclose(dens_thresh_data.threshold, pick_thresh)
    ]
    assert dens_thresh_data.shape[0] == n_items
    assert len(dens_thresh_data.item_id.unique()) == n_items
    return dens_thresh_data


def tp_fp_fn_tn_rates(
    density_threshold_data, confidence_thresh=0, ao_neg_mgit_pos=True, bootstrap=False
):
    """
    Takes density_threshold_data as output by density_vs_threshold above.

    For each wsi in density_threshold_data, finds the density of
    predicted afb, including only predictions w/ confidence higher
    than confidence_thresh. Then, we compute tp, fp, fn, and tn rates using
    this density, relative to a range of threshold densities (unrelated to
    the confidence threshold).

    ao_neg_mgit_pos: chooses whether we should regard WSI which are AO- but
    AFB+ as positive (True, default) or negative (False)

    This is intended as a fairly simplistic baseline decision rule;
    hopefully we can beat this in the future with more data and fancier
    wsi-level classification models.
    """
    if ao_neg_mgit_pos:
        poscol = "afb_pos"
    else:
        poscol = "ao_pos"

    dens_thresh_data = densities_at_threshold(
        density_threshold_data, confidence_thresh=confidence_thresh
    )
    if bootstrap:
        dens_thresh_data = dens_thresh_data.sample(frac=1, replace=True)
    tpr = []
    fpr = []
    tnr = []
    fnr = []
    n_pos = dens_thresh_data[dens_thresh_data[poscol] == True].shape[0]
    n_neg = dens_thresh_data[dens_thresh_data[poscol] == False].shape[0]
    # to give a sense of scale, the lower cutoff for 1+ is 1e-2, and for 4+, it's 1e1
    thresholds = np.logspace(-5, 2, 300)
    for thresh in thresholds:
        dens_thresh_data["thresh+"] = dens_thresh_data.density > thresh
        tn = dens_thresh_data[
            (dens_thresh_data[poscol] == False) & (dens_thresh_data["thresh+"] == False)
        ].shape[0]
        fn = dens_thresh_data[
            (dens_thresh_data[poscol] == True) & (dens_thresh_data["thresh+"] == False)
        ].shape[0]
        tp = dens_thresh_data[
            (dens_thresh_data[poscol] == True) & (dens_thresh_data["thresh+"] == True)
        ].shape[0]
        fp = dens_thresh_data[
            (dens_thresh_data[poscol] == False) & (dens_thresh_data["thresh+"] == True)
        ].shape[0]
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
        tnr.append(tn / n_neg)
        fnr.append(fn / n_pos)
        # print(f"tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}")
    tpr = np.asarray(tpr)
    fpr = np.asarray(fpr)
    tnr = np.asarray(tnr)
    fnr = np.asarray(fnr)
    assert np.isclose(tpr + fnr, 1).all()
    assert np.isclose(fpr + tnr, 1).all()
    rates = {
        "thresholds": thresholds,
        "tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "fnr": fnr,
    }
    return rates


def get_1st_n_patches_w_bboxes(dataset, all_bboxes, n):
    """
    Assumes all_bboxes is a df as output by collect_gt_and_pred_bboxes, except
    that it is SORTED according to whatever criteria caller desires.
    This func goes through all_bboxes row-by-row and for each bbox, it finds the
    patch in dataset it overlaps with most, and then collects all bboxes which
    intersect that patch. It continues until it has accumulated n unique patches.
    Then it returns a torch Subset of dataset containing those n patches and
    a list of DataFrames, one DataFrame per patch, containing the bboxes which
    overlap the corresponding patch in the Subset.
    """
    assert isinstance(
        dataset.extents[0], PixelBox
    ), "unrecognized type for dataset.extents!"
    df_extents = pd.DataFrame(
        [
            (tile.left, tile.top, tile.right, tile.bottom, item_id)
            for tile, item_id in zip(dataset.extents, dataset.item_ids)
        ],
        columns=["x1", "y1", "x2", "y2", "item_id"],
    )
    # df_extents = coerce_pandas_item_ids(df_extents, ImageID)
    # all_bboxes = coerce_pandas_item_ids(all_bboxes, ImageID)
    ds_patch_indices = []
    patch_bboxes_dfs = []
    for i, row in enumerate(all_bboxes.itertuples()):
        bbox = XYXYBox.from_points(x1=row.x1, y1=row.y1, x2=row.x2, y2=row.y2)
        candidate_patches = filter_by_item_id_and_patch(df_extents, bbox, row.item_id)
        # find which row of candidate_patches has biggest overlap w/ bbox
        idx = (
            box_iou(
                torch.tensor((bbox.x1.x, bbox.x1.y, bbox.x2.x, bbox.x2.y)).reshape(
                    (1, 4)
                ),
                torch.tensor(candidate_patches[["x1", "y1", "x2", "y2"]].values),
            )
            .argmax()
            .item()
        )
        # now lookup corresponding index of that patch in the full dataset
        idx = candidate_patches.iloc[idx].name
        if idx not in ds_patch_indices:
            ds_patch_indices.append(idx)
            patch = dataset.extents[idx]
            patch_xyxy = XYXYBox.from_pixel_box(patch)
            patch_bboxes_dfs.append(
                filter_by_item_id_and_patch(all_bboxes, patch_xyxy, row.item_id)
            )
        if len(ds_patch_indices) == n:
            break
    return torch.utils.data.Subset(dataset, ds_patch_indices), patch_bboxes_dfs


def filter_by_item_id_and_patch(all_bboxes, xyxy_box, item_id):
    """
    Returns a copy of all_bboxes that retains only those rows which
    match item_id and intersect xyxy_box.
    """
    # all_bboxes = coerce_pandas_item_ids(all_bboxes, ImageID)
    assert isinstance(item_id, str)
    # partially overlapping is ok, PIL will not draw out of bounds of the image I think, extrapolating
    # from this: https://stackoverflow.com/questions/41528576/how-can-i-write-text-on-an-image-and-not-go-outside-of-the-border
    overlapping_bboxes = all_bboxes[
        (all_bboxes["item_id"] == item_id)
        & ~(
            (all_bboxes["x1"] > xyxy_box.x2.x)
            | (all_bboxes["x2"] < xyxy_box.x1.x)
            | (all_bboxes["y1"] > xyxy_box.x2.y)
            | (all_bboxes["y2"] < xyxy_box.x1.y)
        )
    ].copy()  # copy in case we're iterating over all_bboxes
    return overlapping_bboxes


def collect_gt_and_pred_bboxes(
    dataset,
    bbox_results,
    tgt_cls_idx,
    level: int = 0,
):
    """
    Collect all gnd truth and predicted bboxes from dataset and bbox_results,
    respectively, and assemble them all into a bbox_results-like dataframe,
    except with several extra columns.
    """
    if level != 0:
        raise NotImplementedError
    gt_bboxes = []
    # the shape of every image in the dataset must be the same, otherwise we
    # couldn't batch load, so to avoid loading every image in the dataset,
    # lookup image size outside loop
    *_, im_width, im_height = dataset[0]["image"].shape
    assert im_width == im_height, "only implemented for square images!"
    for patch in dataset:
        # note patch['target'] is itself a dict containing 'boxes' and 'labels'!
        # also target is already in pytorch format
        target = patch["target"]
        # first throw away bboxes w/ labels other than the class of interest
        target = filter_bboxes_by_objclass(target, tgt_cls_idx)
        patch_lefttop = Vec2D(patch["extent"].left, patch["extent"].top)
        spec = MetaData.specimen_lookup(item_id=patch["item_id"])
        for box in target["boxes"].numpy():
            # # TRICKY: we need to map the target bboxes to their correct
            # # coords in their original WSI. Note extent is in that original
            # # coord space, but target['boxes'] won't be if the image was
            # # resized, so we need to rescale target['boxes'] back
            rescale_fac = (patch["extent"].right - patch["extent"].left) / im_width
            box *= rescale_fac
            box = XYXYBox.from_points(
                x1=box[0], y1=box[1], x2=box[2], y2=box[3]
            ).offset(patch_lefttop)
            gt_bboxes.append(
                {
                    "item_id": patch["item_id"],
                    "ground_truth": True,
                    "x1": box.x1.x,
                    "y1": box.x1.y,
                    "x2": box.x2.x,
                    "y2": box.x2.y,
                    "confidence": np.nan,
                    "ao_label": spec.clsi_m48,
                    "wsi_positive": spec.afb_positive,
                }
            )
    gt_bboxes = pd.DataFrame(gt_bboxes)
    # set convenient defaults and partially override below
    gt_bboxes["false_neg"] = True

    pred_bboxes = bbox_results.copy()  # avoid dreaded SettingWithCopyWarning below
    pred_bboxes["false_pos"] = True

    # avoiding pandas FutureWarning about concat w/ empty dfs, following
    # suggestion here: https://stackoverflow.com/a/77467639
    concat_bboxes = pd.concat([df for df in (gt_bboxes, pred_bboxes) if not df.empty])
    all_bboxes = pd.DataFrame()
    # filter gt & preds by item_id, then call boxes_tp_fp_fn once on each whole-slide.
    # sort fails on class ImageID b/c __lt__ op isn't implemented or even meaningful
    for item_id, group_df in concat_bboxes.groupby("item_id", sort=False):
        # copy() to avoid SettingWithCopyWarning
        item_gts = group_df[group_df["ground_truth"] == True].copy()
        item_preds = group_df[group_df["ground_truth"] == False].copy()
        if (len(item_gts) > 0) and (len(item_preds) > 0):
            gnd_t = torch.tensor(item_gts[["x1", "y1", "x2", "y2"]].values)
            preds = torch.tensor(item_preds[["x1", "y1", "x2", "y2"]].values)
            _, _, fp_inds, _, fn_inds, _ = boxes_tp_fp_fn(gnd_t, preds)
            fp = [True if i in fp_inds else False for i in range(len(preds))]
            fn = [True if i in fn_inds else False for i in range(len(gnd_t))]
            item_gts["false_neg"] = fn
            item_preds["false_pos"] = fp
        all_bboxes = pd.concat((all_bboxes, item_gts, item_preds))
    return all_bboxes
