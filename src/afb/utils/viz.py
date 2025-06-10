import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import draw_bounding_boxes, make_grid
from PIL import Image, ImageColor, ImageDraw, ImageFont

from afb.utils.validation import get_1st_n_patches_w_bboxes, tp_fp_fn_tn_rates


def _rescale_scores(scores, range):
    """
    Maps scores from the interval [range[0], range[1]] to [0, 1] for better
    viz & more contrast (especially useful b/c FCOS object scores tend to end
    up mostly bunched between 0.2 and about 0.4 or 0.5, so the transparency on
    boxes all looks about the same.
    """
    rescaled = (scores - range[0]) / (range[1] - range[0])
    if (rescaled < 0).any():
        warnings.warn(
            "Found conf scores < 0 in _rescale_scores after rescaling, check provided range"
        )
    if (rescaled > 1).any():
        warnings.warn(
            "Found conf scores > 1 in _rescale_scores after rescaling, check provided range"
        )
    return rescaled.clamp(min=0, max=1)


def _parse_pred_colors(
    colors: Union[str, List[str]],
    num_objects: int,
    scores: torch.Tensor,
) -> List[Tuple[int, int, int, int]]:
    """
    2023-11-17: MJM, Forked from torchvision.utils to handle transparency.
    Converts predicted box scores to a transparency.
    """
    if isinstance(colors, list):
        if len(colors) < num_objects:
            raise ValueError(
                f"Number of colors must be equal or larger than the number of objects, but got {len(colors)} < {num_objects}."
            )
    else:  # colors specifies a single color for all objects
        colors = [colors] * num_objects
    scaled_scores = [int(score * 255) for score in scores]
    return [
        ImageColor.getrgb(color) + (score,)
        for (color, score) in zip(colors, scaled_scores)
    ]


@torch.no_grad()
def draw_pred_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    colors: Optional[Union[str, List[str]]] = None,
    labels: Optional[List[str]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    text_offset: Tuple[int, int] = None,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
) -> torch.Tensor:
    """
    2023-11-17: MJM, Forked from torchvision.utils to handle transparency.

    Draws bounding boxes on given image.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
    elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any():
        raise ValueError(
            "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
        )

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    if colors is None:
        colors = "green"
    colors = _parse_pred_colors(colors, num_objects=num_boxes, scores=scores)

    if text_offset is None:
        text_offset = (-3, -15)

    if font is None:
        if font_size is not None:
            warnings.warn(
                "Argument 'font_size' will be ignored since 'font' is not set."
            )
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10)

    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.to(torch.int64).tolist()

    # if fill:
    draw = ImageDraw.Draw(img_to_draw, "RGBA")
    # else:
    #     draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            margin = width + 1
            draw.text(
                (bbox[0] + text_offset[0] + margin, bbox[1] + text_offset[1] + margin),
                label,
                fill=color,
                font=txt_font,
            )

    return (
        torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)
    )


def patch_grid_w_bboxes(
    dataset,
    patch_bboxes_dfs,
    name=None,
    color_tp="green",
    color_fp="red",
    color_fn="yellow",
    dpi=300,
    title=None,
    **kwargs,
):
    """
    Assumes dataset and patch_bboxes_dfs are as output by get_1st_n_patches_w_bboxes,
    i.e., dataset consists of n patches and patch_bboxes_dfs is a list of n DataFrames,
    each w/ columns as output by by collect_gt_and_pred_bboxes. Typically dataset
    would be a Subset of an actual WSITilesDataset.

    Assumes object class is the same for all boxes in patch_bboxes_dfs, as filtered
    by get_results, & ignores bboxes (if present) in the dataset itself.
    """
    if name is None:
        name = ""
    n_rows = int(np.sqrt(len(dataset)))

    pred_images = _draw_bboxes_on_patches(
        dataset=dataset,
        patch_bboxes_dfs=patch_bboxes_dfs,
        color_tp=color_tp,
        color_fp=color_fp,
        color_fn=color_fn,
        **kwargs,
    )
    predicted_img_grid = make_grid(pred_images, nrow=n_rows)

    fig_height = 3 * n_rows
    fig, pred_ax = plt.subplots(1, 1, figsize=(fig_height, fig_height), dpi=dpi)

    predicted_img_grid = predicted_img_grid.permute(1, 2, 0).numpy()
    pred_ax.imshow(predicted_img_grid)
    if title is None:
        pred_ax.set_title(
            f"Predictions {name}, TP: {color_tp}, FP: {color_fp}, FN: {color_fn}"
        )
    else:
        pred_ax.set_title(title)
    pred_ax.axis("off")

    return fig


def _draw_bboxes_on_patches(
    dataset,
    patch_bboxes_dfs,
    color_tp=None,
    color_fp=None,
    color_fn=None,
    bbox_lw=3,
    show_labels=None,
    rescale_scores=None,
    text_offset: Tuple[int, int] = None,
):
    """
    Intended for use by patch_grid_w_bboxes and not tested much
    otherwise; user beware! Note this assumes all boxes in all_bboxes
    are the same object class of interest - we'll have to revisit this
    if/when we want to consider a true multi-class problem.
    """
    images = []
    for idx, item in enumerate(dataset):
        img = item["image"]
        img = (img * 255).to(torch.uint8)
        # patch_bboxes_dfs are still in global wsi coords, so need to
        # convert to patch local coords before passing to viz funcs.
        item_bboxes = patch_bboxes_dfs[idx].copy()
        # Is there a tidier/faster way to write this? Tried an apply/lambda but was
        # hard to read. Shouldn't matter b/c df for one patch will never be huge
        patch_left, patch_top = item["extent"].left, item["extent"].top
        item_bboxes["x1"] = item_bboxes["x1"] - patch_left
        item_bboxes["y1"] = item_bboxes["y1"] - patch_top
        item_bboxes["x2"] = item_bboxes["x2"] - patch_left
        item_bboxes["y2"] = item_bboxes["y2"] - patch_top
        # tricky: now item_bboxes are in patch local coords, but at mpp of
        # the ORIGINAL WSI, not at the mpp of img! so need to rescale in
        # order to correctly draw on img
        *_, im_width, im_height = dataset[0]["image"].shape
        assert im_width == im_height, "Reimplement for non-square images!"
        rescale_fac = im_width / (item["extent"].right - item["extent"].left)
        item_bboxes[["x1", "y1", "x2", "y2"]] *= rescale_fac

        gt_bboxes = item_bboxes[item_bboxes["ground_truth"] == True]
        pred_bboxes = item_bboxes[item_bboxes["ground_truth"] == False]
        fns = gt_bboxes[gt_bboxes["false_neg"] == True]
        fps = pred_bboxes[pred_bboxes["false_pos"] == True]
        tps = pred_bboxes[pred_bboxes["false_pos"] == False]

        tp_scores = torch.tensor(tps["confidence"].values)
        fp_scores = torch.tensor(fps["confidence"].values)
        if len(fns) > 0:
            img = draw_bounding_boxes(
                img,
                torch.tensor(fns[["x1", "y1", "x2", "y2"]].values),
                colors=color_fn,
                width=bbox_lw,
            )
        if len(fps) > 0:
            if show_labels:
                labels = [f"{score.item():.2f}" for score in fp_scores]
            else:
                labels = None
            if rescale_scores is not None:
                fp_scores = _rescale_scores(fp_scores, rescale_scores)
            img = draw_pred_bounding_boxes(
                img,
                torch.tensor(fps[["x1", "y1", "x2", "y2"]].values),
                fp_scores,
                colors=color_fp,
                labels=labels,
                width=bbox_lw,
                text_offset=text_offset,
            )
        if len(tps) > 0:
            if show_labels:
                labels = [f"{score.item():.2f}" for score in tp_scores]
            else:
                labels = None
            if rescale_scores is not None:
                tp_scores = _rescale_scores(tp_scores, rescale_scores)
            img = draw_pred_bounding_boxes(
                img,
                torch.tensor(tps[["x1", "y1", "x2", "y2"]].values),
                tp_scores,
                colors=color_tp,
                labels=labels,
                width=bbox_lw,
                text_offset=text_offset,
            )
        images.append(img)
    return images


def show_im_grids_for_dataset(
    dataset,
    all_bboxes,
    output_dir=None,
    name_prefix=None,
    logger=None,
    priority="fp",
    color_tp="green",
    color_fp="red",
    color_fn="yellow",
    n_ims=100,
    n_ims_per_grid=100,
    show_labels=True,
    rescale_scores=None,
    title=None,
    text_offset: Tuple[int, int] = None,
):
    """
    Take a dataset and pre-computed all_bboxes (as output by
    collect_gt_and_pred_bboxes) and generates grid(s) of patches
    with gt & pred bboxes drawn on them. Chooses patches to show
    according to priority:
        - if priority='fp', sorts patches by highest confidence
        false-positive predictions
        - if priority='fn', sorts patches by largest false-negative
        gt boxes that were missed by predictions (largest by area, since
        many fn bboxes seem to be on patch edges and probably not so
        important)
        - if priority='tn_wsi_fp', selects only wsi which are ground-
        truth negative and shows the highest confidence false-positive
        predictions
        - if priority='highconf', sorts patches by highest confidence
        predictions, ignoring whether true- or false-positive
        - if priority='random', randomly chooses patches
    If n_ims > n_ims_per_grid, outputs multiple images. For best visual
    appearance, choose n_ims_per_grid to be a perfect square.
    """
    if n_ims > len(dataset):
        n_ims = len(dataset)
    if n_ims_per_grid > 100:
        warnings.warn(
            f"Do you really want > {n_ims_per_grid} images in a single grid???"
        )
    if priority == "fp":
        sorted_df = all_bboxes.sort_values(
            by=["false_pos", "confidence"], ascending=False
        )
    elif priority == "fn":
        all_bboxes["area"] = (all_bboxes["x2"] - all_bboxes["x1"]) * (
            all_bboxes["y2"] - all_bboxes["y1"]
        )
        sorted_df = all_bboxes.sort_values(by=["false_neg", "area"], ascending=False)
    elif priority == "tn_wsi_fp":
        sorted_df = all_bboxes[all_bboxes["wsi_positive"] == False].sort_values(
            by=["false_pos", "confidence"], ascending=False
        )
    elif priority == "highconf":
        sorted_df = all_bboxes.sort_values(by=["confidence"], ascending=False)
    elif priority == "random":
        sorted_df = all_bboxes.sample(frac=1).reset_index(drop=True)
    else:
        raise ValueError("Unrecognized string supplied for priority!")
    data_subset, patch_bboxes_dfs = get_1st_n_patches_w_bboxes(
        dataset, sorted_df, n_ims
    )
    assert len(data_subset) == len(patch_bboxes_dfs)
    # if n_ims > n_ims_per_grid, need to chunk indices to supply
    # to each separate image grid, but Subset needs a Sequence of
    # indices, not a slice, so lean into the antipattern
    indices = list(range(len(data_subset)))
    chunked_slices = [
        slice(i, i + n_ims_per_grid) for i in range(0, len(indices), n_ims_per_grid)
    ]
    im_grids = []
    for i, s in enumerate(chunked_slices):
        im_grids.append(
            patch_grid_w_bboxes(
                torch.utils.data.Subset(data_subset, indices[s]),
                patch_bboxes_dfs[s],
                name=f"{priority.upper()} grid {i}",
                color_tp=color_tp,
                color_fp=color_fp,
                color_fn=color_fn,
                show_labels=show_labels,
                rescale_scores=rescale_scores,
                title=title,
                text_offset=text_offset,
            )
        )
    if output_dir is not None:
        if name_prefix is None:
            name_prefix = f"patches_grid_{priority}"
        else:
            name_prefix = f"{name_prefix}_patches_grid_{priority}"
        for i, fig in enumerate(im_grids):
            fig.savefig(output_dir / f"{name_prefix}_{i}.png")
    if logger is not None:
        for i, fig in enumerate(im_grids):
            logger.experiment.log_figure(figure=fig)
    return im_grids


def wsi_ROC(
    dens_thresh_dat,
    confidence_thresh=0.0,
    figsize=(10, 4),
    dpi=600,
    ax=None,
    colors=("red", "blue", "green"),
    bootstrap_resamps=0,
):
    """
    Takes density_threshold_data as output by density_vs_threshold in
    utils/validation.py.

    If provided, ax should be a tuple of two axes for this func to use.

    bootstrap_resamps: if > 0, use simple nonparametric bootstrap resampling to
    estimate confidence bands on ROC curve and conditional density CDFs.

    MM: experimented with excluding AO-/MGIT+ samples but the distribution
    of densities for AO-/MGIT+ WSI and AO+ WSI was (surprisingly!) virtually
    identical. And the code to do so was confusing & messy so I removed it.
    """

    rates = tp_fp_fn_tn_rates(
        dens_thresh_dat, confidence_thresh=confidence_thresh, ao_neg_mgit_pos=True
    )
    auroc_actual = np.trapz(rates["tnr"], x=rates["fnr"])
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    else:
        fig = None

    if bootstrap_resamps > 0:
        lq = 0.05
        uq = 0.95
        tnr_resamples = np.zeros((bootstrap_resamps, len(rates["thresholds"])))
        fnr_resamples = np.zeros((bootstrap_resamps, len(rates["thresholds"])))
        roc_proj_grid = np.linspace(0, 1, 150)
        roc_proj = np.zeros((bootstrap_resamps, roc_proj_grid.shape[0]))
        auroc_resamples = []
        for i in range(bootstrap_resamps):
            resamp = tp_fp_fn_tn_rates(
                dens_thresh_dat,
                confidence_thresh=confidence_thresh,
                ao_neg_mgit_pos=True,
                bootstrap=True,
            )
            assert np.isclose(
                resamp["thresholds"], rates["thresholds"], atol=1e-12, rtol=1e-6
            ).all()
            auroc_resamples.append(np.trapz(resamp["tnr"], x=resamp["fnr"]))
            roc_proj[i] = np.interp(roc_proj_grid, resamp["fnr"], resamp["tnr"], left=0)
            tnr_resamples[i] = resamp["tnr"]
            fnr_resamples[i] = resamp["fnr"]
        auroc_resamples = np.asarray(auroc_resamples)
        auroc_mean = np.mean(auroc_resamples)
        assert np.isclose(
            auroc_actual, auroc_mean, rtol=5e-3
        ), "Need more resamplings??"
        auroc_std = np.std(auroc_resamples)
        auroc_label = f"area under curve: {auroc_actual:.2f} +/- {2*auroc_std:.2f}"
        lq_roc_curve, uq_roc_curve = np.quantile(roc_proj, (lq, uq), axis=0)
        ax[0].fill_between(
            roc_proj_grid,
            lq_roc_curve,
            uq_roc_curve,
            alpha=0.3,
            color=colors[0],
        )
        lq_tnr_curve, uq_tnr_curve = np.quantile(tnr_resamples, (lq, uq), axis=0)
        lq_fnr_curve, uq_fnr_curve = np.quantile(fnr_resamples, (lq, uq), axis=0)
        ax[1].fill_between(
            rates["thresholds"], lq_tnr_curve, uq_tnr_curve, alpha=0.3, color=colors[1]
        )
        ax[1].fill_between(
            rates["thresholds"], lq_fnr_curve, uq_fnr_curve, alpha=0.3, color=colors[2]
        )
    else:
        auroc_label = f"area under curve: {auroc_actual:.2f}"

    ax[0].plot(rates["fnr"], rates["tnr"], label=auroc_label, color=colors[0])
    ax[1].semilogx(rates["thresholds"], rates["tnr"], label="TNR", color=colors[1])
    ax[1].semilogx(rates["thresholds"], rates["fnr"], label="FNR", color=colors[2])

    ax[0].grid()
    ax[0].set_xlim(-0.02, 1.02)
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_xlabel("False negative rate")
    ax[0].set_ylabel("True negative rate")
    ax[0].set_title("WSI True vs. False Negative Rate")
    ax[0].legend(loc="lower right")

    ax[1].set_xlabel("Density threshold (AFB per FoV @ 1000x)")
    ax[1].set_ylabel("True/false negative rates")
    ax[1].grid()
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_xlim(3e-5, 1e2)
    ax[1].set_title("WSI Classification vs. Threshold")
    ax[1].axvline(1e-2, color=(0, 0, 0, 0.2), linestyle="--", linewidth=2)
    ax[1].axvline(1e-1, color=(0, 0, 0, 0.2), linestyle="--", linewidth=2)
    ax[1].axvline(1e0, color=(0, 0, 0, 0.2), linestyle="--", linewidth=2)
    ax[1].axvline(1e1, color=(0, 0, 0, 0.2), linestyle="--", linewidth=2)
    ax[1].axvspan(1e-2, 1e-1, color=(0, 0, 0, 0.02), linewidth=0.2, linestyle="--")
    ax[1].axvspan(1e-1, 1e0, color=(0, 0, 0, 0.04), linewidth=0.2, linestyle="--")
    ax[1].axvspan(1e0, 1e1, color=(0, 0, 0, 0.06), linewidth=0.2, linestyle="--")
    ax[1].axvspan(1e1, 1e3, color=(0, 0, 0, 0.07), linewidth=0.2, linestyle="--")
    ax[1].text(2e-2, 0.05, "1+")
    ax[1].text(2e-1, 0.05, "2+")
    ax[1].text(2e0, 0.05, "3+")
    ax[1].text(2e1, 0.05, "4+")
    ax[1].legend(loc="upper left")
    return fig, rates


def pr_curve(recall, precision, epoch, beta, fbeta, t):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    print("recall:", recall)
    print("precision:", precision)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid()
    ax.set_title(
        f"Precision-recall at epoch {epoch} ($f_{{\\beta={beta:.2f}}} = {fbeta:.3f}, t = {t}$)"
    )
    return fig
