# %%
from pathlib import Path

import pandas as pd
import torch

from afb.data.image_annotation import AFBLabel
from afb.data.wsi_tiles_dataset import WSITilesDataset
from afb.utils.viz import patch_grid_w_bboxes
from afb.utils.validation import collect_gt_and_pred_bboxes, get_1st_n_patches_w_bboxes

# %% [markdown]
# This is a hacky copy-paste of some code from `train.py` and `viz.py` that I used to interactively pick some nice tiles that aren't too cluttered so we can show some example tiles with confidence scores on the images directly. Basically I used the various priority sortings to find some interesting tiles, then I looked up the annotations on those tiles from the `patch_bboxes_dfs` list, then constructed a new dataframe of interesting "landmark" annotations to prepend to the filtering process that selects annotations/tiles for the grid.

# %%
val_dataset = WSITilesDataset(
    root_dir=Path(__file__).parents[1] / "datasets" / "obj_det_test",
)
bbox_results = pd.read_csv(
    Path(__file__).parents[1] / "results/paper_unfrozen_bb/obj_det_test/bbox_results.csv"
)
item_results = pd.read_csv(
    Path(__file__).parents[1] / "results/paper_unfrozen_bb/obj_det_test/item_results.csv"
)

# collect_gt_and_pred_bboxes needs extra item-level metadata
bbox_results = bbox_results.merge(item_results, how="left", on="item_id")

print("collect all gnd truth & predicted bboxes that match target_class label")
all_bboxes = collect_gt_and_pred_bboxes(val_dataset, bbox_results, AFBLabel.AFB.index())

all_bboxes["area"] = (all_bboxes["x2"] - all_bboxes["x1"]) * (
    all_bboxes["y2"] - all_bboxes["y1"]
)
all_bboxes

# %%
cherry_picked_rows = pd.read_csv(Path(__file__).parents[0] / "cherry_picked_tiles.csv")

# %%
color_tp = "green"
color_fp = "red"
color_fn = "yellow"
n_ims = 9
n_ims_per_grid = 9
# np.random.seed(2)
# sorted_df = all_bboxes.sample(frac=1).reset_index(drop=True)
sorted_df = pd.concat((cherry_picked_rows, all_bboxes))
sorted_df = sorted_df[(sorted_df.afb_label == 1) | sorted_df.afb_label.isna()].copy()
# sorted_df = sorted_df[(sorted_df.afb_label == 1) | (sorted_df.afb_label.isna())]
# false negatives
# sorted_df = all_bboxes.sort_values(by=["false_neg", "area"], ascending=False)
# highconf
# sorted_df = all_bboxes.sort_values(by=["confidence"], ascending=False)
# false pos
# sorted_df = all_bboxes.sort_values(by=["false_pos", "confidence"], ascending=False)
# short-listed
data_subset, patch_bboxes_dfs = get_1st_n_patches_w_bboxes(
    val_dataset, sorted_df, n_ims
)
assert len(data_subset) == len(patch_bboxes_dfs)
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
            color_tp=color_tp,
            color_fp=color_fp,
            color_fn=color_fn,
            show_labels=True,
            rescale_scores=None,
            title="",
            text_offset=(-3, -15),
            dpi=900,
        )
    )
im_grids

# %%
im_grids[0].savefig(Path(__file__).parents[0] / "cherry_picked_tiles.png")

# %%
