import logging
from pathlib import Path
from typing import Optional
import warnings

import typer
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from afb.utils.gpus import choose_single_gpu_if_available
from afb.utils.postproc import post_proc_preds
from afb.models.torch_viz_obj_det import TorchVizObjDet
from afb.data.transforms import collate_fn
from afb.data.wsi_tiles_dataset import WSITilesDataset

# from afb.data.schema.pandas import coerce_pandas_item_ids
from afb.utils.validation import (
    density_vs_threshold,
    draw_categorization_vs_threshold,
)
from afb.utils.viz import wsi_ROC

logger = logging.getLogger(__name__)
app = typer.Typer()


def load_model(
    model_name,
    weights,
    device,
    out_dir,
):
    if weights.suffix == ".ckpt":
        # then we got a complete checkpoint, not just the statedict
        model = TorchVizObjDet.load_from_checkpoint(
            weights,
            map_location=torch.device("cpu"),
            model_name=model_name,
        )
        model.eval()
        model.to(device)
    # else we must've got a plain .pt statedict, so create model directly
    elif (
        model_name == "faster_rcnn"
        or model_name == "convnext_rcnn"
        or model_name == "resnet_rcnn"
        or model_name == "fcos_resnet50"
        or model_name == "convnext_fcos"
    ):
        model = TorchVizObjDet(
            weights=weights,
            device=device,
            model_name=model_name,
            out_dir=out_dir,
        )
        model.eval()
    else:
        raise ValueError(f"Unsupported model {model_name}")
    return model


def save_output(
    out_dir: Path,
    bbox_results: pd.DataFrame,
    item_results: pd.DataFrame,
    density_threshold_data: pd.DataFrame,
):
    if bbox_results.empty:
        return  # No results to write

    bboxes = bbox_results.copy()
    # bboxes = coerce_pandas_item_ids(bboxes, str)
    items = item_results.copy()
    # items = coerce_pandas_item_ids(items, str)
    densities = density_threshold_data.copy()
    # densities = coerce_pandas_item_ids(densities, str)

    if out_dir is not None:
        bboxes.to_csv(out_dir / "bbox_results.csv", index=False)
        items.to_csv(out_dir / "item_results.csv", index=False)
        densities.to_csv(out_dir / "density_threshold.csv", index=False)


@app.command()
def infer(
    model_name: str,
    model_weights: Path,
    data_path: Path,
    out_dir: Optional[Path] = None,
    min_vram_avail: float = 0.8,
    device: int = None,
    dev: bool = False,
    batch_size: int = 256,
):
    load_dotenv()

    # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/3
    torch.multiprocessing.set_sharing_strategy("file_system")

    out_dir.mkdir(exist_ok=True)

    dataset = WSITilesDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    if device is None:
        device = choose_single_gpu_if_available(min_mem_avail=min_vram_avail, dev=dev)
    else:
        device = torch.device(f"cuda:{device}")
    model = load_model(model_name, model_weights, device, out_dir)

    bbox_results, tiles_info = model.obj_det_predict(dataloader, dev)

    bbox_results, item_meta = post_proc_preds(bbox_results, tiles_info)
    logger.debug("Computing densities...")
    density_threshold_data = density_vs_threshold(bbox_results, item_meta)

    logger.debug("Saving output...")
    save_output(
        out_dir,
        bbox_results,
        item_meta,
        density_threshold_data,
    )
    logger.debug("drawing categorization vs threshold...")
    draw_categorization_vs_threshold(density_threshold_data, out_dir, agg=False)
    if item_meta.wsi_positive.any() and (~item_meta.wsi_positive).any():
        # a WSI-level ROC only makes sense if we have true positives & negatives
        # in the dataset, else we'll get div-by-0
        logger.debug("Generating WSI ROC fig...")
        fig, rates = wsi_ROC(density_threshold_data)
        fig.savefig(out_dir / "wsi_ROC.png")
        pd.DataFrame(rates).to_csv(out_dir / "wsi_ROC_rates.csv", index=False)


if __name__ == "__main__":
    warnings.warn(
        "This script is not meant to be run directly, use the afb CLI instead.",
        DeprecationWarning,
    )
    app()
