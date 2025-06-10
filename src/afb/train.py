import os
from pathlib import Path
from typing import List, Optional
from pprint import pprint
import warnings

import torch.distributed
import yaml
from comet_ml.integration.pytorch import log_model
import torch
import typer
from dotenv import load_dotenv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from afb.utils.validation import (
    density_vs_threshold,
    draw_categorization_vs_threshold,
    collect_gt_and_pred_bboxes,
)
from afb.utils.viz import show_im_grids_for_dataset
from afb.utils.postproc import post_proc_preds
from afb.utils.gpus import get_single_gpu_idx, choose_single_gpu_if_available
from afb.data.wsi_tiles_dataset import WSITilesDataset
from afb.data.transforms import ImgAugTransform, imgaug_worker_init_fn, collate_fn
from afb.data.image_annotation import AFBLabel
from afb.models.torch_viz_obj_det import TorchVizObjDet
from afb.inference import save_output


app = typer.Typer(
    # https://typer.tiangolo.com/tutorial/exceptions/
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=False,
)


@app.command()
def main(
    train_dir: Path,
    val_dir: Path,
    output_dir: Path,
    run_name: Optional[str] = None,
    weights: Optional[Path] = None,
    num_epochs: int = 100,
    batch_size: int = 168,
    learning_rate: float = 1e-4,
    lr_warmup_ratio: float = 1 / 20,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: int = 13000,
    weight_decay: float = 1e-3,
    checkpoint_epoch_freq: int = 5,
    checkpoint: Path = None,
    betas: List[float] = (0.91, 0.99),
    fscore_betasq: float = 0.8,
    dev: bool = False,
    freeze_backbone: bool = False,
    model_name: str = "fcos_resnet50",
    gpu: bool = True,
    devices: str = None,
    min_vram_avail: float = 0.8,
    optimizer: str = "adamw",
    early_stopping_patience: int = 50,
    num_dl_workers: int = 4,
    # gradient_clip_val: float = 0.5,
    # accumulate_grad_batches: int = 10,
    sync_batchnorm: bool = True,
    stochastic_weight_avg: bool = True,
    stochastic_weight_avg_lrs: float = 1e-2,
):
    # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/3
    torch.multiprocessing.set_sharing_strategy("file_system")

    load_dotenv()

    class_names = AFBLabel.get_class_names()

    if dev:
        gpu = False
    if not torch.cuda.is_available():
        gpu = False
    if gpu:
        # if cuda device has tensor cores
        # e.g. CUDA device ('NVIDIA RTX A6000')
        torch.set_float32_matmul_precision("medium")

        accelerator = "gpu"
        precision = "16-mixed"
        if devices is None:
            devices = f"{get_single_gpu_idx(min_vram_avail)},"
        else:
            # MM: my naive implementation of object-level metrics doesn't work
            # with DDP b/c there's no synchronization, so you only see logging
            # from one rank. It's approximately correct but still we should
            # fix if we need/want DDP
            raise NotImplementedError(
                "Need to fix validation metric logging for DDP/multi-GPU training"
            )
    else:
        accelerator = "cpu"
        precision = None
        devices = "1"

    # Convert all conf items to a string - avoids errors when items might be objects
    # or other things we cant dump to yaml
    output_dir.mkdir(exist_ok=True, parents=True)
    conf_items = dict((k, str(v)) for k, v in locals().items())
    with open(output_dir / "conf.yaml", "w") as fh:
        fh.write(yaml.safe_dump(conf_items))
    pprint(conf_items)

    train_dataset = WSITilesDataset(
        root_dir=train_dir,
        transforms=ImgAugTransform(),
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_dl_workers,
        pin_memory=True,
        worker_init_fn=imgaug_worker_init_fn,
        persistent_workers=True,
    )

    # might be worth exploring separate batchsizes for train & val dataloaders?
    # in initial test of FCOS model, train gpu utilization was above 80% or so
    # while val utilization was maybe ~30%
    val_dataset = WSITilesDataset(
        root_dir=val_dir,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_dl_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    print("# train images", len(train_dataset.img_names))
    print("# val images", len(val_dataset.img_names))

    if not dev:
        print("Comet workspace", os.environ.get("COMET_WORKSPACE"))
        logger = pl.loggers.CometLogger(
            save_dir=output_dir / "comet_logs",
            project_name="afb",
            workspace=os.environ.get("COMET_WORKSPACE"),
            rest_api_key=os.environ.get("COMET_API_KEY"),
        )
        logger.experiment.log_parameters(conf_items)
        if run_name:
            logger.experiment.set_name(run_name)
    else:
        logger = None

    if (
        model_name == "resnet_rcnn"
        or model_name == "convnext_rcnn"
        or model_name == "faster_rcnn"
        or model_name == "fcos_resnet50"
        or model_name == "convnext_fcos"
    ):
        model = TorchVizObjDet(
            learning_rate,
            weight_decay,
            betas,
            fscore_betasq,
            val_dataset,
            weights=weights,
            model_name=model_name,
            class_names=class_names,
            lr_warmup_iters=lr_warmup_iters,
            lr_decay_iters=lr_decay_iters,
            lr_warmup_ratio=lr_warmup_ratio,
            freeze_backbone=freeze_backbone,
            optimizer=optimizer,
            out_dir=output_dir,
        )

        if checkpoint is not None:
            model = TorchVizObjDet.load_from_checkpoint(
                checkpoint,
                map_location="cpu",
                class_names=class_names,
                optimizer=optimizer,
                model_name=model_name,
                lr_warmup_iters=lr_warmup_iters,
                lr_decay_iters=lr_decay_iters,
                out_dir=output_dir,
            )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    trainer_callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=output_dir,
            filename=run_name + "_{epoch}",
            monitor="fbeta",
            save_top_k=3,
            mode="max",
            verbose=True,
        ),
    ]

    if early_stopping_patience > 0:
        trainer_callbacks.append(
            EarlyStopping(
                monitor="fbeta",
                patience=early_stopping_patience,
                mode="max",
                verbose=True,
            )
        )

    if stochastic_weight_avg:
        trainer_callbacks.append(
            StochasticWeightAveraging(swa_lrs=stochastic_weight_avg_lrs)
        )

    model_tot_params = sum(p.numel() for p in model.parameters())
    model_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"Creating model with {model_trainable_params} trainable out of {model_tot_params} total parameters"
    )

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        logger=logger,
        fast_dev_run=dev,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=num_epochs,
        log_every_n_steps=1,  # change depending on num batches
        callbacks=trainer_callbacks,
        # gradient_clip_val=gradient_clip_val,
        # accumulate_grad_batches=accumulate_grad_batches,
        sync_batchnorm=sync_batchnorm,
        # strategy="ddp",
    )

    trainer.fit(
        model,
        train_data_loader,
        val_data_loader,
    )

    # # we have to manually destroy the ddp process group created by Trainer.fit
    # # before we save the model, run inference, etc
    # # see https://github.com/Lightning-AI/pytorch-lightning/issues/8375#issuecomment-879678629
    if torch.distributed.is_initialized():
        # barrier seems necessary to avoid weird race conditions where one
        # process finishing early crashes the whole group??
        trainer.strategy.barrier()
        torch.distributed.destroy_process_group()
        if trainer.global_rank != 0:
            # kill all but one process before continuing
            exit(0)

    print("saving model")
    # save model before inference & plotting so we've got the trained
    # statedict in case there's a bug/crash in inference/viz code
    torch.save(model.model.state_dict(), output_dir / f"{model_name}.pt")

    print("get model results")
    # apparently trainer.fit sends model back to cpu when it's done,
    # so we need to manually send it back to gpu before calling obj_det_predict
    device = choose_single_gpu_if_available(min_mem_avail=min_vram_avail, dev=dev)
    model.to(device)
    bbox_results, tiles_info = model.obj_det_predict(val_data_loader, dev=dev)
    bbox_results, item_meta = post_proc_preds(bbox_results, tiles_info)
    density_threshold_data = density_vs_threshold(bbox_results, item_meta)
    save_output(
        output_dir,
        bbox_results,
        item_meta,
        density_threshold_data,
    )

    print("draw categorization vs threshold")
    draw_categorization_vs_threshold(density_threshold_data, output_dir, agg=False)

    # collect_gt_and_pred_bboxes needs extra item-level metadata
    bbox_results = bbox_results.merge(item_meta, how="left", on="item_id")

    print("collect all gnd truth & predicted bboxes that match target_class label")
    all_bboxes = collect_gt_and_pred_bboxes(
        val_dataset, bbox_results, model.tgt_cls_idx
    )
    all_bboxes.to_csv(output_dir / "all_bboxes.csv", index=False)

    if not dev:
        # in dev mode these calls error out unpredictably b/c whether
        # or not false pos/false neg are present is quite random when
        # starting with an untrained model.
        print("draw image grids")
        show_im_grids_for_dataset(
            val_dataset,
            all_bboxes,
            output_dir=output_dir,
            logger=logger,
            priority="fp",
            n_ims=50,
            n_ims_per_grid=25,
        )
        show_im_grids_for_dataset(
            val_dataset,
            all_bboxes,
            output_dir=output_dir,
            logger=logger,
            priority="fn",
            n_ims=50,
            n_ims_per_grid=25,
        )
        show_im_grids_for_dataset(
            val_dataset,
            all_bboxes,
            output_dir=output_dir,
            logger=logger,
            priority="tn_wsi_fp",
            n_ims=50,
            n_ims_per_grid=25,
        )
        show_im_grids_for_dataset(
            val_dataset,
            all_bboxes,
            output_dir=output_dir,
            logger=logger,
            priority="highconf",
            n_ims=50,
            n_ims_per_grid=25,
        )

    if not dev:
        log_model(logger.experiment, model.model, model_name=model_name)
        logger.experiment.log_model(
            model_name,
            str(output_dir),
        )
        print("Finished logging model!")


if __name__ == "__main__":
    warnings.warn(
        "This script is not meant to be run directly, use the afb CLI instead.",
        DeprecationWarning,
    )
    app()
