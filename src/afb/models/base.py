import json
import logging
from collections import defaultdict

import torch
import torch.distributed
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pytorch_lightning as pl
from tqdm import tqdm
import pandas as pd
import torch
from torchvision.models.detection.fcos import FCOS
import numpy as np
import matplotlib.pyplot as plt

from afb.utils.training import WarmupCosineLRScheduler
from afb.utils.validation import boxes_tp_fp_fn
from afb.data.schema.pandas import BBOX_SCHEMA
from afb.data.image_annotation import filter_bboxes_by_objclass

from afb.config import (
    BOX_SCORE_THRESH_LOW,
    BOX_SCORE_THRESH_HIGH,
    BOX_SCORE_THRESH_STEPS,
)


class BaseLightningModule(pl.LightningModule):
    def __init__(self, val_dataset, fscore_betasq, optimizer, out_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mean_ap = MeanAveragePrecision(extended_summary=True)

        # need quantities below to be tensors to handle 0/0 div gracefully
        # but they don't need to be included in state_dict, so
        # set persistent=False
        self.register_buffer(
            "box_score_thresh",
            torch.linspace(
                BOX_SCORE_THRESH_LOW,
                BOX_SCORE_THRESH_HIGH,
                steps=BOX_SCORE_THRESH_STEPS,
            ),
            persistent=False,
        )
        self.register_buffer(
            "val_tps", torch.zeros_like(self.box_score_thresh), persistent=False
        )
        self.register_buffer(
            "val_tns", torch.zeros_like(self.box_score_thresh), persistent=False
        )
        self.register_buffer(
            "val_fps", torch.zeros_like(self.box_score_thresh), persistent=False
        )
        self.register_buffer(
            "val_fns", torch.zeros_like(self.box_score_thresh), persistent=False
        )
        self.fbet_sq = fscore_betasq

        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.out_dir = out_dir
        self.val_metrics = {"box_score_thresh": self.box_score_thresh.cpu().tolist()}

    def infer__call__(self, batch):
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch)
        return [
            (
                torch.column_stack([img_result["boxes"], img_result["scores"]]),
                img_result["labels"],
            )
            for img_result in output
        ]

    def forward(self, x):
        # trainer.predict doesn't split the image & target for us
        if isinstance(x, dict):
            x = x["image"]
        if isinstance(x, list) or isinstance(x, tuple):
            x, _target = x
        return self.model(x)

    def _collect_batch_predictions(self, outputs, batch, all_patch_results):
        empty_batch_preds = True
        # batch['image'] should have shape batchsize x channels x w x h,
        # but I don't want to deal w/ non-square images
        assert batch["image"].shape[-1] == batch["image"].shape[-2]
        im_size = batch["image"].shape[-1]
        for preds, item_id, extent in zip(outputs, batch["item_id"], batch["extent"]):
            # preds has all the preds for a single patch
            if len(preds["scores"]) > 0:
                # TRICKY: we need to map the pred bboxes to their correct
                # coords in their original WSI. Note extent is in that original
                # coord space, but preds['boxes'] won't be if the image was
                # resized, so we need to rescale preds['boxes'] back
                empty_batch_preds = False
                rescale_fac = (extent.right - extent.left) / im_size
                assert (extent.right - extent.left) == (
                    extent.bottom - extent.top
                ), "non-square image, not implemented!"
                offset = torch.as_tensor(
                    [extent.left, extent.top, extent.left, extent.top],
                    device=preds["boxes"].device,
                ).reshape(1, 4)
                patch_results = pd.DataFrame(
                    (rescale_fac * preds["boxes"] + offset).detach().cpu().numpy(),
                    columns=[
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    ],
                )
                patch_results["confidence"] = preds["scores"].detach().cpu().numpy()
                patch_results["afb_label"] = preds["labels"].detach().cpu().numpy()
                patch_results["ground_truth"] = False
                patch_results["item_id"] = item_id
                all_patch_results.append(patch_results)
        return empty_batch_preds

    def _collect_batch_extents(self, batch, all_tiles_info):
        tiles_info = {item_id: [] for item_id in set(batch["item_id"])}
        # we could add a short-circuit option for wsi inference where we
        # check if there's only one item_id, and if so just dump the whole
        # list of extents into tiles_info rather than looping over & appending
        for item_id, extent in zip(batch["item_id"], batch["extent"]):
            tiles_info[item_id].append(extent)
        all_tiles_info.append(tiles_info)
        return None

    def training_step(self, batch, batch_idx):
        image, target = batch["image"], batch["target"]
        loss_dict = self.model(image, target)
        losses = sum(loss for loss in loss_dict.values())

        batch_size = len(image[0])
        self.log_dict(loss_dict, batch_size=batch_size)
        self.log(
            "train_loss",
            losses,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return losses

    def validation_step(self, batch, batch_idx):
        image, target = batch["image"], batch["target"]
        preds = self.model(image)
        self.mean_ap.update(preds, target)
        return {"predictions": preds}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Compute validation true/false pos/negatives for this batch & add
        to running total for end-of-epoch logging.
        """
        # recall target is a list of len batchsize, each element has keys
        # 'boxes' and 'labels' whose values are tensors w/ shape Nx4 and N,
        # respectively. preds is sim except each list element has keys
        # 'boxes', 'labels', and 'scores', w/ shapes Nx4, N, & N resp
        target = batch["target"]
        preds = outputs["predictions"]

        for pred, targ in zip(preds, target):
            # first drop any bboxes w/ labels other than afb
            pred = filter_bboxes_by_objclass(pred, self.tgt_cls_idx)
            targ = filter_bboxes_by_objclass(targ, self.tgt_cls_idx)
            for i, thresh in enumerate(self.box_score_thresh):
                # first threshold predicted boxes by scores: ignore any boxes
                # with low scores.
                highscores = pred["scores"] > thresh
                inds = torch.nonzero(highscores).flatten()
                pred_boxes = torch.index_select(pred["boxes"], 0, inds)
                n_targ_boxes = len(targ["boxes"])
                n_pred_boxes = len(pred_boxes)
                if n_targ_boxes == 0 and n_pred_boxes == 0:
                    # This is a true negative case
                    self.val_tns[i] += 1
                elif n_targ_boxes == 0:
                    # then all the pred boxes are false positives
                    self.val_fps[i] += n_pred_boxes
                elif n_pred_boxes == 0:
                    # then we missed all the gt boxes -> all false neg
                    self.val_fns[i] += n_targ_boxes
                else:  # we know n_targ_boxes > 0 and n_pred_boxes > 0
                    tp_inds, _, fp_inds, _, fn_inds, _ = boxes_tp_fp_fn(
                        targ["boxes"], pred_boxes
                    )
                    self.val_tps[i] += len(tp_inds)
                    self.val_fps[i] += len(fp_inds)
                    self.val_fns[i] += len(fn_inds)

    def on_validation_epoch_end(self, *args, **kwargs):
        mean_ap = self.mean_ap.compute()
        self.log_dict(
            {
                k: v
                for k, v in mean_ap.items()
                if k != "classes"
                and k != "ious"
                and k != "precision"
                and k != "recall"
                and k != "scores"
            }
        )

        eps = 1e-10  # avoid divide by zero
        specificity = self.val_tns / (self.val_tns + self.val_fps + eps)

        num_positives = self.val_tps + self.val_fns
        num_negatives = self.val_tns + self.val_fps
        tpr = self.val_tps / (num_positives + eps)
        fpr = self.val_fps / (num_negatives + eps)
        plt.figure()
        plt.plot(fpr.tolist(), tpr.tolist())
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(self.out_dir / "roc_curve.png")

        recall = self.val_tps / (self.val_tps + self.val_fns + eps)
        precision = self.val_tps / (self.val_tps + self.val_fps + eps)
        # b/c nan's will wreck max below, convert nans to zero now
        recall = torch.nan_to_num(recall)
        precision = torch.nan_to_num(precision)

        # need to bypass lightning and access underlying comet logger
        # for specialty things like log_curve
        self.logger.experiment.log_curve(
            "pr_curve", x=recall.tolist(), y=precision.tolist(), step=self.current_epoch
        )
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        # send tensors to cpu, then matplotlib will cast to numpy
        ax.plot(recall.cpu(), precision.cpu())
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.1))
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.grid()
        ax.set_title(f"Precision-recall at epoch {self.current_epoch}")
        plt.savefig(self.out_dir / "pr_curve.png")
        self.logger.experiment.log_figure(figure=fig)
        fbeta = (1 + self.fbet_sq) * (
            precision * recall / (self.fbet_sq * precision + recall)
        )
        # nans in fbeta are the real prob b/c they break prf_dict logging
        fbeta = torch.nan_to_num(fbeta)

        self.val_metrics[f"epoch_{self.current_epoch}"] = {
            "fbeta": fbeta.cpu().tolist(),
            "precision": precision.cpu().tolist(),
            "recall": recall.cpu().tolist(),
        }
        with open(self.out_dir / "val_metrics.json", "w") as f:
            json.dump(self.val_metrics, f)

        # log the maximum fbeta achieved and the corresponding prec & recall
        max_fbeta, idx_max = torch.max(fbeta, dim=0)

        # see https://github.com/Lightning-AI/pytorch-lightning/issues/18803,
        # sounds like now we should leave metrics on gpu for logging, esp in
        # ddp, rather than moving to cpu?
        to_scalar = lambda x: x.item()
        prf_dict = {
            "prec": to_scalar(precision[idx_max]),
            "recall": to_scalar(recall[idx_max]),
            "fbeta": to_scalar(max_fbeta),
            "specificity": to_scalar(specificity[idx_max]),
            "tn": int(to_scalar(self.val_tns[idx_max])),
            "fp": int(to_scalar(self.val_fps[idx_max])),
            "fn": int(to_scalar(self.val_fns[idx_max])),
            "tp": int(to_scalar(self.val_tps[idx_max])),
            "tpr": to_scalar(tpr[idx_max]),
            "fpr": to_scalar(fpr[idx_max]),
        }
        self.log_dict(prf_dict, on_epoch=True, logger=True, sync_dist=True)
        with open(self.out_dir / "performance_metrics.json", "w") as f:
            json.dump(prf_dict, f)

        logging.info(f"tile/box confusion matrix (idx: {idx_max}, fbeta: {max_fbeta})")
        logging.info(f"+-------------+--------------+--------------+------------+")
        logging.info(f"|             |    pp        |    pn        |            |")
        logging.info(f"+-------------+--------------+--------------+------------+")
        logging.info(
            f'| p: {int(num_positives[idx_max]):8} | tp: {prf_dict["tp"]:8} | fn: {prf_dict["fn"]:8} | tpr: {prf_dict["tpr"]:.3f} |'
        )
        logging.info(
            f'| n: {int(num_negatives[idx_max]):8} | fp: {prf_dict["fp"]:8} | tn: {prf_dict["tn"]:8} | fpr: {prf_dict["fpr"]:.3f} |'
        )
        logging.info(f"+-------------+--------------+--------------+------------+")

        # type_as is needed to preserve device & avoid cuda errors
        self.val_tps = torch.zeros_like(self.box_score_thresh).type_as(self.val_tps)
        self.val_tns = torch.zeros_like(self.box_score_thresh).type_as(self.val_tns)
        self.val_fns = torch.zeros_like(self.box_score_thresh).type_as(self.val_fns)
        self.val_fps = torch.zeros_like(self.box_score_thresh).type_as(self.val_fps)

        self.mean_ap.reset()

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer == "lamb":
            from torch_optimizer import Lamb

            opt = Lamb(
                params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=self.hparams.betas,
            )
        elif self.optimizer == "adamw":
            opt = torch.optim.AdamW(
                params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=self.hparams.betas,
            )
        else:
            raise Exception(f"Unknown optimizer `{self.optimizer}`")

        lr_schedule = WarmupCosineLRScheduler(
            optimizer=opt,
            min_lr=self.hparams.learning_rate * self.hparams.lr_warmup_ratio,
            max_lr=self.hparams.learning_rate,
            warmup_iters=self.hparams.lr_warmup_iters,
            lr_decay_iters=self.hparams.lr_decay_iters,
        )
        scheduler_config = {
            "scheduler": lr_schedule,
            "interval": "step",  # could also be epoch
        }

        return {"optimizer": opt, "lr_scheduler": scheduler_config}

    def obj_det_predict(self, dl, dev=False):
        """
        Generates predictions from self over the provided DataLoader dl.
        Lightning's Trainer.predict supposedly makes multi-gpu ddp easy, but
        that turned into a deep rabbit hole, set aside for now.

        This returns raw, unfiltered predictions; post-processing that was
        formerly in get_results is deferred to utils/postproc.py

        Also note this assumes that self is already on the desired device.
        IOW, the caller should move the model to whatever device they want
        before calling this func.
        """
        all_patch_results = []
        all_tiles_info = []
        empty_batches = 0
        total_batches = len(dl)

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dl):
                images = batch["image"].to(self.device)
                outputs = self.model(images)
                # note all_patch_results and all_tiles_info accumulate
                # results accross all batches for the entire dataloader
                was_batch_empty = self._collect_batch_predictions(
                    outputs,
                    batch,
                    all_patch_results,
                )
                empty_batches += was_batch_empty
                self._collect_batch_extents(batch, all_tiles_info)

        # since FCOS doesn't use roialign => it's exempt from this bug, so don't raise.
        # need to soften this check for dev runs, needs more thought
        if (
            (empty_batches >= total_batches - 1)
            and not isinstance(self.model, FCOS)
            and not dev
        ):
            raise RuntimeError(
                "Non-empty predictions for at most one batch!!! This \
                is almost certainly the bug reported at \
                https://github.com/pytorch/vision/issues/8206!"
            )

        all_preds = pd.DataFrame([], columns=BBOX_SCHEMA)
        if len(all_patch_results) > 0:
            all_preds = pd.concat((all_preds, *all_patch_results))
        # coerce to int instead of object for downstream sanity
        all_preds["afb_label"] = all_preds["afb_label"].astype(np.int64)
        logging.info(f"pred bboxes: {all_preds.shape[0]}")
        all_tiles_info = collate_tile_extents(all_tiles_info)

        return all_preds, all_tiles_info


def collate_tile_extents(list_of_tile_dicts):
    # see here https://stackoverflow.com/a/5946322, I got tricky &
    # confusing duplication errors until I switched to defaultdict
    all_tiles_info = defaultdict(list)
    for tiles_batch in list_of_tile_dicts:
        for item_id, extents in tiles_batch.items():
            all_tiles_info[item_id].extend(extents)
    return dict(all_tiles_info)
