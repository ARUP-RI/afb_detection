import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor,
    BackboneWithFPN,
)
from torchvision.models.detection.faster_rcnn import FasterRCNN as _FasterRCNN

from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection import FCOS, fcos_resnet50_fpn
from torch import nn

from afb.models.base import BaseLightningModule
from afb.config import BOX_IOU_THRESH, NMS_IOU_THRESH
from afb.data.image_annotation import AFBLabel

from torchvision.models import convnext_base, ConvNeXt_Base_Weights


def get_resnetrcnn(num_classes):
    # Define new anchor sizes and aspect ratios
    # You can customize these values based on your requirements
    # PE NOTE, the network is happy with five defined anchor sizes, there's
    # some other parameter related to the feature maps that would let us use
    # fewer, but I don't know what it is.
    new_anchor_sizes = (
        (4,),
        (8,),
        (16,),
        (32,),
        (64,),
    )
    new_aspect_ratios = ((0.5, 1.0, 2.0, 4.0, 8.0),) * len(new_anchor_sizes)

    # Create a new AnchorGenerator with custom anchor sizes and aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=new_anchor_sizes,
        aspect_ratios=new_aspect_ratios,
    )
    print("Anchors", anchor_generator.num_anchors_per_location())

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2,
    )

    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, norm_layer=nn.BatchNorm2d)
    resnet_fpn = _resnet_fpn_extractor(resnet, 5)

    model = _FasterRCNN(
        resnet_fpn,
        num_classes=num_classes,
        min_size=256,
        max_size=256,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        # rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        # rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        # rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        # rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        # rpn_nms_thresh=0.4,
        rpn_nms_thresh=0.7,  # default
        # rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
        #    considered as positive during training of the RPN.
        # rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
        #    considered as negative during training of the RPN.
        # rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
        #    for computing the loss
        # rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
        #    of the RPN
        # rpn_score_thresh (float): during inference, only return proposals with a classification score
        #    greater than rpn_score_thresh
        # box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
        #    the locations indicated by the bounding boxes
        # box_head (nn.Module): module that takes the cropped feature maps as input
        # box_predictor (nn.Module): module that takes the output of box_head and returns the
        #    classification logits and box regression deltas.
        # box_score_thresh (float): during inference, only return proposals with a classification score
        #    greater than box_score_thresh
        # box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        # box_nms_thresh=0.3,
        # box_nms_thresh=0.5, # default
        box_nms_thresh=NMS_IOU_THRESH,
        # box_detections_per_img (int): maximum number of detections per image, for all classes.
        # box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
        #    considered as positive during training of the classification head
        box_fg_iou_thresh=BOX_IOU_THRESH,
        # box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
        #    considered as negative during training of the classification head
        box_bg_iou_thresh=BOX_IOU_THRESH,
        # box_batch_size_per_image (int): number of proposals that are sampled during training of the
        #    classification head
        # box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
        #    of the classification head
        # bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
        #    bounding boxes
    )

    # OSX w/ corporate proxy:
    # ```
    # wget http://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    # mv fasterrcnn_resnet50_fpn_coco-258fb6c6.pth ~/.cache/torch/hub/checkpoints/
    # ```
    # URL can be found at: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    #    #weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
    # )
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(
    #    in_features,
    #    num_classes
    # )

    ## Freeze all the layers by default
    # for param in model.parameters():
    #    param.requires_grad = False
    ## Unfreeze the last block (layer4) of the ResNet-50 backbone
    # for param in model.backbone.body.layer4.parameters():
    #    param.requires_grad = True
    ## Unfreeze the feature pyramid network
    # for param in model.backbone.fpn.parameters():
    #    param.requires_grad = True
    ## Unfreeze the RPN and RoI heads
    # for param in model.rpn.parameters():
    #    param.requires_grad = True
    # for param in model.roi_heads.parameters():
    #    param.requires_grad = True

    return model


def get_convnextrcnn(num_classes):
    # For reasons I don't understand we can't use 5 anchor sizes here, but 4 works fine
    new_anchor_sizes = ((4,), (8,), (16,), (32,))
    new_aspect_ratios = ((0.5, 1.0, 2.0, 4.0),) * len(new_anchor_sizes)

    # Create a new AnchorGenerator with custom anchor sizes and aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=new_anchor_sizes,
        aspect_ratios=new_aspect_ratios,
    )
    print("Anchors", anchor_generator.num_anchors_per_location())

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["p2"],
        output_size=7,
        sampling_ratio=2,
    )

    convnext = convnext_base(weights=ConvNeXt_Base_Weights)
    convnext_no_classifier = convnext.features
    convnext_no_classifier.out_channels = 1024
    convnext_fpn = BackboneWithFPN(
        backbone=convnext_no_classifier,
        return_layers={
            "1": "p2",  # 128
            "3": "p3",  # 256
            "5": "p4",  # 512
        },
        in_channels_list=[128, 256, 512],
        out_channels=256,
    )

    model = _FasterRCNN(
        convnext_fpn,
        num_classes=num_classes,
        min_size=256,
        max_size=256,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        # rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        # rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        # rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        # rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        # rpn_nms_thresh=0.4,
        rpn_nms_thresh=0.7,  # default
        # rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
        #    considered as positive during training of the RPN.
        # rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
        #    considered as negative during training of the RPN.
        # rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
        #    for computing the loss
        # rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
        #    of the RPN
        # rpn_score_thresh (float): during inference, only return proposals with a classification score
        #    greater than rpn_score_thresh
        # box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
        #    the locations indicated by the bounding boxes
        # box_head (nn.Module): module that takes the cropped feature maps as input
        # box_predictor (nn.Module): module that takes the output of box_head and returns the
        #    classification logits and box regression deltas.
        # box_score_thresh (float): during inference, only return proposals with a classification score
        #    greater than box_score_thresh
        # box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        # box_nms_thresh=0.3,
        # box_nms_thresh=0.5, # default
        box_nms_thresh=NMS_IOU_THRESH,
        # box_detections_per_img (int): maximum number of detections per image, for all classes.
        # box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
        #    considered as positive during training of the classification head
        box_fg_iou_thresh=BOX_IOU_THRESH,
        # box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
        #    considered as negative during training of the classification head
        box_bg_iou_thresh=BOX_IOU_THRESH,
        # box_batch_size_per_image (int): number of proposals that are sampled during training of the
        #    classification head
        # box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
        #    of the classification head
        # bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
        #    bounding boxes
    )

    # To freeze backbone do this
    # convnext_no_classifier.requires_grad_(False)

    return model


def get_fcos_convnextbase(num_classes):
    backbone = torchvision.models.convnext_base(
        weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    )
    bb_w_fpn = BackboneWithFPN(
        backbone=backbone.features,
        return_layers={
            "1": "p2",  # 128
            "3": "p3",  # 256
            "5": "p4",  # 512
        },
        in_channels_list=[128, 256, 512],
        out_channels=256,
    )

    # not really an anchor generator, FCOS just repurposed the code for it
    # I get errors if len of anchor_sizes is anything other than 4, not sure why
    anchor_sizes = (
        (4,),
        (8,),
        (16,),
        (32,),
    )  # equal to strides of multi-level feature map??
    aspect_ratios = ((1.0,),) * len(anchor_sizes)  # set only one "anchor" for FCOS
    anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)

    model = FCOS(
        bb_w_fpn,
        num_classes=num_classes,
        min_size=256,
        max_size=256,
        nms_thresh=NMS_IOU_THRESH,
        anchor_generator=anchor_gen,
    )
    return model


def get_fcos_resnet50(num_classes):
    model = fcos_resnet50_fpn(
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        num_classes=num_classes,
        trainable_backbone_layers=5,  # default to all layers trainable
        min_size=256,
        max_size=256,
    )
    return model


class TorchVizObjDet(BaseLightningModule):
    def __init__(
        self,
        learning_rate=0.0001,
        weight_decay=0.0001,
        betas=(0.99, 0.999),
        fscore_betasq=0.8,
        val_dataset=None,
        class_names=AFBLabel.get_class_names(),
        model_name="convnext_rcnn",
        lr_warmup_ratio=1 / 30,
        lr_warmup_iters=1000,
        lr_decay_iters=10000,
        freeze_backbone=False,
        weights=None,
        wts_conf=None,
        device=None,
        optimizer=None,
        out_dir=None,
    ):
        super().__init__(val_dataset, fscore_betasq, optimizer, out_dir)

        if weights is None:
            num_classes = len(class_names)
        else:
            # we can lookup num_classes from provided trained weights,
            # unneeded for training but more convenient for inference
            state_dict = torch.load(weights, map_location="cpu")
            if (
                model_name == "resnet_rcnn"
                or model_name == "faster_rcnn"
                or model_name == "convnext_rcnn"
            ):
                num_classes = len(state_dict["roi_heads.box_predictor.cls_score.bias"])
            elif model_name == "convnext_fcos" or model_name == "fcos_resnet50":
                num_classes = len(
                    state_dict["head.classification_head.cls_logits.bias"]
                )
            else:
                raise ValueError(f"Unknown model name `{model_name}`")
            # Thanks to AFBLabel enum, we need not check for strict equivalence, but classes
            # in trained model better be a subst of AFBLabel enum or the enum needs updating.
            # I'm not asserting strict equality here b/c it would break backwards
            # compat w/ old trained models, & we'd need to update the unit
            # testing model weights every time the enum changes
            assert num_classes <= len(
                class_names
            ), f"Trained model's state dict contains \
                more classes than user-supplied class_names: {num_classes} vs {class_names}!"
        self.tgt_cls_idx = AFBLabel.AFB.index()

        if model_name == "resnet_rcnn" or model_name == "faster_rcnn":
            self.model = get_resnetrcnn(num_classes)
        elif model_name == "convnext_rcnn":
            self.model = get_convnextrcnn(num_classes)
        elif model_name == "convnext_fcos":
            self.model = get_fcos_convnextbase(num_classes)
        elif model_name == "fcos_resnet50":
            self.model = get_fcos_resnet50(num_classes)
        else:
            raise ValueError(f"Unknown model name `{model_name}`")
        self.to(device)

        if weights is not None:
            self.model.load_state_dict(state_dict)

        if freeze_backbone:
            print("Freezing model backbone 🧊🧊🧊")
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        self.val_dataset = val_dataset

        self.save_hyperparameters(
            "learning_rate",
            "weight_decay",
            "betas",
            "lr_warmup_iters",
            "lr_decay_iters",
            "lr_warmup_ratio",
        )
