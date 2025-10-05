import torch.nn as nn
from utils.pyt_utils import load_model
from importlib import import_module
from collections import OrderedDict
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry("models", parent=MMCV_MODELS)
BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        cfg.setdefault("train_cfg", train_cfg)
        cfg.setdefault("test_cfg", test_cfg)
    return SEGMENTORS.build(cfg)


def build_model(cfg, **kwargs):
    # cfg.model.type = "EncoderDecoderText"
    # cfg.model.backbone.type = "dformerv2"
    # cfg.model.decode_head.type = "HierarchicalSemanticGuidedHead"

    model_name = cfg.model.get("type", "EncoderDecoder")
    if model_name == "EncoderDecoder":
        # from .segmentors.encoder_decoder import EncoderDecoder as segmodel
        EncoderDecoder = import_module("models.segmentors.encoder_decoder").EncoderDecoder
        model = EncoderDecoder(
            cfg.model.backbone,
            cfg.model.decode_head,
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg"),
            **kwargs,
        )
    elif model_name == "EncoderDecoderText":
        # from .segmentors.encoder_decoder_text import EncoderDecoderText
        EncoderDecoderText = import_module("models.segmentors.encoder_decoder_text").EncoderDecoderText
        model = EncoderDecoderText(
            cfg.model.backbone,
            cfg.model.decode_head,
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg"),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    if cfg.model.get("pretrained"):
        state_dict = torch.load(cfg.model.get("pretrained"))
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                new_state_dict[k[9:]] = v
            else:
                new_state_dict[k] = v
        model.backbone.load_state_dict(new_state_dict, strict=False)

    return model