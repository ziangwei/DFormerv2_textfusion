import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init_func import init_weight
from utils.engine.logger import get_logger

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        cfg=None,
        criterion=nn.CrossEntropyLoss(reduction="none", ignore_index=255),
        norm_layer=nn.BatchNorm2d,
        syncbn=False,
    ):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer
        self.cfg = cfg
        self.enable_text_guidance = getattr(cfg, "enable_text_guidance", False)

        if cfg.backbone == "DFormer-Large":
            from .encoders.DFormer import DFormer_Large as backbone
            self.channels = [96, 192, 288, 576]
        elif cfg.backbone == "DFormer-Base":
            from .encoders.DFormer import DFormer_Base as backbone
            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == "DFormer-Small":
            from .encoders.DFormer import DFormer_Small as backbone
            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == "DFormer-Tiny":
            from .encoders.DFormer import DFormer_Tiny as backbone
            self.channels = [32, 64, 128, 256]
        elif cfg.backbone == "DFormerv2_L":
            from .encoders.DFormerv2 import DFormerv2_L as backbone
            self.channels = [112, 224, 448, 640]
        elif cfg.backbone == "DFormerv2_B":
            from .encoders.DFormerv2 import DFormerv2_B as backbone
            self.channels = [80, 160, 320, 512]
        elif cfg.backbone == "DFormerv2_S":
            from .encoders.DFormerv2 import DFormerv2_S as backbone
            self.channels = [64, 128, 256, 512]
        else:
            raise NotImplementedError

        if syncbn:
            norm_cfg = dict(type="SyncBN", requires_grad=True)
        else:
            norm_cfg = dict(type="BN", requires_grad=True)

        backbone_kwargs = dict(norm_cfg=norm_cfg)
        backbone_kwargs["drop_path_rate"] = cfg.drop_path_rate if cfg.drop_path_rate is not None else 0.1

        if cfg.backbone.startswith("DFormerv2"):
            backbone_kwargs["text_dim"] = getattr(cfg, "text_feature_dim", 512)
            backbone_kwargs["sam_enc_stages"] = getattr(cfg, "sam_enc_stages", [0, 1, 2, 3])
            # 透传 SAM Top-K 策略
            backbone_kwargs["sam_use_topk"] = getattr(cfg, "sam_use_topk", True)
            backbone_kwargs["sam_top_m"] = getattr(cfg, "sam_top_m", 5)
            backbone_kwargs["superpower"] = getattr(cfg, "superpower", False)

        self.backbone = backbone(**backbone_kwargs)
        self.aux_head = None

        if cfg.decoder == "HSGHead":
            logger.info("Using Hierarchical Semantic-Guided Decoder")
            from .decoders.hsg_head import HierarchicalSemanticGuidedHead
            chs = list(self.channels)
            self.decode_head = HierarchicalSemanticGuidedHead(in_channels=chs[1:], in_index=[1, 2, 3], input_transform=(
                "multiple_select" if len(chs) > 1 else None), channels=getattr(cfg, "decoder_embed_dim", 512),
                                                              num_classes=cfg.num_classes, norm_cfg=norm_cfg,
                                                              text_dim=getattr(cfg, "text_feature_dim", 512),
                                                              sam_dec_stages=getattr(cfg, "sam_dec_stages",
                                                                                     [0, 1, 2, 3]),
                                                              sam_use_topk=getattr(cfg, "sam_use_topk", True),
                                                              sam_top_m=getattr(cfg, "sam_top_m", 5),
                                                              backbone_num_heads=getattr(self.backbone, "num_heads",
                                                                                         [4, 4, 8, 16]))
            if cfg.aux_rate != 0:
                from .decoders.fcnhead import FCNHead
                self.aux_index = 2
                self.aux_rate = cfg.aux_rate
                self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == "MLPDecoder":
            logger.info("Using MLP Decoder")
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(
                in_channels=self.channels,
                num_classes=cfg.num_classes,
                norm_layer=norm_layer,
                embed_dim=cfg.decoder_embed_dim,
            )

        elif cfg.decoder == "ham":
            logger.info("Using Ham Decoder")
            from .decoders.ham_head import LightHamHead as DecoderHead
            self.decode_head = DecoderHead(
                in_channels=self.channels[1:],
                num_classes=cfg.num_classes,
                in_index=[1, 2, 3],
                norm_cfg=norm_cfg,
                channels=cfg.decoder_embed_dim,
            )
            from .decoders.fcnhead import FCNHead
            if cfg.aux_rate != 0:
                self.aux_index = 2
                self.aux_rate = cfg.aux_rate
                self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == "UPernet":
            logger.info("Using Upernet Decoder")
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(
                in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512
            )
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == "deeplabv3+":
            logger.info("Using Decoder: DeepLabV3+")
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == "nl":
            logger.info("Using Decoder: nl+")
            from .decoders.nl_head import NLHead as Head
            self.decode_head = Head(
                in_channels=self.channels[1:], in_index=[1, 2, 3], num_classes=cfg.num_classes,
                norm_cfg=norm_cfg, channels=512
            )
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info("No decoder(FCN-32s)")
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(
                in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer
            )

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info("Loading pretrained model: {}".format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info("Initing weights ...")
        init_weight(
            self.decode_head,
            nn.init.kaiming_normal_,
            self.norm_layer,
            cfg.bn_eps,
            cfg.bn_momentum,
            mode="fan_in",
            nonlinearity="relu",
        )
        if self.aux_head:
            init_weight(
                self.aux_head,
                nn.init.kaiming_normal_,
                self.norm_layer,
                cfg.bn_eps,
                cfg.bn_momentum,
                mode="fan_in",
                nonlinearity="relu",
            )

    def encode_decode(self, rgb, modal_x, text_features=None):
        orisize = rgb.shape
        if self.enable_text_guidance:
            x = self.backbone(rgb, modal_x, text_features)
        else:
            x = self.backbone(rgb, modal_x)

        if len(x) == 2:
            x = x[0]
        if isinstance(x, (list, tuple)):
            feats = list(x)
            x = tuple(feats)

        if self.enable_text_guidance:
            out = self.decode_head.forward(x, text_features)
        else:
            out = self.decode_head.forward(x)

        out = F.interpolate(out, size=orisize[-2:], mode="bilinear", align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x=None, label=None, text_features=None):
        if self.enable_text_guidance and text_features is None:
            raise ValueError("text_features must be provided when enable_text_guidance is True")

        # Check for NaN/Inf in text_features
        if text_features is not None:
            if torch.isnan(text_features).any():
                logger.warning("NaN detected in text_features, replacing with zeros")
                text_features = torch.nan_to_num(text_features, nan=0.0)
            if torch.isinf(text_features).any():
                logger.warning("Inf detected in text_features, clamping values")
                text_features = torch.clamp(text_features, min=-1e6, max=1e6)

        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x, text_features)
        else:
            out = self.encode_decode(rgb, modal_x, text_features)

        if label is not None:
            # Create valid mask (non-background pixels)
            valid_mask = (label.long() != self.cfg.background)

            # Main loss
            main_loss_per_pixel = self.criterion(out, label.long())
            valid_main_loss = main_loss_per_pixel[valid_mask]

            # Check if we have valid pixels to compute loss
            if valid_main_loss.numel() > 0:
                loss = valid_main_loss.mean()
            else:
                # All pixels are background - return zero loss with gradient
                logger.warning("Batch has no valid pixels (all background), using zero loss")
                loss = main_loss_per_pixel.sum() * 0.0  # Returns 0 but maintains gradient graph

            # Auxiliary loss
            if self.aux_head:
                aux_loss_per_pixel = self.criterion(aux_fm, label.long())
                valid_aux_loss = aux_loss_per_pixel[valid_mask]

                if valid_aux_loss.numel() > 0:
                    loss += self.aux_rate * valid_aux_loss.mean()
                else:
                    loss += self.aux_rate * (aux_loss_per_pixel.sum() * 0.0)

            # Final NaN check
            if torch.isnan(loss):
                logger.error("NaN loss detected! Replacing with zero.")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

            return loss
        return out