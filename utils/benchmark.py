import argparse
from models.builder import EncoderDecoder as segmodel
import numpy as np
import torch
import torch.nn as nn
from importlib import import_module
from thop import profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path")
    args = parser.parse_args()
    # config network and criterion
    config = getattr(import_module(args.config), "C")
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
    BatchNorm2d = nn.BatchNorm2d
    model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    device = torch.device("cuda:0")
    model.eval()
    model.to(device)
    dump_input = torch.ones(1, 3, 480, 640).to(device)

    src = getattr(config, "text_source", "both")
    cap_k = getattr(config, "caption_topk", 0)
    cap_max = getattr(config, "max_caption_sentences", 0)
    cap_tokens = cap_k if (isinstance(cap_k, int) and cap_k > 0) else cap_max

    if src == "labels":
               text_tokens = config.num_classes
    elif src == "captions":
        text_tokens = cap_tokens
    else:
        text_tokens = config.num_classes + cap_tokens

    text_dim = getattr(config, "text_feature_dim", 512)
    if getattr(config, "enable_text_guidance", False):
        dummy_text = torch.zeros(1, text_tokens, text_dim, device=device)
        inputs = (dump_input, dump_input, None, dummy_text)
    else:
        inputs = (dump_input, dump_input)
    flops, params = profile(model, inputs=inputs)
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))
