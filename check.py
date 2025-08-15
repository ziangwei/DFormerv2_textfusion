import torch

ckpt = torch.load("checkpoints/NYUDepthv2_DFormerv2_S_20250724-062229/epoch-627_miou_55.95.pth", map_location="cpu")
state_dict = ckpt["model"] if "model" in ckpt else ckpt

for k in state_dict.keys():
    if "semantic_attn" in k:
        print(k)
