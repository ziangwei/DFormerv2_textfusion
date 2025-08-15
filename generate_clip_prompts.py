import os
import torch
import clip
from PIL import Image
import json
from tqdm import tqdm
import numpy as np

# 加载CLIP模型（可用cuda）
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# NYUdepth2数据集RGB路径
rgb_folder_path = "datasets/NYUDepthv2/RGB"

# 类别列表，可以用NYU数据集的已有类别或自行设定
class_list = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
    'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refrigerator', 'television',
    'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet',
    'sink', 'lamp', 'bathtub', 'bag',  'otherstructure',  'otherfurniture',  'otherprop'
]

# 一次性文本编码类别列表
text_tokens = clip.tokenize(class_list).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 存储结果的字典
label_prompts = {}

# 遍历文件夹中的所有图片
for img_name in tqdm(os.listdir(rgb_folder_path)):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(rgb_folder_path, img_name)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # 将图片分割为8个patch（2x4）
        cols, rows = 4, 2
        patch_width, patch_height = W // cols, H // rows
        total_scores = np.zeros(len(class_list))

        # 处理每个patch
        for i in range(rows):
            for j in range(cols):
                left = j * patch_width
                upper = i * patch_height
                right = (j + 1) * patch_width if j != cols - 1 else W
                lower = (i + 1) * patch_height if i != rows - 1 else H

                patch = img.crop((left, upper, right, lower))
                patch_tensor = preprocess(patch).unsqueeze(0).to(device)

                # 提取patch特征并计算相似度
                with torch.no_grad():
                    image_features = model.encode_image(patch_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()

                    # 取出当前patch前5个类别的概率并累加
                    top5_indices = similarity.argsort()[-5:]
                    for idx in top5_indices:
                        total_scores[idx] += similarity[idx]

        # 汇总所有patch结果后取整体前5
        final_top5_indices = total_scores.argsort()[-5:][::-1]
        final_top5_labels = [class_list[idx] for idx in final_top5_indices]

        # 存入字典
        label_prompts[img_name] = final_top5_labels

# 输出到JSON文件
output_json_path = "nyu_clip_patch8_top5_labels.json"
with open(output_json_path, "w") as f:
    json.dump(label_prompts, f, indent=2)

print(f"所有图片的类别prompt已生成并保存至 {output_json_path}")

