import os
import cv2
import torch
import numpy as np
import re
import torch.utils.data as data
import json
from models.builder import logger
from utils.prompt_utils import (
    encode_prompts,
    build_prompt_groups_from_labels,
    unload_clip_model,
    split_into_sentences,
    select_caption_topk
)
def get_path(
        dataset_name,
        _rgb_path,
        _rgb_format,
        _x_path,
        _x_format,
        _gt_path,
        _gt_format,
        x_modal,
        item_name,
):
    if dataset_name == "StanFord2D3D":
        rgb_path = os.path.join(
            _rgb_path,
            item_name.replace(".jpg", "").replace(".png", "") + _rgb_format,
        )
        d_path = os.path.join(
            _x_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("/rgb/", "/depth/").replace("_rgb", "_newdepth")
            + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace(".jpg", "")
            .replace(".png", "")
            .replace("/rgb/", "/semantic/")
            .replace("_rgb", "_newsemantic")
            + _gt_format,
        )
    elif dataset_name == "new_stanford":
        area = item_name.split(" ")[0]
        name = item_name.split(" ")[1]
        # print(area,name)
        rgb_path = os.path.join(_rgb_path, area + "/image/" + name + _rgb_format)
        d_path = os.path.join(_x_path, area + "/hha/" + name + _x_format)
        gt_path = os.path.join(_gt_path, area + "/label/" + name + _gt_format)
    elif dataset_name == "KITTI-360":
        rgb_path = os.path.join(_rgb_path, item_name.split(" ")[0])
        d_path = os.path.join(
            _x_path,
            item_name.split(" ")[0]
            .replace("data_2d_raw", "data_3d_rangeview")
            .replace("image_00/data_rect", "velodyne_points/data"),
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.split(" ")[1].replace("data_2d_semantics", "data_2d_semantics_trainID"),
            # .replace("/train", ""),
        )
    elif dataset_name == "Scannet":
        rgb_path = os.path.join(
            _rgb_path,
            item_name.replace(".jpg", "").replace(".png", "") + _rgb_format,
        )
        d_path = os.path.join(
            _x_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("color", "convert_depth") + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("color", "convert_label") + _gt_format,
        )
    elif dataset_name == "MFNet":
        rgb_path = os.path.join(_rgb_path, item_name + _rgb_format)
        d_path = os.path.join(_x_path, item_name + _x_format)
        gt_path = os.path.join(_gt_path, item_name + _gt_format)
    elif dataset_name == "EventScape":
        item_name = item_name.split(".png")[0]
        img_name = item_name.split("/")[-1]
        img_id = img_name.replace("_image", "").split("_")[-1]
        rgb_path = os.path.join(_rgb_path, item_name + _rgb_format)
        d_path = os.path.join(
            _x_path,
            item_name.replace("rgb", "depth").replace("data", "frames").replace(img_name, img_id) + _x_format,
        )
        e_path = os.path.join(
            _x_path,
            item_name.replace("rgb", "events").replace("data", "frames").replace(img_name, img_id) + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace("rgb", "semantic").replace("image", "gt_labelIds") + _gt_format,
        )
    else:  # Default for NYUDepthv2 and others
        base_name = os.path.basename(item_name).split('.')[0]
        rgb_path = os.path.join(_rgb_path, base_name + _rgb_format)
        d_path = os.path.join(_x_path, base_name + _x_format)
        gt_path = os.path.join(_gt_path, base_name + _gt_format)

    path_result = {"rgb_path": rgb_path, "gt_path": gt_path,
                   "item_name": item_name}  # Return item_name for text feature lookup
    for modal in x_modal:
        path_result[modal + "_path"] = eval(modal + "_path")
    return path_result


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting["rgb_root"]
        self._rgb_format = setting["rgb_format"]
        self._gt_path = setting["gt_root"]
        self._gt_format = setting["gt_format"]
        self._transform_gt = setting["transform_gt"]
        self._x_path = setting["x_root"]
        self._x_format = setting["x_format"]
        self._train_source = setting["train_source"]
        self._eval_source = setting["eval_source"]
        self.class_names = setting["class_names"]
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.dataset_name = setting["dataset_name"]
        self.x_modal = setting.get("x_modal", ["d"])
        self.backbone = setting["backbone"]

        self.enable_text_guidance = setting.get("enable_text_guidance", False)
        self.text_feature_dim = setting.get("text_feature_dim", 512)
        self.label_txt_path = setting.get("label_txt_path")
        self.caption_json_path = setting.get("caption_json_path")
        self.max_caption_sentences = int(setting.get("max_caption_sentences", 6))
        self.max_caption_sentences = max(self.max_caption_sentences, 0)
        self.text_template_set = setting.get("text_template_set", "clip")
        self.max_templates_per_label = int(setting.get("max_templates_per_label", 3))
        self.text_feature_dim = setting.get("text_feature_dim", 512)
        self.text_template_set = setting.get("text_template_set", "clip")
        self.max_templates_per_label = int(setting.get("max_templates_per_label", 3))
        self.text_source = setting.get("text_source", "both")
        self.text_encoder = setting.get("text_encoder", "jinaclip")
        self.text_encoder_name = setting.get("text_encoder_name", None)
        self.max_caption_sentences = int(setting.get("max_caption_sentences", 8))
        self.caption_topk = int(setting.get("caption_topk", 0))  # 0 表示不开句级Top-K
        self.caption_topk_mode = setting.get("caption_topk_mode", "class_sim")
        # 预备：类原型（给 class_sim Top-K 用）
        self.labels_txt_path = setting.get("label_txt_path", None)

        from utils.prompt_utils import build_prompt_groups_from_labels, encode_prompts

        if self.labels_txt_path is not None:
            labels = [ln.strip() for ln in open(self.labels_txt_path, 'r', encoding='utf-8').read().splitlines() if ln.strip()]
            groups = build_prompt_groups_from_labels(labels, L=len(labels),
                                                     template_set=self.text_template_set,
                                                     max_templates_per_label=self.max_templates_per_label)
            self.classbank = encode_prompts(groups,
                                            encoder=self.text_encoder,
                                            encoder_name=self.text_encoder_name,
                                            target_dim=self.text_feature_dim).cpu()  # [C,D]
        else:
            self.classbank = None

        self.class_text_features = None
        self.caption_text_features = {}
        # 统一句子通道长度（便于 DataLoader 默认collate）：K = caption_topk(>0) 否则用 max_caption_sentences
        self._cap_tokens = self.caption_topk if self.caption_topk > 0 else self.max_caption_sentences
        self.caption_fallback = torch.zeros(self._cap_tokens, self.text_feature_dim, dtype=torch.float32)

        if self.enable_text_guidance:
            self._prepare_text_guidance_assets()

    def _prepare_text_guidance_assets(self):
        """Load and encode textual assets for guidance."""
        try:
            self.class_text_features = self.classbank if getattr(self, "classbank", None) is not None else self._encode_class_labels()
        except Exception as exc:
            logger.error(f"Failed to encode class label prompts: {exc}")
            self.class_text_features = None

        try:
            self.caption_text_features = self._encode_caption_descriptions()
        except Exception as exc:
            logger.error(f"Failed to encode caption descriptions: {exc}")
            self.caption_text_features = {}
        finally:
            unload_clip_model()

        if self.class_text_features is None:
            self.class_text_features = torch.zeros(0, self.text_feature_dim, dtype=torch.float32)

    def _encode_class_labels(self):
        labels = []
        if self.label_txt_path and os.path.exists(self.label_txt_path):
            with open(self.label_txt_path, "r", encoding="utf-8") as f:
                labels = [ln.strip() for ln in f if ln.strip()]
        elif self.class_names:
            labels = list(self.class_names)

        if not labels:
            logger.warning("No class labels found for text guidance; class embeddings will be empty.")
            return None

        prompt_groups = build_prompt_groups_from_labels(
            labels,
            L=len(labels),
            template_set=self.text_template_set,
            max_templates_per_label=self.max_templates_per_label,
        )
        text_feats = encode_prompts(
            prompt_groups,
            encoder = self.text_encoder,
            encoder_name = self.text_encoder_name,
            target_dim = self.force_text_dim,
        ).cpu().to(torch.float32)
        return text_feats

    def _encode_caption_descriptions(self, rgb_path: str, description_text: str):
        caption_dict = {}
        if not self.caption_json_path or not os.path.exists(self.caption_json_path):
            logger.warning("Caption JSON path not provided or not found; only class prompts will be used.")
            return caption_dict

        with open(self.caption_json_path, "r", encoding="utf-8") as f:
            try:
                caption_entries = json.load(f)
            except json.JSONDecodeError as exc:
                logger.error(f"Failed to parse caption json {self.caption_json_path}: {exc}")
                return caption_dict

        for entry in caption_entries:
            rgb_rel = entry.get("rgb_path")
            description = entry.get("description", "")
            if not rgb_rel or not description:
                continue
            key = os.path.basename(rgb_rel)
            # 句级 Top-K 预筛（或全量切句）

            if self.caption_topk > 0:
                chosen, feats = select_caption_topk(
                    description,
                    K = self.caption_topk,
                    mode = self.caption_topk_mode,
                    labels_txt_path = self.label_txt_path,
                    template_set = self.text_template_set,
                    max_templates_per_label = self.max_templates_per_label,
                    encoder = self.text_encoder,
                    encoder_name = self.text_encoder_name,
                    target_dim = self.text_feature_dim,
                )
                text_feats = feats
            else:
                sents = self._split_description(description)

                if len(sents) == 0:
                    continue

                text_feats = encode_prompts(
                    sents,
                    encoder = self.text_encoder,
                    encoder_name = self.text_encoder_name,
                    target_dim = self.text_feature_dim,
                ).cpu().to(torch.float32)

                # 对齐长度到 K（_cap_tokens）

                if text_feats.shape[0] < self._cap_tokens:
                    pad = torch.zeros(self._cap_tokens - text_feats.shape[0], text_feats.shape[1],
                                      dtype=torch.float32)
                    text_feats = torch.cat([text_feats, pad], dim=0)
                elif text_feats.shape[0] > self._cap_tokens:
                    text_feats = text_feats[: self._cap_tokens]
                caption_dict[key] = text_feats

        return caption_dict

    def _split_description(self, description: str):
        description = description.replace("\n", " ").strip()
        if not description:
            return []
        sentences = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", description) if seg.strip()]
        if not sentences:
            sentences = [description]
        if self.max_caption_sentences > 0:
            sentences = sentences[: self.max_caption_sentences]
        return sentences

    def _get_text_features_for_item(self, rgb_path: str, item_name: str):
        if not self.enable_text_guidance:
            return torch.zeros(1, self.text_feature_dim, dtype=torch.float32)

        class_feats = self.class_text_features
        if class_feats is None:
            class_feats = torch.zeros(0, self.text_feature_dim, dtype=torch.float32)

        caption_feats = None
        key_candidates = []
        if rgb_path:
            key_candidates.append(os.path.basename(rgb_path))
        if item_name:
            key_candidates.append(os.path.basename(item_name))

        for key in key_candidates:
            if key in self.caption_text_features:
                caption_feats = self.caption_text_features[key]
                break

        if caption_feats is None:
            caption_feats = self.caption_fallback.clone()

        if self.text_source == "labels":
            return class_feats
        elif self.text_source == "captions":
            return caption_feats
        else:  # both
            return torch.cat([class_feats, caption_feats], dim=0)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        path_dict = get_path(
            self.dataset_name,
            self._rgb_path,
            self._rgb_format,
            self._x_path,
            self._x_format,
            self._gt_path,
            self._gt_format,
            self.x_modal,
            item_name,
        )

        rgb_mode = "BGR"
        rgb = self._open_image(path_dict["rgb_path"], rgb_mode)

        gt = self._open_image(path_dict["gt_path"], cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if self._transform_gt:
            gt = self._gt_transform(gt)

        x = {}
        for modal in self.x_modal:
            if modal == "d":
                x_img = self._open_image(path_dict[modal + "_path"], cv2.IMREAD_GRAYSCALE)
                if x_img is not None and len(x_img.shape) == 2:
                    x[modal] = cv2.merge([x_img, x_img, x_img])
                else:  # Handle cases where depth image is not found or has wrong format
                    x[modal] = np.zeros_like(rgb, dtype=np.uint8)
            else:
                x[modal] = self._open_image(path_dict[modal + "_path"], "RGB")

        if len(self.x_modal) == 1:
            x = x[self.x_modal[0]]

        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        x = torch.from_numpy(np.ascontiguousarray(x)).float()

        output_dict = dict(
            data=rgb,
            label=gt,
            modal_x=x,
            fn=str(path_dict["rgb_path"]),
            n=len(self._file_names),
        )

        text_feats = self._get_text_features_for_item(path_dict.get("rgb_path"), path_dict.get("item_name", ""))
        output_dict['text_features'] = text_feats

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ["train", "val"]
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[: length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        try:
            if mode == "RGB":
                img = np.array(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), dtype=dtype)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif mode == "BGR":
                img = np.array(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), dtype=dtype)
            else:
                img = np.array(cv2.imread(filepath, mode), dtype=dtype)
            return img
        except Exception as e:
            logger.error(f"Error opening image {filepath}: {e}")
            return None

    @staticmethod
    def _gt_transform(gt):
        return gt - 1

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors