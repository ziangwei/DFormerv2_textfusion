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
        # 每图最多保留多少个标签（0 表示不限，回退到数据中的最大长度）
        self.max_image_labels = int(setting.get("max_image_labels", 0))
        # 预备：类原型（给 class_sim Top-K 用）
        self.labels_txt_path = setting.get("label_txt_path", None)
        self.image_labels_json_path = setting.get("image_labels_json_path", None)

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
        self.imglabel_text_features = {}  # 新增：{basename: Tensor[Ki, D]}
        self._imglabel_tokens = None  # 新增：对齐长度（=全数据最大Ki；不做Top-K）
        # 统一句子通道长度（便于 DataLoader 默认collate）：K = caption_topk(>0) 否则用 max_caption_sentences
        self._cap_tokens = self.caption_topk if self.caption_topk > 0 else self.max_caption_sentences
        self.caption_fallback = torch.zeros(self._cap_tokens, self.text_feature_dim, dtype=torch.float32)

        if self.enable_text_guidance:
            self._prepare_text_guidance_assets()

    def is_master(self) -> bool:
        """是否为主进程（rank 0）。在未初始化分布式时也返回 True。"""
        try:
            import torch.distributed as dist
            if (not dist.is_available()) or (not dist.is_initialized()):
                return True
            return dist.get_rank() == 0
        except Exception:
            return True

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

        # 新增：每图标签文本

        try:
            self.imglabel_text_features = self._encode_image_labels()
        except Exception as exc:
            logger.error(f"Failed to encode image-level label texts: {exc}")
            self.imglabel_text_features = {}
        finally:
            unload_clip_model()

        if self.class_text_features is None:
            self.class_text_features = torch.zeros(0, self.text_feature_dim, dtype=torch.float32)

    # ---------------- 新增：按图的标签文本 ----------------

    def _encode_image_labels(self):
        """
        读取 image_labels_json（key 为相对路径或文件名，value 为该图的标签文本列表），
        对每张图的标签进行编码，可选地截断到固定的 max_image_labels，
        再将特征 pad/截断到统一长度后缓存到与 JSON 同目录的 .text_cache 下。
        """

        out = {}

        if (not self.image_labels_json_path) or (not os.path.exists(self.image_labels_json_path)):
            return out

        cache_dir = os.path.join(os.path.dirname(self.image_labels_json_path), ".text_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = (
            f"imglabels_{os.path.splitext(os.path.basename(self.image_labels_json_path))[0]}_"
            f"dim{self.text_feature_dim}_tmpl{self.max_templates_per_label}_"
            f"set{self.text_template_set}_maxk{self.max_image_labels}.pt"
        )
        cache_pt = os.path.join(cache_dir, cache_name)

        # 非主进程尝试直接读缓存

        if (not self.is_master()) and os.path.isfile(cache_pt):
            try:
                payload = torch.load(cache_pt, map_location="cpu")
                self._imglabel_tokens = payload.get("pad_len", None)
                return payload.get("feats", {})
            except Exception:
                pass

        # 读取 JSON

        with open(self.image_labels_json_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)  # {path_or_basename: [str,...]}
        # 根据设置统计 pad 长度（limit>0 时固定为 limit）
        limit = max(int(self.max_image_labels or 0), 0)
        max_len = 0
        # 先收集标准化后的列表
        standardized = {}

        for k, lab_list in mapping.items():
            if not isinstance(lab_list, list) or len(lab_list) == 0:
                continue
            # 去重保序 + 小写规整
            seen, clean = set(), []
            for lb in lab_list:
                s = self._extract_image_label_text(lb)
                if not s:
                    continue
                s = s.lower()
                if s not in seen:
                    clean.append(s.lower()); seen.add(s)

            if not clean:
                continue
            if limit > 0:
                # 保持 JSON 中既有的相似度排序，仅截断到前 limit 个
                clean = clean[:limit]
            standardized[k] = clean
            if limit == 0:
                max_len = max(max_len, len(clean))

        pad_len = limit if limit > 0 else max_len

        if pad_len == 0:
            self._imglabel_tokens = 0
            return out

        # 编码并对齐长度（pad 到 pad_len）
        for key, labels in standardized.items():
            # 复用“每个 label 一个模板组 → 组内平均”得到 [K,D]
            groups = build_prompt_groups_from_labels(
                labels,
                L=len(labels),
                template_set=self.text_template_set,
                max_templates_per_label=self.max_templates_per_label,
            )
            feats = encode_prompts(
                groups,
                encoder=self.text_encoder,
                encoder_name=self.text_encoder_name,
                target_dim=self.text_feature_dim,
            ).cpu().to(torch.float32)  # [Ki, D]
            if feats.shape[0] < pad_len:
                pad = torch.zeros(pad_len - feats.shape[0], feats.shape[1], dtype=torch.float32)
                feats = torch.cat([feats, pad], dim=0)
            elif feats.shape[0] > pad_len:
                feats = feats[:pad_len]
            out[key] = feats
            base = os.path.basename(key)
            if base not in out:
                out[base] = feats

        self._imglabel_tokens = pad_len

        # 主进程写缓存，其它进程 barrier 后读取

        try:
            if self.is_master():
                torch.save({"pad_len": pad_len, "feats": out}, cache_pt)
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                except Exception:
                    pass
        except Exception:
            pass

        return out

    def _extract_image_label_text(self, entry):
        """标准化 image-level 标签条目，兼容多种 JSON 格式。"""

        if entry is None:
            return ""

        if isinstance(entry, str):
            return entry.strip()

        if isinstance(entry, dict):
            for key in ("label", "name", "text", "category", "class", "cls", "cls_name"):
                val = entry.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            # 兜底：如果字典里只有一个值，也尝试取出
            for val in entry.values():
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return ""

        if isinstance(entry, (list, tuple)) and entry:
            first = entry[0]
            if isinstance(first, str):
                return first.strip()
            if isinstance(first, dict):
                return self._extract_image_label_text(first)

        return str(entry).strip()

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
            encoder=self.text_encoder,
            encoder_name=self.text_encoder_name,
            target_dim=self.text_feature_dim,
        ).cpu().to(torch.float32)
        return text_feats


    def _encode_caption_descriptions(self):
        """
        将 caption JSON 统一编码成 {<rgb文件名>: Tensor[K, D]}。
        - rank0 负责计算并写入磁盘缓存；
        - 其他 rank 若检测到缓存，则直接加载；
        - Top-K 模式下按 classbank 做句级选择；否则按句子截断/填充到固定 K。
        """
        caption_dict = {}

        # 基本检查
        if not self.caption_json_path or not os.path.exists(self.caption_json_path):
            logger.warning("Caption JSON path not provided or not found; only class prompts will be used.")
            return caption_dict

        # 缓存文件位置：和 JSON 放一起的 .text_cache 目录
        cache_dir = os.path.join(os.path.dirname(self.caption_json_path), ".text_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = f"captions_{os.path.splitext(os.path.basename(self.caption_json_path))[0]}_" \
                     f"dim{self.text_feature_dim}_topk{self.caption_topk}_{self.caption_topk_mode}.pt"
        cache_pt = os.path.join(cache_dir, cache_name)

        # 非 rank0：若命中缓存，直接读返回
        if (not self.is_master()) and os.path.isfile(cache_pt):
            try:
                return torch.load(cache_pt, map_location="cpu")
            except Exception as e:
                logger.warning(f"Load caption cache failed on non-master: {e}; fallback to recompute on this rank.")

        # 读 JSON
        try:
            with open(self.caption_json_path, "r", encoding="utf-8") as f:
                caption_entries = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse caption json {self.caption_json_path}: {exc}")
            return caption_dict

        # 预取 classbank（给 Top-K 用），构造一个空张量占位
        classbank = None
        if self.caption_topk > 0:
            # 优先用在 __init__ 里已构建好的 self.classbank
            classbank = getattr(self, "classbank", None)
            if classbank is None and self.label_txt_path and os.path.exists(self.label_txt_path):
                labels = [ln.strip() for ln in open(self.label_txt_path, "r", encoding="utf-8").read().splitlines() if
                          ln.strip()]
                groups = build_prompt_groups_from_labels(
                    labels, L=len(labels),
                    template_set=self.text_template_set,
                    max_templates_per_label=self.max_templates_per_label,
                )
                classbank = encode_prompts(
                    groups,
                    encoder=self.text_encoder,
                    encoder_name=self.text_encoder_name,
                    target_dim=self.text_feature_dim,
                ).cpu()  # [C, D]

        # 主循环（逐条样本编码；_load_encoder 在 prompt_utils 里有全局缓存，不会重复建模）
        for entry in caption_entries:
            rgb_rel = entry.get("rgb_path")
            description = (entry.get("description") or "").strip()
            if not rgb_rel or not description:
                continue

            key = os.path.basename(rgb_rel)

            if self.caption_topk > 0 and classbank is not None:
                # 句级 Top-K：先切句 -> 编码每句 -> 与 classbank 做相似度 -> 选 K 句
                sents = self._split_description(description)
                if len(sents) == 0:
                    # 回退到全 0，占位
                    caption_dict[key] = self.caption_fallback.clone()
                    continue

                sent_feats = encode_prompts(
                    sents,
                    encoder=self.text_encoder,
                    encoder_name=self.text_encoder_name,
                    target_dim=self.text_feature_dim,
                )  # [N, D], on CPU by default
                scores = (sent_feats @ classbank.T).amax(dim=1)  # [N]
                topk = min(self.caption_topk, len(sents))
                idx = torch.topk(scores, k=topk, dim=0).indices
                feats = sent_feats[idx].cpu()

                # 对齐到固定 K（_cap_tokens==caption_topk），必要时填充/截断
                if feats.shape[0] < self._cap_tokens:
                    pad = torch.zeros(self._cap_tokens - feats.shape[0], feats.shape[1], dtype=torch.float32)
                    feats = torch.cat([feats, pad], dim=0)
                elif feats.shape[0] > self._cap_tokens:
                    feats = feats[: self._cap_tokens]

                caption_dict[key] = feats.to(torch.float32)
            else:
                # 非 Top-K：直接切句并编码，再对齐长度
                sents = self._split_description(description)
                if len(sents) == 0:
                    caption_dict[key] = self.caption_fallback.clone()
                    continue

                text_feats = encode_prompts(
                    sents,
                    encoder=self.text_encoder,
                    encoder_name=self.text_encoder_name,
                    target_dim=self.text_feature_dim,
                ).cpu().to(torch.float32)

                if text_feats.shape[0] < self._cap_tokens:
                    pad = torch.zeros(self._cap_tokens - text_feats.shape[0], text_feats.shape[1], dtype=torch.float32)
                    text_feats = torch.cat([text_feats, pad], dim=0)
                elif text_feats.shape[0] > self._cap_tokens:
                    text_feats = text_feats[: self._cap_tokens]

                caption_dict[key] = text_feats

        # rank0 写缓存，其他 rank 读（加一个 barrier 更稳）
        try:
            if self.is_master():
                torch.save(caption_dict, cache_pt)
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Save caption cache failed: {e}")

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
        elif self.text_source == "imglabels":
            # 优先用 rgb_path 的 basename 匹配
            key_candidates = []
            if rgb_path:
                key_candidates.extend([rgb_path, os.path.basename(rgb_path)])
            if item_name:
                key_candidates.extend([item_name, os.path.basename(item_name)])
            img_feats = None
            for k in key_candidates:
                if not k:
                    continue
                key = os.path.basename(k) if k not in self.imglabel_text_features else k
                if key in self.imglabel_text_features:
                    img_feats = self.imglabel_text_features[key]
                    break
                if k in self.imglabel_text_features:
                    img_feats = self.imglabel_text_features[k]; break

            if img_feats is None:
                # 未命中：用全零占位，长度按 _imglabel_tokens
                pad_len = int(self._imglabel_tokens or 0)
                if pad_len > 0:
                    img_feats = torch.zeros(pad_len, self.text_feature_dim, dtype=torch.float32)
                else:
                    img_feats = torch.zeros(0, self.text_feature_dim, dtype=torch.float32)
            return img_feats
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