import os
import cv2
import torch
import numpy as np
import re
import torch.utils.data as data
import json
from pathlib import Path
import hashlib

from models.builder import logger
from utils.prompt_utils import (
    encode_prompts,
    build_prompt_groups_from_labels,
    encode_labels_batch,
    unload_clip_model,
    split_into_sentences,
    select_caption_topk
)

# === Project-level persistent text cache utilities ===
def _project_root() -> Path:
    """
    以当前文件所在目录为基准推断项目根。
    这里按你的仓库结构：.../DFormer/utils/dataloader/RGBXDataset.py -> 项目根是上上级的上级，即 parents[2]
    如有需要可改为 parents[3] 等。
    """
    return Path(__file__).resolve().parents[2]

def _stable_cache_root(custom_root: str | None = None) -> Path:
    """
    缓存根目录优先级：
      1) 调用方传入的 custom_root
      2) 环境变量 TEXT_EMBED_CACHE
      3) <项目根>/datasets/.text_cache   <-- 默认放在 datasets/ 下（按你的要求）
    支持相对路径：相对路径会以项目根为基准解析。
    """
    if custom_root:
        p = Path(custom_root)
        if not p.is_absolute():
            p = _project_root() / p
        root = p
    else:
        env = os.environ.get("TEXT_EMBED_CACHE", "")
        root = Path(env) if env else _project_root() / "datasets" / ".text_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root

def _sha1_of_file(p: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _text_cache_key(dataset_name: str,
                    text_encoder: str,
                    text_encoder_name: str | None,
                    text_feature_dim: int,
                    text_template_set: str,
                    max_templates_per_label: int,
                    caption_topk: int,
                    caption_topk_mode: str,
                    max_image_labels: int,
                    src_labels_txt: str | None = None,
                    src_captions_json: str | None = None,
                    src_imglabels_json: str | None = None) -> str:
    """
    用“编码器/维度/模板配置 + 文本源内容哈希”生成稳定 key，保证跨多次训练命中同一缓存。
    """
    parts = [f"ds={dataset_name}", f"enc={text_encoder}", f"encn={text_encoder_name or 'none'}",
             f"D={int(text_feature_dim)}", f"tmpl={text_template_set}", f"mtpl={int(max_templates_per_label)}",
             f"capk={int(caption_topk)}", f"capm={caption_topk_mode or 'none'}", f"maximgk={int(max_image_labels)}",
             f"labels={_sha1_of_file(src_labels_txt)}" if src_labels_txt and os.path.isfile(
                 src_labels_txt) else "labels=none",
             f"caps={_sha1_of_file(src_captions_json)}" if src_captions_json and os.path.isfile(
                 src_captions_json) else "caps=none",
             f"imglab={_sha1_of_file(src_imglabels_json)}" if src_imglabels_json and os.path.isfile(
                 src_imglabels_json) else "imglab=none"]
    return "__".join(parts)
# === end utilities ===


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
        self.text_cache_root = setting.get('text_cache_root', None)  # 支持覆盖缓存根目录
        self.max_caption_sentences = int(setting.get("max_caption_sentences", 8))
        self.caption_topk = int(setting.get("caption_topk", 0))  # 0 表示不开句级Top-K
        self.caption_topk_mode = setting.get("caption_topk_mode", "class_sim")
        # 每图最多保留多少个标签（0 表示不限，回退到数据中的最大长度）
        self.max_image_labels = int(setting.get("max_image_labels", 0))
        # 预备：类原型（给 class_sim Top-K 用）
        self.labels_txt_path = setting.get("label_txt_path", None)
        self.image_labels_json_path = setting.get("image_labels_json_path", None)

        # === 文本源开关 ===
        source = (self.text_source or "both").lower()
        self._use_label_text = source in ("labels", "both")
        self._use_caption_text = source in ("captions", "both")
        self._use_imglabel_text = source == "imglabels"
        # caption Top-K 需要 classbank 做相似度
        self._need_classbank = self._use_label_text or (self._use_caption_text and self.caption_topk > 0)

        self.classbank = None
        if self._need_classbank:
            try:
                self.classbank = self._encode_class_labels()
            except Exception as exc:
                logger.error(f"Failed to precompute classbank: {exc}")
                self.classbank = None

        self.class_text_features = None
        self.caption_text_features = {}
        self.imglabel_text_features = {}  # {basename: Tensor[Ki, D]}
        self.imglabel_text_names = {}
        self._imglabel_tokens = None      # 对齐长度（=全数据最大Ki；不做Top-K）
        # 统一句子通道长度（K = caption_topk(>0) 否则用 max_caption_sentences）
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
        built_sources = []

        # --- 类别文本（labels） ---
        if self._use_label_text:
            try:
                if self.classbank is not None:
                    self.class_text_features = self.classbank
                else:
                    self.class_text_features = self._encode_class_labels()
                if self.class_text_features is not None and self.class_text_features.numel() > 0:
                    built_sources.append("labels")
            except Exception as exc:
                logger.error(f"Failed to encode class label prompts: {exc}")
                self.class_text_features = None
        else:
            self.class_text_features = None

        # --- Caption 文本 ---
        caption_model_used = False
        if self._use_caption_text:
            try:
                self.caption_text_features = self._encode_caption_descriptions()
                caption_model_used = True
                if isinstance(self.caption_text_features, dict) and len(self.caption_text_features) > 0:
                    built_sources.append("captions")
            except Exception as exc:
                logger.error(f"Failed to encode caption descriptions: {exc}")
                self.caption_text_features = {}
            finally:
                if caption_model_used:
                    unload_clip_model()
        else:
            self.caption_text_features = {}

        # --- 每图标签文本（image labels） ---
        imglabel_model_used = False
        if self._use_imglabel_text:
            try:
                self.imglabel_text_features = self._encode_image_labels()
                imglabel_model_used = True
                if isinstance(self.imglabel_text_features, dict) and len(self.imglabel_text_features) > 0:
                    built_sources.append("imglabels")
            except Exception as exc:
                logger.error(f"Failed to encode image-level label texts: {exc}")
                self.imglabel_text_features = {}
            finally:
                if imglabel_model_used:
                    unload_clip_model()
        else:
            self.imglabel_text_features = {}
            self._imglabel_tokens = 0
            self.imglabel_text_names = {}

        if self.class_text_features is None:
            self.class_text_features = torch.zeros(0, self.text_feature_dim, dtype=torch.float32)

        if built_sources:
            logger.info(f"[Text guidance] built caches for: {', '.join(built_sources)}")
        else:
            logger.info("[Text guidance] no text caches were constructed for current source setting.")

    # ---------------- 按图的标签文本（imglabels） ----------------
    def _encode_image_labels(self):
        """
        读取 image_labels_json（key 为相对路径或文件名，value 为该图的标签文本列表），
        对每张图的标签进行编码，可选地截断到固定的 max_image_labels，
        再将特征 pad/截断到统一长度后缓存到项目级缓存根（datasets/.text_cache）下。
        """
        out = {}
        name_map = {}
        self.imglabel_text_names = {}

        if (not self.image_labels_json_path) or (not os.path.exists(self.image_labels_json_path)):
            return out

        cache_root = _stable_cache_root(getattr(self, "text_cache_root", None))
        ckey = _text_cache_key(
            dataset_name=self.dataset_name,
            text_encoder=self.text_encoder,
            text_encoder_name=self.text_encoder_name,
            text_feature_dim=self.text_feature_dim,
            text_template_set=self.text_template_set,
            max_templates_per_label=self.max_templates_per_label,
            caption_topk=self.caption_topk,
            caption_topk_mode=self.caption_topk_mode,
            max_image_labels=self.max_image_labels,
            src_labels_txt=self.labels_txt_path,
            src_captions_json=None,
            src_imglabels_json=self.image_labels_json_path,
        )
        cache_dir = (cache_root / ckey / "imglabels")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_pt = cache_dir / "embeds.pt"

        # 命中缓存：直接载入
        if os.path.isfile(cache_pt):
            try:
                payload = torch.load(cache_pt, map_location="cpu")
                if not isinstance(payload, dict):
                    raise ValueError("legacy cache without metadata")
                self._imglabel_tokens = payload.get("pad_len", payload.get("K", None))
                feats = payload.get("feats", {})
                names = payload.get("names", {})
                if feats and isinstance(names, dict):
                    self.imglabel_text_names = names
                    return feats
                raise ValueError("image-label cache missing feats or names")
            except Exception as e:
                logger.warning(f"Load image-label cache failed: {e}; will recompute.")
                if self.is_master():
                    try:
                        cache_pt.unlink()
                    except OSError:
                        pass
        # 读取 JSON
        with open(self.image_labels_json_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)  # {path_or_basename: [str,...]}

        # 统计 pad 长度（limit>0 时固定为 limit）
        limit = max(int(self.max_image_labels or 0), 0)
        max_len = 0
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
                    clean.append(s); seen.add(s)

            if not clean:
                continue
            if limit > 0:
                clean = clean[:limit]  # 保持原顺序，截断
            standardized[k] = clean
            if limit == 0:
                max_len = max(max_len, len(clean))

        pad_len = limit if limit > 0 else max_len
        if pad_len == 0:
            self._imglabel_tokens = 0
            return out

        # ===== 优化：标签级去重批量编码 =====
        try:
            # 步骤1：收集所有唯一标签
            all_labels = []
            for labels in standardized.values():
                all_labels.extend(labels)

            # 步骤2：批量编码所有唯一标签
            unique_count = len(set(all_labels))
            logger.info(f"[Image labels] Batch encoding {unique_count} unique labels from {len(standardized)} images (optimized)...")
            label_embeds = encode_labels_batch(
                labels=all_labels,
                template_set=self.text_template_set,
                max_templates_per_label=self.max_templates_per_label,
                encoder=self.text_encoder,
                encoder_name=self.text_encoder_name,
                target_dim=self.text_feature_dim,
                batch_size=512,
            )  # {label_name: tensor[D]}

            # 步骤3：为每张图组装特征（从标签缓存查表）
            for key, labels in standardized.items():
                # 查表获取每个标签的embedding
                img_feats = []
                for lb in labels:
                    lb_norm = lb.lower()
                    if lb_norm in label_embeds:
                        img_feats.append(label_embeds[lb_norm])
                    else:
                        # 降级：未找到则使用零向量
                        logger.warning(f"Label '{lb}' not found in batch-encoded labels, using zero vector")
                        img_feats.append(torch.zeros(self.text_feature_dim, dtype=torch.float32))

                if len(img_feats) > 0:
                    feats = torch.stack(img_feats, dim=0)  # [num_labels, D]
                else:
                    feats = torch.zeros(0, self.text_feature_dim, dtype=torch.float32)

                # Pad/截断到统一长度
                if feats.shape[0] < pad_len:
                    pad = torch.zeros(pad_len - feats.shape[0], feats.shape[1], dtype=torch.float32)
                    feats = torch.cat([feats, pad], dim=0)
                elif feats.shape[0] > pad_len:
                    feats = feats[:pad_len]

                out[key] = feats
                name_map[key] = list(labels)
                base = os.path.basename(key)
                if base not in out:
                    out[base] = feats  # 同时支持 basename 命中
                    name_map[base] = name_map[key]

            logger.info(f"[Image labels] Batch encoding completed successfully")

        except Exception as e:
            # 回退到旧版逐图编码（向后兼容）
            logger.warning(f"[Image labels] Batch encoding failed ({e}), falling back to per-image encoding...")
            for key, labels in standardized.items():
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
                name_map[key] = list(labels)
                base = os.path.basename(key)
                if base not in out:
                    out[base] = feats
                    name_map[base] = name_map[key]
            logger.info(f"[Image labels] Fallback encoding completed")

        self._imglabel_tokens = pad_len
        self.imglabel_text_names = name_map

        # 主进程写缓存
        if self.is_master():
            try:
                torch.save({"pad_len": pad_len, "feats": out, "names": name_map}, cache_pt)
            except Exception as exc:
                logger.warning(f"Saving image label cache failed: {exc}")

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

        # 缓存：项目级根（datasets/.text_cache）+ 内容哈希 key
        cache_root = _stable_cache_root(getattr(self, "text_cache_root", None))
        ckey = _text_cache_key(
            dataset_name=self.dataset_name,
            text_encoder=self.text_encoder,
            text_encoder_name=self.text_encoder_name,
            text_feature_dim=self.text_feature_dim,
            text_template_set=self.text_template_set,
            max_templates_per_label=self.max_templates_per_label,
            caption_topk=self.caption_topk,
            caption_topk_mode=self.caption_topk_mode,
            max_image_labels=self.max_image_labels,
            src_labels_txt=self.labels_txt_path,
            src_captions_json=self.caption_json_path,
            src_imglabels_json=None,
        )
        cache_dir = (cache_root / ckey / "captions")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_pt = cache_dir / "embeds.pt"

        # 命中缓存即返回
        if os.path.isfile(cache_pt):
            try:
                payload = torch.load(cache_pt, map_location="cpu")
                return payload["feats"] if isinstance(payload, dict) and "feats" in payload else payload
            except Exception as e:
                logger.warning(f"Load caption cache failed: {e}; will recompute.")

        # 读 JSON
        try:
            with open(self.caption_json_path, "r", encoding="utf-8") as f:
                caption_entries = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse caption json {self.caption_json_path}: {exc}")
            return caption_dict

        # 预取 classbank（给 Top-K 用）
        classbank = None
        if self.caption_topk > 0:
            classbank = getattr(self, "classbank", None)
            if classbank is None and self.label_txt_path and os.path.exists(self.label_txt_path):
                labels = [ln.strip() for ln in open(self.label_txt_path, "r", encoding="utf-8").read().splitlines() if ln.strip()]
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

        # 主循环：切句 -> 编码 -> 对齐长度
        for entry in caption_entries:
            rgb_rel = entry.get("rgb_path")
            description = (entry.get("description") or "").strip()
            if not rgb_rel or not description:
                continue

            key = os.path.basename(rgb_rel)

            if self.caption_topk > 0 and classbank is not None:
                sents = self._split_description(description)
                if len(sents) == 0:
                    caption_dict[key] = self.caption_fallback.clone()
                    continue

                sent_feats = encode_prompts(
                    sents,
                    encoder=self.text_encoder,
                    encoder_name=self.text_encoder_name,
                    target_dim=self.text_feature_dim,
                ).cpu()
                scores = (sent_feats @ classbank.T).amax(dim=1)  # [N]
                topk = min(self.caption_topk, len(sents))
                idx = torch.topk(scores, k=topk, dim=0).indices
                feats = sent_feats[idx].cpu().to(torch.float32)

                if feats.shape[0] < self._cap_tokens:
                    pad = torch.zeros(self._cap_tokens - feats.shape[0], feats.shape[1], dtype=torch.float32)
                    feats = torch.cat([feats, pad], dim=0)
                elif feats.shape[0] > self._cap_tokens:
                    feats = feats[: self._cap_tokens]

                caption_dict[key] = feats
            else:
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

        # 写缓存
        try:
            if self.is_master():
                torch.save({"feats": caption_dict}, cache_pt)
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
        empty_meta = {"names": [], "types": []}
        if not self.enable_text_guidance:
            return torch.zeros(1, self.text_feature_dim, dtype=torch.float32), empty_meta

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
            return class_feats, empty_meta
        elif self.text_source == "captions":
            return caption_feats, empty_meta
        elif self.text_source == "imglabels":
            # 优先用 rgb_path 的 basename 匹配
            key_candidates = []
            if rgb_path:
                key_candidates.extend([rgb_path, os.path.basename(rgb_path)])
            if item_name:
                key_candidates.extend([item_name, os.path.basename(item_name)])
            img_feats = None
            names = []
            for k in key_candidates:
                if not k:
                    continue
                key = os.path.basename(k) if k not in self.imglabel_text_features else k
                if key in self.imglabel_text_features:
                    img_feats = self.imglabel_text_features[key]
                    names = list(
                        self.imglabel_text_names.get(key) or self.imglabel_text_names.get(os.path.basename(key), []))
                    break
                if k in self.imglabel_text_features:
                    img_feats = self.imglabel_text_features[k]
                    names = list(
                        self.imglabel_text_names.get(k) or self.imglabel_text_names.get(os.path.basename(k), []))
                    break

            if img_feats is None:
                pad_len = int(self._imglabel_tokens or 0)
                if pad_len > 0:
                    img_feats = torch.zeros(pad_len, self.text_feature_dim, dtype=torch.float32)
                else:
                    img_feats = torch.zeros(0, self.text_feature_dim, dtype=torch.float32)
                names = []
            meta = {"names": names, "types": ["imglabel"] * len(names)}
            return img_feats, meta
        else:  # both
            return torch.cat([class_feats, caption_feats], dim=0), empty_meta

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
                else:
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

        text_feats, text_meta = self._get_text_features_for_item(path_dict.get("rgb_path"), path_dict.get("item_name", ""))
        output_dict['text_features'] = text_feats
        output_dict['text_token_meta'] = json.dumps(text_meta, ensure_ascii=False)

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
    def _open_image(filepath: str, mode: "int | str" = cv2.IMREAD_COLOR, dtype=None):
        """
        mode: 可以是 OpenCV 读旗标(int)，或 "RGB"/"BGR" 两种字符串。
        """
        try:
            if mode == "RGB":
                img = np.array(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), dtype=dtype)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif mode == "BGR":
                img = np.array(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), dtype=dtype)
            else:
                # 其余情况按 OpenCV 旗标处理（int）
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