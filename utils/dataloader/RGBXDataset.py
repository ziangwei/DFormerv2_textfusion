import os
import cv2
import torch
import numpy as np
import re
import torch.utils.data as data
import json
from models.builder import logger


# from utils.prompt_utils import load_scene_list, sample_prompt, PROMPT_EMBEDS

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

        # =======================================================================================
        # 核心改动 1: 初始化文本特征相关的属性
        # =======================================================================================
        self.text_features_path = setting.get("text_features_path", None)
        self.text_mode = setting.get("text_mode", "class")  # "class" or "caption"
        self.text_features = self._load_text_features()

    def _load_text_features(self):
        """Loads text features from a file."""
        if self.text_features_path and os.path.exists(self.text_features_path):
            logger.info(f"Loading text features from {self.text_features_path}")
            try:
                # Assuming the features are stored in a .pt file
                features = torch.load(self.text_features_path, map_location='cpu')
                logger.info(f"Text features loaded successfully. Mode: {self.text_mode}")
                return features
            except Exception as e:
                logger.error(f"Failed to load text features: {e}")
                return None
        logger.warning("No text features path provided or file not found. Running without text guidance.")
        return None

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

        # =======================================================================================
        # 核心改动 2: 根据模式获取并添加文本特征到返回字典中
        # =======================================================================================
        if self.text_features is not None:
            if self.text_mode == 'class':
                # For class mode, all samples get the same set of class embeddings
                output_dict['text_features'] = self.text_features
            elif self.text_mode == 'caption':
                # For caption mode, look up the feature for the specific image
                # The key in the saved dictionary should match 'item_name' from the txt file
                img_key = os.path.basename(path_dict["item_name"])
                if img_key in self.text_features:
                    output_dict['text_features'] = self.text_features[img_key].unsqueeze(0)  # Shape -> (1, C_text)
                else:
                    # Fallback if a caption is not found for an image
                    logger.warning(f"Caption for {img_key} not found in text features file. Using zero tensor.")
                    # Assuming text features is a dict of tensors, get dim from a sample
                    sample_feat = next(iter(self.text_features.values()))
                    output_dict['text_features'] = torch.zeros(1, sample_feat.shape[-1])
            else:
                raise ValueError(f"Unknown text_mode: {self.text_mode}")
        else:
            # Provide a dummy tensor if text features are not loaded, to prevent crashes
            output_dict['text_features'] = torch.zeros(1)  # A dummy tensor

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