from .._base_.datasets.NYUDepthv2 import *

""" Settings for network, this would be different for each kind of model"""
C.backbone = "DFormerv2_S"  # Remember change the path below.
C.pretrained_model = "checkpoints/pretrained/DFormerv2_Small_pretrained.pth"
# C.decoder = "ham"
C.decoder = "HSGHead"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

# ==============================
# Text Guidance (统一文本引导)
# ==============================
# 总开关
C.enable_text_guidance = True
# 统一文本向量维度（必须和 SAM 的 text_dim 一致；Jina-CLIP 默认 512，OpenAI CLIP ViT-B/16 也是 512）
C.text_feature_dim = 512
# 文本来源：labels / captions / both
C.text_source = "labels"
# 选择文本编码器：clip / jinaclip
C.text_encoder = "clip"
# 具体模型名（留空走默认：clip→openai/clip-vit-base-patch16；jinaclip→jinaai/jina-clip-v2）
C.text_encoder_name = None
# 标签与描述数据
C.label_txt_path = "datasets/NYUDepthv2/nyu40_labels.txt"
C.caption_json_path = "datasets/NYUDepthv2/generated_rgb_descriptions_internvl3.json"
# 模板与数量（对“类名→多模板短句”的扩写）
C.text_template_set = "clip"           # clip / none
C.max_templates_per_label = 3
# 描述最多切几句送进模型（上限，进一步可用 caption_topk 再筛）
C.max_caption_sentences = 8
# 单条描述的 Top-K 预筛（0 表示不开）
C.caption_topk = 6
# 句子 Top-K 选择策略：class_sim / firstk / lenk
# - class_sim：和全类原型库的最大相似度（推荐，轻量稳定）
# - firstk：按句序取前 K
# - lenk：按长度取前 K（启发式）
C.caption_topk_mode = "class_sim"


"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 16
C.nepochs = 600
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 0
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.25
C.aux_rate = 0.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]  # [0.75, 1, 1.25] #
C.eval_flip = True  # False #
C.eval_crop_size = [480, 640]  # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 250
C.checkpoint_step = 25

"""Path Config"""
C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone)
C.log_dir = C.log_dir + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()).replace(" ", "_")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(
    osp.join(C.log_dir, "checkpoint")
)  #'/mnt/sda/repos/2023_RGBX/pretrained/'#osp.abspath(osp.join(C.log_dir, "checkpoint"))
if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir, exist_ok=True)
exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"

