# utils/text_embed_manager.py

import os
import torch
import torch.distributed as dist
from utils.engine.logger import get_logger

# (--- 关键 ---)
# 我们从 prompt_utils 导入真正执行编码的函数
# 这样 prompt_utils.py 自己保持原样，完全不被修改
from utils.prompt_utils import (
    load_text_encoder,
    build_class_embeddings,
    build_img_embeddings,
    unload_clip_model
)


def get_or_create_text_embeddings(config):
    """
    自动缓存文本嵌入，DDP安全。
    - 检查缓存是否存在。
    - 如果不存在，仅 Rank 0 计算、保存。
    - 其他 Rank 等待。
    - 最终所有 Rank 从缓存加载。
    """
    logger = get_logger()

    # 1. 检查是否启用了文本功能
    if not getattr(config, 'enable_text_guidance', False):
        logger.info("文本引导未启用，跳过嵌入加载。")
        return None

    # 2. 确定唯一的缓存路径
    # 缓存将基于数据集、文本来源和编码器名称
    cache_dir = "checkpoints/text_cache"
    text_source = getattr(config, 'text_source', 'none')
    text_encoder_name = getattr(config, 'text_encoder', 'none')  # (例如 'clip' 或 'jinaclip')

    cache_key = f"{config.dataset_name}_{text_source}_{text_encoder_name}"

    # (可选) 您还可以添加更多配置项到key中
    # cache_key += f"_{getattr(config, 'text_encoder_name', 'default')}"

    cache_file = f"{cache_key}.pth"
    cache_path = os.path.join(cache_dir, cache_file)

    # 3. DDP 同步逻辑
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()

    if rank == 0:
        # 只有 Rank 0 负责检查和创建
        os.makedirs(cache_dir, exist_ok=True)

        if not os.path.exists(cache_path):
            logger.info(f"Rank 0: 未找到缓存 {cache_path}")
            logger.info("Rank 0: 正在启动文本编码器 (这在首次运行时可能需要几分钟到几十分钟)...")

            try:
                # 只在 Rank 0 上加载和使用文本编码器
                text_encoder_model = load_text_encoder(config)

                # 根据 config 调用从 prompt_utils 导入的函数
                # 这对应您 train.py 中第 320 行的逻辑
                if getattr(config, "text_source", None) == "labels":
                    text_embeddings_dict = build_class_embeddings(config, text_encoder_model)
                else:
                    # 这对应您 train.py 中第 323 行的逻辑
                    text_embeddings_dict = build_img_embeddings(config, text_encoder_model)

                logger.info(f"Rank 0: 编码完成。正在保存到缓存: {cache_path}")
                torch.save(text_embeddings_dict, cache_path)
                logger.info(f"Rank 0: 保存缓存成功。")

                # 卸载模型以释放显存
                unload_clip_model()

            except Exception as e:
                logger.error(f"Rank 0: 文本编码失败: {e}")
                if os.path.exists(cache_path):
                    os.remove(cache_path)  # 删除可能的损坏文件
                raise e
        else:
            logger.info(f"Rank 0: 找到文本嵌入缓存: {cache_path}")

    # 4. 设置屏障 (Barrier)
    # Rank 1+ 会在这里等待，直到 Rank 0 完成上面的 'if' 块并保存了文件
    if dist.is_initialized():
        logger.info(f"Rank {rank}: 正在等待所有进程同步文本嵌入...")
        dist.barrier()

    # 5. 所有进程从缓存加载
    logger.info(f"Rank {rank}: 正在从缓存加载文本嵌入: {cache_path}")
    try:
        # map_location='cpu' 确保在DDP中加载安全
        text_embeddings_dict = torch.load(cache_path, map_location='cpu')
        logger.info(f"Rank {rank}: 加载缓存成功。")
        return text_embeddings_dict
    except Exception as e:
        logger.error(f"Rank {rank}: 从缓存 {cache_path} 加载失败! {e}")
        if rank == 0:
            logger.error(f"缓存文件可能已损坏。请手动删除它: 'rm {cache_path}' 然后重试。")
        raise e