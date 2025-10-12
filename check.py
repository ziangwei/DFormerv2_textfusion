from transformers import CLIPModel
CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch16",
    use_safetensors=True,
    force_download=True,
    local_files_only=False,
)
print("OK")