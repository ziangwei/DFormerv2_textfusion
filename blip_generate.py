from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch, os, json
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")\
                 .to(device).eval()

    in_dir = "datasets/NYUDepthv2/RGB"
    out_json = "nyu_blip_captions.json"
    captions = {}

    for fn in tqdm(os.listdir(in_dir)):
        img = Image.open(os.path.join(in_dir, fn)).convert("RGB")
        inputs = proc(img, return_tensors="pt").to(device)
        out_ids = model.generate(**inputs, max_new_tokens=50)
        captions[fn] = proc.decode(out_ids[0], skip_special_tokens=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

