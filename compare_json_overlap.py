# compare_json_overlap.py
import json, sys, re

def extract_id(name: str) -> str:
    m = re.search(r"(\d+)", name)
    return m.group(1) if m else None

def load_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, labels in raw.items():
        iid = extract_id(k)
        if iid is None:
            continue
        # 简单去空格+小写；不做同义词映射
        out[iid] = set(str(x).strip().lower() for x in labels if isinstance(x, str))
    return out

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_json_overlap.py file1.json file2.json")
        sys.exit(1)

    m1 = load_map(sys.argv[1])
    m2 = load_map(sys.argv[2])

    ids = sorted(set(m1.keys()) & set(m2.keys()), key=lambda x: int(x))
    if not ids:
        print("No overlapping image IDs found.")
        return

    total = 0
    for iid in ids:
        overlap = len(m1[iid] & m2[iid])
        total += overlap
        print(f"{iid}\t{overlap}")  # 形如: 941\t3

    print(f"pairs={len(ids)}\taverage_overlap={total/len(ids):.3f}")

if __name__ == "__main__":
    main()
