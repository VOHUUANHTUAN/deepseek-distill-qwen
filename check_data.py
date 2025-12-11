import json

path = "openr1_math_220k_distill.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == 1: break
        sample = json.loads(line)
        print(f"\n--- Sample {i+1} ---")
        for k, v in sample.items():
            print(f"{k}: {v}")
