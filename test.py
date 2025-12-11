# test_safe.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, traceback, sys

model_path = "qwen2.5-math-7b-distilled"

try:
    print("Loading tokenizer & model (bf16, device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,   # prefer bf16 on H200
        trust_remote_code=True
    )

    # small prompt, force truncation so we don't exceed model context
    prompt = """Solve step by step:\nWhat is larger, 9.9 or 9.11?"""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(next(model.parameters()).device)

    torch.cuda.empty_cache()

    print("Generating (do_sample=True, max_new_tokens=64)...")
    out = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,         # must set if using temperature
        temperature=0.2,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("=== OUTPUT ===")
    print(text)

except Exception as e:
    print("Exception during generation:", e)
    traceback.print_exc(file=sys.stdout)
    # Optional: force sync so cuda errors visible
    import os
    os.system("nvidia-smi -q | sed -n '1,120p'")
