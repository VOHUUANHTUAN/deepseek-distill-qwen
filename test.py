# test_safe.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, traceback, sys

model_path = "qwen2.5-math-1.5b-distilled-v1"

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
    prompt = """
    Problem: Relativistic Particle Collision and Energy Conservation

    Two identical particles, each with rest mass m₀ = 1.67 × 10⁻²⁷ kg (proton mass),
    collide head-on in a particle accelerator:

    - Particle 1: kinetic energy K₁ = 5 GeV, moving in +x direction
    - Particle 2: kinetic energy K₂ = 3 GeV, moving in -x direction
    - Speed of light: c = 3 × 10⁸ m/s
    - Rest energy of proton: E₀ = 0.938 GeV

    Calculate:

    1. Total energy of each particle (E = γm₀c²)
    2. Lorentz factor (γ) for each particle
    3. Velocity of each particle
    4. Momentum of each particle
    5. Total momentum before collision
    6. Total energy before collision
    7. Invariant mass of the system (M_inv² = (E_total)² - (p_total·c)²)
    8. What is the center-of-mass energy available for particle creation?
    9. Could this collision create a particle with mass 4 GeV/c²?
    10. Compare this to the Newtonian prediction if particles moved at 0.9c
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(next(model.parameters()).device)

    torch.cuda.empty_cache()

    print("Generating (do_sample=True, max_new_tokens=64)...")
    out = model.generate(
        input_ids,
        max_new_tokens=4096,
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
    torch.cuda.synchronize()