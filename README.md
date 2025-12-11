---
base_model: Qwen/Qwen2.5-Math-1.5B
library_name: peft
---

# DeepSeek Distillation for Qwen Models

This repository provides a full pipeline to distill DeepSeek models into smaller Qwen models. It includes data preparation, question generation, answer collection, dataset merging, LoRA-based distillation, evaluation, and model comparison.

---

## 📦 Data Preparation

### **1. Prepare distillation dataset**

Use `prepare_openr1_dataset.py` to generate `openr1_math_220k_distill.jsonl` from the OpenR1 dataset.

* Produces a JSONL file compatible with all training scripts.

---

## 🔍 Testing the Model

To quickly evaluate your fine-tuned or distilled model, use:

```
python test.py
```

This script loads the model specified in the configuration and runs generation tests on sample prompts.

Includes:

* Loading model and tokenizer
* Running test prompts
* Displaying generated outputs

Use this as the fastest way to sanity-check your trained weights before running full evaluation.

---

## 🔥 Distillation / Training

### **Main training script**

```
train_distill.py
```

Includes:

* LoRA configuration
* Tokenization logic
* `DistillationTrainer`
* Training arguments

Example configuration:

```
conf/train_config.yaml
```

### Alternative SFT-style training

Use:

```
deepseek_distill_qwen.py
```

This performs supervised fine-tuning using the `format_instruction` template.

---

## 🧪 Evaluation & Model Comparison

### **Evaluate a fine‑tuned model**

Run:

```
python test_qwen_model.py
```

Generates responses for test prompts.

### **Compare models**

Run:

```
python compare_qwen_model.py
```

Metrics supported:

* Throughput
* Perplexity
* BLEU / ROUGE
* Memory usage
* Model size

---

## 📝 Notes & Tips

### Tokenizers

Some tokenizers do not define a padding token.

* Scripts automatically set `pad_token = eos_token` when missing.

### Device Handling

Training often loads teacher & student on CPU first, then moves the student to GPU.
Check `train_distill.py` if modifying device placement.

### Model Layout

Saved models, adapters, and checkpoints appear in:

* `distill-qwen7b/`
* `qwen2.5-math-7b-distilled/`
* `model/`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests when appropriate (see `test_qwen_model.py`, `test.py`)
4. Open a Pull Request with a clear description

---

## 📄 License

This project is licensed under **Apache License 2.0**.
See `LICENSE` for details.

---

##  Contact / Acknowledgments

* Author metadata is embedded in scripts (e.g., `deepseek_distill_qwen.py`).
* Thanks to the **DeepSeek** and **Qwen** teams for providing base models and datasets.
