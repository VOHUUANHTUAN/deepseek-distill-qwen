import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq,
    TrainingArguments, 
    Trainer
)
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================
teacher_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
student_model_name = "Qwen/Qwen2.5-Math-1.5B"  

print("="*60)
print("LOADING MODELS...")
print("="*60)

# Load teacher model with device offloading to CPU
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu"  
)
teacher.eval()  # Set to eval mode
for param in teacher.parameters():
    param.requires_grad = False  # Freeze teacher

teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
if teacher_tokenizer.pad_token is None:
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

# Load student model on CPU first (avoid meta tensor issue with PEFT)
student = AutoModelForCausalLM.from_pretrained(
    student_model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu" 
)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token

print(f"✓ Teacher model loaded (on CPU): {teacher_model_name}")
print(f"✓ Student model loaded (on CPU): {student_model_name}")

# ==========================================
# 2. LOAD DATASET
# ==========================================
print("\n" + "="*60)
print("LOADING DATASET...")
print("="*60)

ds = load_dataset("json", data_files="openr1_math_220k_distill.jsonl", split="train")
print(f"✓ Dataset loaded. Size: {len(ds)}")
print(f"✓ Sample columns: {ds.column_names}")
print(f"✓ Sample data:\n{ds[0]}\n")

# ==========================================
# 3. FORMAT DATASET
# ==========================================
def format_sample(example):
    """Convert dataset format to prompt and label"""
    prompt = example["instruction"]  
    label = example["output"]       
    return {"prompt": prompt, "label": label}

ds = ds.map(format_sample, remove_columns=ds.column_names)
print(f"✓ Dataset formatted. New columns: {ds.column_names}")

# ==========================================
# 4. TOKENIZATION
# ==========================================
print("\n" + "="*60)
print("TOKENIZING DATASET...")
print("="*60)

def tokenize_fn(example):
    """Tokenize prompt + label concatenated"""
    # Combine prompt and label
    full_text = example["prompt"] + "\n" + example["label"]
    
    # Tokenize full text
    tokenized = student_tokenizer(
        full_text,
        truncation=True,
        max_length=2048,
        padding=False
    )
    
    input_ids = tokenized["input_ids"]
    
    # Tokenize just prompt to get boundary
    prompt_tokens = student_tokenizer(
        example["prompt"],
        truncation=True,
        max_length=2048,
        padding=False
    )
    prompt_length = len(prompt_tokens["input_ids"]) + 1  # +1 for newline token
    
    # Create labels: -100 for prompt tokens (ignore in loss), actual labels for output
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    
    # Pad labels to match input_ids length
    labels = labels + [-100] * (len(input_ids) - len(labels))
    
    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

ds = ds.map(
    tokenize_fn,
    remove_columns=ds.column_names,
    desc="Tokenizing dataset"
)
print(f"✓ Dataset tokenized. New columns: {ds.column_names}")

# ==========================================
# 5. CUSTOM DISTILLATION TRAINER
# ==========================================
class DistillationTrainer(Trainer):
    """Custom trainer for knowledge distillation"""
    
    def __init__(self, teacher_model, temperature=4.0, alpha=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss (1-alpha for CE loss)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute KL divergence loss between student and teacher"""
        
        # Get student outputs (without computing loss inside model)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            output_hidden_states=False,
        )
        student_logits = outputs.logits
        
        # Get teacher outputs (no gradient)
        with torch.no_grad():
            # Move teacher to same device as student for inference
            teacher_device = next(model.parameters()).device
            teacher_input_ids = inputs["input_ids"].to("cpu")
            teacher_attention_mask = inputs.get("attention_mask", None)
            if teacher_attention_mask is not None:
                teacher_attention_mask = teacher_attention_mask.to("cpu")
            
            teacher_outputs = self.teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask
            )
            teacher_logits = teacher_outputs.logits.to(teacher_device)
        
        # Match sequence length first
        min_seq_len = min(student_logits.shape[1], teacher_logits.shape[1])
        student_logits = student_logits[:, :min_seq_len, :]
        teacher_logits = teacher_logits[:, :min_seq_len, :]
        
        # Match vocab size - truncate teacher to student vocab
        student_vocab = student_logits.shape[2]
        teacher_vocab = teacher_logits.shape[2]
        
        if student_vocab != teacher_vocab:
            # Use smaller vocab size
            min_vocab = min(student_vocab, teacher_vocab)
            student_logits = student_logits[:, :, :min_vocab]
            teacher_logits = teacher_logits[:, :, :min_vocab]
        
        # KL Divergence loss (Distillation) - MUST use student logits with requires_grad
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence: D_KL(teacher || student)
        kl_loss = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs), dim=-1)
        loss_kl = kl_loss.mean()
        
        # Ensure loss has requires_grad
        if not loss_kl.requires_grad:
            loss_kl = loss_kl.detach().requires_grad_(True)
        
        # Cross Entropy loss with labels (optional, can set alpha=1.0 for KL only)
        labels = inputs.get("labels", None)
        loss_ce = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        if labels is not None:
            # Shift for next token prediction
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten and compute CE loss
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            if ce_loss.requires_grad:
                loss_ce = ce_loss
        
        # Combined loss - make sure both have requires_grad
        loss = self.alpha * loss_kl + (1 - self.alpha) * loss_ce
        
        # Ensure loss requires grad before returning
        if not loss.requires_grad:
            loss.requires_grad_(True)
        
        return (loss, outputs) if return_outputs else loss


# ==========================================
# 6. LORA CONFIGURATION
# ==========================================
print("\n" + "="*60)
print("CONFIGURING LORA...")
print("="*60)

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
student = get_peft_model(student, lora_config)
student.print_trainable_parameters()

# Enable gradient for LoRA parameters
for param in student.parameters():
    if 'lora' in param.__class__.__name__.lower() or 'lora' in str(param.shape):
        param.requires_grad = True

# ==========================================
# 7. DATA COLLATOR
# ==========================================
data_collator = DataCollatorForSeq2Seq(
    student_tokenizer,
    model=student,
    label_pad_token_id=-100,
    padding=True
)
print("✓ Data collator configured")

# ==========================================
# 8. TRAINING ARGUMENTS
# ==========================================
training_args = TrainingArguments(
    output_dir="distill-qwen7b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=1e-4,
    bf16=True,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    gradient_checkpointing=False,
    optim="paged_adamw_8bit",
    ddp_find_unused_parameters=False,
    max_grad_norm=1.0,
    seed=42,
    tf32=True,
    max_steps=1000, 
)

# ==========================================
# 9. TRAINER & TRAINING
# ==========================================
print("\n" + "="*60)
print("STARTING TRAINING...")
print("="*60)

# Don't move teacher to GPU - keep on CPU to save memory
# Only student will be moved to GPU by Trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Student will be moved to GPU by Trainer automatically
# Teacher stays on CPU and will be moved to GPU only during inference

trainer = DistillationTrainer(
    model=student,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
    teacher_model=teacher,
    temperature=4.0,
    alpha=0.7,  # 70% KL loss + 30% CE loss
)

trainer.train()

# ==========================================
# 10. SAVE MODEL
# ==========================================
print("\n" + "="*60)
print("SAVING MODEL...")
print("="*60)

student.save_pretrained("qwen2.5-math-7b-distilled")
student_tokenizer.save_pretrained("qwen2.5-math-7b-distilled")

print("✓ Model saved to: qwen2.5-math-7b-distilled")
print("="*60)
print("TRAINING COMPLETED!")
print("="*60)