# finetune.py
import json
import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# ✅ Login to Hugging Face Hub (Run `huggingface-cli login` in terminal once before running this script)
# If you want to embed login inside script:
# login(token="YOUR_HF_TOKEN")

# ---------------- CONFIGURATION ---------------- #
config = {
    "hugging_face_username": "Shekswess",
    "model_config": {
        "base_model": "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "finetuned_model": "llama-3-8b-Instruct-bnb-4bit-medical",
        "max_seq_length": 2048,
        "dtype": torch.float16,
        "load_in_4bit": True,
    },
    "lora_config": {
        "r": 16,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": True,
        "use_rslora": False,
        "use_dora": False,
        "loftq_config": None
    },
    "training_dataset": {
        "name": "Shekswess/medical_llama3_instruct_dataset_short",
        "split": "train",
        "input_field": "prompt",
    },
    "training_config": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "max_steps": 0,
        "num_train_epochs": 1,
        "learning_rate": 2e-4,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 42,
        "output_dir": "outputs",
    }
}

# ---------------- LOAD MODEL ---------------- #
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["model_config"]["base_model"],
    max_seq_length=config["model_config"]["max_seq_length"],
    dtype=config["model_config"]["dtype"],
    load_in_4bit=config["model_config"]["load_in_4bit"],
)

# Apply LoRA/QLoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora_config"]["r"],
    target_modules=config["lora_config"]["target_modules"],
    lora_alpha=config["lora_config"]["lora_alpha"],
    lora_dropout=config["lora_config"]["lora_dropout"],
    bias=config["lora_config"]["bias"],
    use_gradient_checkpointing=config["lora_config"]["use_gradient_checkpointing"],
    random_state=42,
    use_rslora=config["lora_config"]["use_rslora"],
    use_dora=config["lora_config"]["use_dora"],
    loftq_config=config["lora_config"]["loftq_config"],
)

# ---------------- LOAD DATA ---------------- #
dataset_train = load_dataset(
    config["training_dataset"]["name"],
    split=config["training_dataset"]["split"]
)

# ---------------- TRAINER ---------------- #
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    dataset_text_field=config["training_dataset"]["input_field"],
    max_seq_length=config["model_config"]["max_seq_length"],
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=config["training_config"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training_config"]["gradient_accumulation_steps"],
        warmup_steps=config["training_config"]["warmup_steps"],
        max_steps=config["training_config"]["max_steps"],
        num_train_epochs=config["training_config"]["num_train_epochs"],
        learning_rate=config["training_config"]["learning_rate"],
        fp16=config["training_config"]["fp16"],
        bf16=config["training_config"]["bf16"],
        logging_steps=config["training_config"]["logging_steps"],
        optim=config["training_config"]["optim"],
        weight_decay=config["training_config"]["weight_decay"],
        lr_scheduler_type=config["training_config"]["lr_scheduler_type"],
        seed=config["training_config"]["seed"],
        output_dir=config["training_config"]["output_dir"],
    ),
)

# ---------------- MEMORY BEFORE ---------------- #
gpu_statistics = torch.cuda.get_device_properties(0)
reserved_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
max_memory = round(gpu_statistics.total_memory / 1024**3, 2)
print(f"Reserved Memory: {reserved_memory}GB | Max GPU Memory: {max_memory}GB")

# ---------------- TRAIN ---------------- #
trainer_stats = trainer.train()

# ---------------- MEMORY AFTER ---------------- #
used_memory = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
used_memory_lora = round(used_memory - reserved_memory, 2)
print(f"Used Memory: {used_memory}GB")
print(f"Used Memory for LoRA: {used_memory_lora}GB")

# Save stats
with open("trainer_stats.json", "w") as f:
    json.dump(trainer_stats, f, indent=4)

# ---------------- SAVE MODEL ---------------- #
model.save_pretrained(config["model_config"]["finetuned_model"])
model.push_to_hub(config["model_config"]["finetuned_model"], tokenizer=tokenizer)

# Optional merged formats
model.save_pretrained_merged(config["model_config"]["finetuned_model"], tokenizer, save_method="merged_16bit")
model.save_pretrained_merged(config["model_config"]["finetuned_model"], tokenizer, save_method="merged_4bit")
model.save_pretrained_gguf(config["model_config"]["finetuned_model"], tokenizer, quantization_method="f16")
model.save_pretrained_gguf(config["model_config"]["finetuned_model"], tokenizer, quantization_method="q4_k_m")

# ---------------- INFERENCE TEST ---------------- #
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["model_config"]["finetuned_model"],
    max_seq_length=config["model_config"]["max_seq_length"],
    dtype=config["model_config"]["dtype"],
    load_in_4bit=config["model_config"]["load_in_4bit"],
)

FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [
        "<|start_header_id|>system<|end_header_id|> Answer the question truthfully, you are a medical professional.<|eot_id|><|start_header_id|>user<|end_header_id|> This is the question: Can you provide an overview of the lung's squamous cell carcinoma?<|eot_id|>"
    ], return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
