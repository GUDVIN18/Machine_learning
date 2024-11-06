# !python -m venv venv 
# !source venv/bin/activate

# !pip install torch transformers datasets accelerate peft bitsandbytes wandb

# !pip install huggingface_hub


import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from huggingface_hub import login

# Авторизация в Hugging Face
login(token="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")

# Проверка CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

# Очистка кэша CUDA
torch.cuda.empty_cache()

# Параметры конфигурации
BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"
TRAIN_DATASET_PATH = "./main.jsonl"
VAL_DATASET_PATH = "./val_dataset.jsonl"
OUTPUT_DIR = "./llama2_finetuned"
MAX_LENGTH = 128

# Загрузка базовой модели с оптимизацией для CUDA
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",  # Автоматическое распределение на GPU
    torch_dtype=torch.float16,  # Используем float16 для экономии памяти
    load_in_8bit=True,  # Квантизация 8-бит для уменьшения потребления памяти
)

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# Функции подготовки данных
def formatting_func(example):
    return f"### Question: {example['input']}\n### Answer: {example['output']}"

def generate_and_tokenize_prompt(prompt):
    tokens = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Загрузка и токенизация данных
train_dataset = load_dataset('json', data_files=TRAIN_DATASET_PATH, split='train')
val_dataset = load_dataset('json', data_files=VAL_DATASET_PATH, split='train')

tokenized_train_dataset = train_dataset.map(
    generate_and_tokenize_prompt,
    remove_columns=train_dataset.column_names,
    batch_size=32
)
tokenized_val_dataset = val_dataset.map(
    generate_and_tokenize_prompt,
    remove_columns=val_dataset.column_names,
    batch_size=32
)

# Оптимизированная конфигурация LoRA
lora_config = LoraConfig(
    r=16,  # Увеличен ранг
    lora_alpha=32,  # Увеличена alpha
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",  # Добавлены дополнительные слои
        "up_proj",
        "down_proj"
    ],
    bias="none",
    lora_dropout=0.1,  # Увеличен dropout
    task_type=TaskType.CAUSAL_LM
)

# Применяем LoRA к модели
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Улучшенные аргументы тренировки
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    warmup_steps=100,  # Увеличено количество шагов прогрева
    per_device_train_batch_size=8,  # Увеличен размер батча
    gradient_accumulation_steps=8,  # Увеличено накопление градиентов
    max_steps=1000,  # Значительно увеличено количество шагов
    learning_rate=2e-4,
    optim="adamw_torch",
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=200,  # Увеличен интервал сохранения
    evaluation_strategy="steps",
    eval_steps=200,  # Увеличен интервал оценки
    do_eval=True,
    report_to="none",
    gradient_checkpointing=True,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    weight_decay=0.001,
    lr_scheduler_type="cosine",
    dataloader_pin_memory=True,
    torch_compile=True,
    optim_args={"capturable": True},
)
# Тренер с включенным кэшированием
# trainer = Trainer(
#     model=model,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_val_dataset,
#     args=training_args,
#     data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
# )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# # Запуск обучения с очисткой кэша
# def train_with_cache_cleanup():
#     model.config.use_cache = False
#     trainer.train()
#     torch.cuda.empty_cache()  # Очистка кэша после обучения

# # Запуск функции
# train_with_cache_cleanup()

# Сохранение модели
trainer.save_model(OUTPUT_DIR)
print(f"Модель сохранена в {OUTPUT_DIR}")