import torch
import os
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from huggingface_hub import login

# Авторизация в Hugging Face
login(token="")

# Проверка CUDA и очистка памяти
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./mistral_finetuned_v1.0"
MAX_LENGTH = 128
JSONL_PATH = "./new_dataset.jsonl"

# Конфигурация квантизации для экономии памяти
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Функция загрузки данных из JSONL
def load_jsonl_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append({
                    'input': item['input'],
                    'output': item['output']
                })
            except json.JSONDecodeError:
                print(f"Ошибка при чтении строки: {line}")
                continue
    return data

# Загрузка модели с оптимизированными настройками
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Подготовка модели для 4-bit обучения
model = prepare_model_for_kbit_training(model)

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
    return f"<s>[INST] {example['input']} [/INST] {example['output']}</s>"

def generate_and_tokenize_prompt(prompt):
    tokens = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Загрузка и подготовка данных
json_data = load_jsonl_data(JSONL_PATH)
dataset = Dataset.from_list(json_data)
train_val = dataset.train_test_split(test_size=0.1, seed=42)

# Токенизация данных
tokenized_train_dataset = train_val["train"].map(
    generate_and_tokenize_prompt,
    remove_columns=train_val["train"].column_names,
    batch_size=32
)
tokenized_val_dataset = train_val["test"].map(
    generate_and_tokenize_prompt,
    remove_columns=train_val["test"].column_names,
    batch_size=32
)

# Оптимизированная конфигурация LoRA для меньшего потребления памяти
lora_config = LoraConfig(
    r=8,  # Уменьшаем ранг
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",
    lora_dropout=0.05,  # Уменьшаем dropout
    task_type=TaskType.CAUSAL_LM
)

# Применяем LoRA к модели
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # Улучшаем процесс обучения
    warmup_steps=100,               # Увеличиваем для лучшей стабилизации
    max_steps=1400,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    
    # Оптимизатор и learning rate
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    weight_decay=0.01,              # Немного увеличиваем для лучшей регуляризации
    max_grad_norm=0.5,              # Увеличиваем для стабильности
    
    # Логирование и валидация
    logging_steps=10,               # Чаще логируем
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=50,                  # Чаще проводим валидацию
    do_eval=True,
    save_total_limit=3,             # Сохраняем только лучшие чекпоинты
    
    # Оптимизация памяти и скорости
    gradient_checkpointing=True,
    fp16=True,
    bf16=False,
    dataloader_pin_memory=True,
    torch_compile=False,
    report_to="none",
)

# Инициализация тренера
trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Запуск обучения
model.config.use_cache = False
trainer.train()
torch.cuda.empty_cache()

# Сохранение модели
trainer.save_model(OUTPUT_DIR)
print(f"Модель сохранена в {OUTPUT_DIR}")
