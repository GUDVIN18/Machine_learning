import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import time
from typing import Dict, Union


model_name = "./mistral_finetuned_v1.0"


def load_model() -> tuple:
    """
    Загружает и оптимизирует модель для инференса.
    """
    print("Загрузка модели...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    config = PeftConfig.from_pretrained(model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        padding_side="left",
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(
        base_model,
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model.eval()
    return model, tokenizer

def generate_answer(
    model: PeftModel, 
    tokenizer: AutoTokenizer, 
    question: str,
    max_length: int = 256  # Уменьшили максимальную длину для кратких ответов
) -> tuple:
    """
    Генерирует краткий, детерминированный ответ на русском языке на основе обученных данных.
    """
    system_prompt = """Меня зовут Макс. Я отвечаю кратко, только на русском языке, 
    используя только проверенные факты из датасета. Моя цель - дать точный и 
    лаконичный ответ в 1-3 предложения без повторов."""
    
    prompt = f"[INST] {system_prompt}\n\nВопрос: {question} [/INST]"

    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length,
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()

    with torch.inference_mode(): 
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.001,  # Почти детерминированная генерация
            do_sample=False,
            top_p=1.0,
            top_k=1,
            repetition_penalty=1.2,  # Небольшой штраф за повторения
            length_penalty=0.8,  # Поощряем более короткие ответы
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            num_beams=1,
            early_stopping=True,  # Включаем раннюю остановку
            use_cache=True
        )

    generation_time = time.time() - start_time

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return response.strip(), generation_time

def main(question: str) -> Dict[str, Union[str, float]]:
    """
    Основная функция для генерации краткого ответа.
    """
    try:
        if not question or not question.strip():
            return {
                "response": "Вопрос не может быть пустым",
                "status": "error_input"
            }

        if question.lower() in ['exit', 'quit', 'выход']:
            return {
                "response": "Работа завершена",
                "status": "break"
            }

        model, tokenizer = load_model()

        try:
            answer, gen_time = generate_answer(model, tokenizer, question)
            
            return {
                "response": f"Ответ: {answer}\nВремя генерации: {gen_time:.2f} сек.",
                "status": "success"
            }

        except Exception as e:
            return {
                "response": f"Ошибка генерации: {str(e)}",
                "status": "error_generation"
            }

    except Exception as e:
        return {
            "response": f"Ошибка загрузки модели: {str(e)}",
            "status": "error_model"
        }
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    question_1 = 'Мне нужно создать бота для риелторского агентства. Помоги составить ТЗ'
    result_1 = main(question=question_1)
    print(f"{result_1}\n\n")