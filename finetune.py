"""
LLM LoRA Fine-tuning Script (zero-to-llm)
Azure ML 환경에서 실행하는 범용 파인튜닝 코드
Hugging Face에 올라온 모든 모델에 동일하게 적용 가능

사용법:
    # EXAONE
    python finetune.py --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --data data/train.jsonl

    # LLaMA 3
    python finetune.py --model meta-llama/Meta-Llama-3-8B-Instruct --data data/train.jsonl

    # Mistral
    python finetune.py --model mistralai/Mistral-7B-Instruct-v0.3 --data data/train.jsonl

    # Qwen
    python finetune.py --model Qwen/Qwen2.5-7B-Instruct --data data/train.jsonl
"""

import argparse
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_jsonl(path: str) -> Dataset:
    """JSONL 파일을 로드해서 HuggingFace Dataset으로 반환"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"[데이터] {len(data)}개 대화 쌍 로드 완료")
    return Dataset.from_list(data)


def format_prompt(example: dict) -> dict:
    """대화를 EXAONE 프롬프트 형식으로 변환"""
    text = (
        f"[|system|] 당신은 도움이 되는 AI 어시스턴트입니다.\n"
        f"[|human|] {example['human']}\n"
        f"[|assistant|] {example['assistant']}"
    )
    return {"text": text}


def main(args):
    print(f"[모델] {args.model} 로딩 중...")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드 (4bit 양자화 - 메모리 절감)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        load_in_4bit=True,          # QLoRA: 4bit 양자화
        device_map="auto",
    )

    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,                      # LoRA 랭크 (높을수록 정확하지만 느림)
        lora_alpha=args.lora_r * 2,         # 보통 r의 2배 설정
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 예시 출력: trainable params: 8,388,608 || all params: 7,808,475,136 || trainable%: 0.107

    # 데이터 로드 및 전처리
    dataset = load_jsonl(args.data)
    dataset = dataset.map(format_prompt)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,     # 효과적 배치 크기 = 8
        learning_rate=2e-4,
        fp16=True,                          # 메모리 절감
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("[훈련] 시작...")
    trainer.train()

    # 모델 저장
    print(f"[저장] {args.output}에 저장 중...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("[완료] 파인튜닝 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXAONE LoRA 파인튜닝")
    parser.add_argument("--model", default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    parser.add_argument("--data", required=True, help="JSONL 데이터 경로")
    parser.add_argument("--output", default="outputs/exaone-finetuned")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 랭크")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)
