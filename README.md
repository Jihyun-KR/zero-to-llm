# 🚀 zero-to-llm

> ChatGPT 대화 데이터로 나만의 LLM을 파인튜닝하는 end-to-end 파이프라인

[![Platform](https://img.shields.io/badge/Platform-Azure%20ML-blue)](https://ml.azure.com)
[![Method](https://img.shields.io/badge/Method-LoRA%20%2F%20QLoRA-orange)](https://github.com/huggingface/peft)
[![Models](https://img.shields.io/badge/Models-EXAONE%20%7C%20LLaMA%20%7C%20Mistral%20%7C%20Qwen-green)](https://huggingface.co/models)

---

## 📌 프로젝트 개요

이 프로젝트는 개인 ChatGPT 대화 데이터를 이용해 **오픈소스 LLM을 파인튜닝**하는 end-to-end 파이프라인입니다.
EXAONE, LLaMA, Mistral, Qwen 등 **Hugging Face에 올라온 모든 모델**에 동일하게 적용됩니다.

- **데이터 수집**: ChatGPT HTML 내보내기 → JSONL 변환 (노코드)
- **데이터 가공**: 개인정보 마스킹, 비중 조절 (Data Mixing)
- **모델 훈련**: Azure ML + LoRA / QLoRA (파라미터 효율적 파인튜닝)
- **배포**: Azure ML Endpoint → REST API

---

## 🗂 전체 파이프라인

```
ChatGPT HTML 파일
      ↓
[1] converter.html    → JSONL 변환 (브라우저, 노코드, 로컬 처리)
      ↓
[2] data_mix.py       → 데이터 비중 조절 및 전처리
      ↓
[3] finetune.py       → Azure ML LoRA 파인튜닝
      ↓
[4] Azure ML Endpoint → REST API 배포
```

---

## 🤖 지원 모델

Hugging Face에서 받을 수 있는 모델이라면 모두 동일하게 사용 가능합니다.

| 모델 | 추천 크기 | 특징 |
|------|----------|------|
| **LG EXAONE 3.5** | 7.8B | 한국어 특화, 국산 모델 |
| **Meta LLaMA 3** | 8B | 가장 널리 사용됨 |
| **Mistral** | 7B | 유럽산, 성능 우수 |
| **Qwen 2.5** | 7B | 한국어 지원, 알리바바 |
| **Google Gemma** | 7B | 경량, 효율적 |

---

## 📁 파일 구조

```
zero-to-llm/
├── converter.html       # ChatGPT HTML → JSONL 변환기 (노코드 웹앱)
├── data_mix.py          # 데이터 비중 조절 스크립트
├── finetune.py          # Azure ML LoRA 파인튜닝 코드
├── requirements.txt     # Python 패키지 목록
├── docs/
│   └── pipeline.md      # 전체 파이프라인 상세 설명
└── README.md
```

---

## 🚀 빠른 시작

### 1단계 — 데이터 변환 (노코드)

`converter.html`을 브라우저에서 열고 ChatGPT HTML 파일을 드래그앤드롭하면 JSONL로 변환됩니다.

- 개인정보 자동 마스킹 (전화번호, 이메일 등)
- 데이터 비중 강조 설정 (반복 횟수)
- 브라우저 내에서만 처리 (데이터 외부 전송 없음 🔒)

### 2단계 — 데이터 전처리

```bash
pip install -r requirements.txt
python data_mix.py --input data/raw.jsonl --output data/train.jsonl --weight 3
```

### 3단계 — Azure ML 파인튜닝

모델 이름만 바꾸면 어떤 모델이든 동일하게 동작합니다.

```bash
# EXAONE
python finetune.py --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --data data/train.jsonl

# LLaMA 3
python finetune.py --model meta-llama/Meta-Llama-3-8B-Instruct --data data/train.jsonl

# Mistral
python finetune.py --model mistralai/Mistral-7B-Instruct-v0.3 --data data/train.jsonl
```

---

## 🛠 기술 스택

| 구분 | 기술 |
|------|------|
| 파인튜닝 방법 | LoRA / QLoRA (PEFT) |
| 훈련 인프라 | Azure ML (GPU: A100 스팟 인스턴스) |
| 프레임워크 | Hugging Face Transformers, PEFT |
| 데이터 처리 | 브라우저 기반 노코드 변환기 |
| 모델 소스 | Hugging Face Hub |

---

## 💰 예상 비용

| 항목 | 비용 |
|------|------|
| A100 스팟 인스턴스 (10시간) | 약 $15~30 |
| Azure Blob Storage | 월 $1~5 |
| Azure 첫 가입 크레딧 | **$200 무료** |

---

## 📚 참고 자료

- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [Azure ML 공식 문서](https://docs.microsoft.com/azure/machine-learning/)
- [EXAONE 3.5](https://github.com/LG-AI-EXAONE/EXAONE-3.5)
- [LLaMA 3](https://huggingface.co/meta-llama)
- [Hugging Face AutoTrain](https://huggingface.co/autotrain)
