# zero-to-llm 전체 파이프라인 상세 설명

## 개요

이 문서는 ChatGPT 대화 데이터를 이용해 오픈소스 LLM을 파인튜닝하는 전체 과정을 설명합니다.
EXAONE, LLaMA, Mistral, Qwen 등 Hugging Face의 모든 모델에 동일하게 적용됩니다.

---

## 1단계 — 데이터 준비 (converter.html)

ChatGPT에서 대화를 내보내면 HTML 파일이 생성됩니다.
`converter.html`을 브라우저에서 열어 드래그앤드롭으로 JSONL로 변환합니다.

**출력 형식:**
```json
{"human": "사용자 질문", "assistant": "ChatGPT 답변"}
{"human": "사용자 질문2", "assistant": "ChatGPT 답변2"}
```

**주요 기능:**
- 개인정보 자동 마스킹 (전화번호, 이메일 등)
- 빈 응답 자동 제거
- 데이터 비중 강조 (반복 횟수 설정)
- 100% 로컬 처리 (데이터 보안)

---

## 2단계 — 데이터 전처리 (data_mix.py)

여러 데이터셋을 원하는 비율로 믹싱합니다.

```bash
python data_mix.py --input data/raw.jsonl --output data/train.jsonl --weight 3
```

**비중 조절 원리:**
- `--weight 1`: 기본 (변경 없음)
- `--weight 3`: 해당 데이터를 3배 반복 → 3배 강조 학습

---

## 3단계 — Azure ML 설정

### 컴퓨팅 클러스터 생성
1. ml.azure.com 접속
2. 컴퓨팅 → 클러스터 만들기
3. GPU: Standard_NC24ads_A100_v4 (스팟 인스턴스 선택 시 ~70% 할인)

### 데이터 업로드
1. 데이터 → 데이터 자산 만들기
2. train.jsonl 업로드

---

## 4단계 — LoRA 파인튜닝 (finetune.py)

### LoRA란?

전체 70억 파라미터를 학습하는 대신 **약 0.1%만 학습**합니다.

```
전체 파라미터:  7,808,475,136개
LoRA 파라미터:      8,388,608개 (0.107%)
```

비용과 시간이 99% 절감됩니다.

### 실행

```bash
python finetune.py \
  --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --data data/train.jsonl \
  --output outputs/exaone-finetuned \
  --lora_r 16 \
  --epochs 3
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `lora_r` | 16 | LoRA 랭크. 높을수록 정확하지만 느림 |
| `epochs` | 3 | 전체 데이터 반복 횟수 |
| `learning_rate` | 2e-4 | 학습률 |

---

## 5단계 — 배포

훈련 완료 후 Azure ML Endpoint로 배포합니다.

1. ml.azure.com → 모델 → 등록
2. 엔드포인트 → 실시간 엔드포인트 만들기
3. REST API로 호출:

```python
import requests

response = requests.post(
    "https://your-endpoint.azureml.net/score",
    json={"human": "안녕하세요!"},
    headers={"Authorization": "Bearer YOUR_KEY"}
)
print(response.json())
```

---

## 예상 비용 (7.8B 기준)

| 항목 | 비용 |
|------|------|
| A100 스팟 인스턴스 (10시간) | 약 $15~30 |
| Azure Blob Storage | 월 $1~5 |
| Azure 첫 가입 크레딧 | $200 무료 |

**소규모 실험은 사실상 무료로 가능합니다.**

---

## 라이선스 주의사항

EXAONE 3.5는 **비상업적(NC) 라이선스**입니다.
- 연구/학습 목적: 자유롭게 사용 가능
- 상업적 서비스: LG AI Research 별도 계약 필요
- 문의: contact_us@lgresearch.ai

---

## 모델별 라이선스 요약

모델마다 라이선스가 다릅니다. 사용 전 반드시 확인하세요.

| 모델 | 라이선스 | 상업적 사용 |
|------|---------|-----------|
| EXAONE 3.5 | NC (비상업) | LG 별도 계약 필요 |
| LLaMA 3 | Meta 커뮤니티 | MAU 7억 이하 무료 |
| Mistral 7B | Apache 2.0 | 자유롭게 사용 가능 |
| Qwen 2.5 | Apache 2.0 | 자유롭게 사용 가능 |
| Gemma | Google 조건부 | 상업적 사용 허용 |
