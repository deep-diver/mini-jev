# mini-jev: Google Gemma 3 270M 기반 Jev 호환 초고속 의사결정 엔진

> **System One Programmatic Decision Engine for Apple Silicon (MPS)**  
> Inspired by [TypeSafe AI's Jev](https://typesafe.ai)

`mini-jev`는 구글의 초경량 언어 모델인 **Gemma 3 270M-IT**를 활용하여, TypeSafe Jev의 **"System One" 비생성형(Non-autoregressive) 의사결정 인터페이스**를 로컬 Apple Silicon 환경에서 초고속으로 구동할 수 있도록 구현한 라이브러리입니다.

---

## ⚡ 주요 특징

1. **텍스트 생성 루프 제거 (Single Forward Pass)**
   * `generate()`로 JSON 텍스트를 한 글자씩 생성하지 않고, **단 1회의 순전파(Forward pass)로 후보 토큰의 로짓(Logits)만 추출**하여 확률을 계산합니다.
2. **압도적인 속도 (Apple Silicon MPS 최적화)**
   * M2 Max 기준 단일 판단에 **~27ms**, 2개 질문 동시 평가에 **~78ms**의 극도로 빠른 응답 속도를 달성합니다.
3. **TypeSafe Jev 완전 호환 3대 프리미티브 지원**
   * **`noul`**: 이진(Yes/No) 참/거짓 확률 판단 ($p \in [0.0, 1.0]$)
   * **`choice`**: 최대 255개 사전 정의 카테고리 분류 및 전 옵션 확률 분포
   * **`score`**: 루브릭 기반 연속형 수치 점수 및 레벨별 확률 계산
4. **구조적 타입 에러 제로 (0%)**
   * 사전 정의된 스키마 내부에서만 값이 결정되므로, JSON 파싱 에러나 없는 필드를 만들어내는 환각이 원천 차단됩니다.

---

## 🚀 빠른 시작 (Quickstart)

### 1. 환경 설정

```bash
conda create -n mini-jev python=3.11 -y
conda activate mini-jev
pip install -r requirements.txt
```

> **참고**: `google/gemma-3-270m-it`는 Hugging Face 게이트 모델이므로, [Hugging Face 모델 페이지](https://huggingface.co/google/gemma-3-270m-it)에서 라이선스 동의 후 로컬에 토큰이 등록되어 있어야 합니다.

### 2. 기본 사용법

```python
from mini_jev import MiniJevClient

# Apple Silicon MPS에 모델 로드 (bfloat16)
client = MiniJevClient(model="google/gemma-3-270m-it")

# 비정형 상태 데이터
state = {
    "user": "alice",
    "message": "우리 프로덕션 웹훅 서버가 2시간째 500 에러를 내고 있습니다. 결제가 전혀 안 되고 있어요!"
}

# Jev 스타일 판단 질문들
response = client.evaluate(
    state=state,
    questions={
        "is_outage": {
            "type": "noul",
            "instructions": "이 메시지가 치명적인 프로덕션 장애나 서비스 다운을 묘사합니까?"
        },
        "department": {
            "type": "choice",
            "instructions": "어느 부서가 처리해야 합니까?",
            "criteria": {
                "billing": "결제 및 청구 문제",
                "infrastructure": "서버 장애, 프로덕션 다운, API 에러",
                "general": "일반 문의, 피드백"
            }
        },
        "urgency_score": {
            "type": "score",
            "instructions": "긴급도 수준을 평가하세요.",
            "criteria": [
                "Low: 영향 없음",
                "Medium: 일부 기능 제약, 우회 방법 존재",
                "High: 심각한 영향, 일부 시스템만 가동",
                "Critical: 완전한 장애 또는 치명적 손실"
            ]
        }
    }
)

# 결과 확인
print(response.answers["is_outage"].noul)         # 0.9903 (Yes 99.03%)
print(response.answers["department"].choice)      # 'infrastructure'
print(response.answers["urgency_score"].score)    # 2.05 (Level 2~3 집중)
print(f"지연 시간: {response.usage.latency_ms} ms") # ~79 ms
```

---

## 📊 Apple Silicon M2 Max 벤치마크 결과

`benchmark.py` 실행 결과 (50회 반복 측정, 2개 복합 질문 동시 평가 기준):

| 지표 | 측정값 (ms) | 비고 |
| :--- | :--- | :--- |
| **평균 지연 시간 (Mean)** | **78.17 ms** | 질문당 약 39 ms |
| **중간값 (P50)** | **77.38 ms** | 매우 안정적인 레이턴시 분포 |
| **P95 지연 시간** | **82.46 ms** | 스파이크 없음 |
| **최소 지연 시간 (Min)** | **75.83 ms** | |
| **처리량 (Throughput)** | **12.8 queries/sec** | 단일 스트림 기준 |
| **단일 포워드 패스 속도** | **~27.2 ms** | MPS `bfloat16` 연산 |

---

## 🧠 왜 Jev는 RLCD(강화학습)를 필요로 했는가?

`mini-jev`를 구현하며 얻은 중요한 통찰:
* **기본 LLM의 사전 편향(Prior Bias)**:
  학습되지 않은 일반 LLM(Gemma 3 270M 포함)은 객관식에서 첫 번째 보기(A)나 "Yes" 토큰의 빈도 사전 확률(Unigram Frequency)이 자연어 특성상 매우 높습니다.
* **Jev의 해결책**:
  TypeSafe Jev가 일반 프롬프트 LLM과 구별되는 이유는 바로 **RLCD (Reinforcement Learning for Calibrated Decisions)**로 모델 가중치를 의사결정 확률 보정에 직접 최적화했기 때문입니다.
* **mini-jev의 고도화 방향**:
  1. **Contextual Calibration**: Null Prompt (`State: N/A`)의 로짓을 차감하여 기본 편향 상쇄
  2. **LoRA / Head Fine-tuning**: 270M 모델에 가벼운 태스크별 분류 헤드 또는 LoRA 튜닝 적용

---

## 📁 프로젝트 구조

```
mini-jev/
├── mini_jev/
│   ├── __init__.py       # 패키지 진입점
│   ├── schemas.py        # Noul, Choice, Score Pydantic 스키마 정의
│   ├── engine.py         # Gemma 3 모델 로드 및 Single-Pass Logit Scorer
│   └── client.py         # TypeSafe Jev 호환 MiniJevClient
├── examples/
│   ├── basic_usage.py    # 고객 지원 티켓 분류 및 긴급도 평가 데모
│   └── safety_guard.py   # AI 에이전트 터미널 명령어 위험성 사전 검사 데모
├── benchmark.py          # Apple Silicon MPS 성능 측정 벤치마크
├── requirements.txt      # 의존성 목록
└── README.md
```
