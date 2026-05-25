# 면접 문답 — 1단계: 글자 단위 토크나이저

> 사용법: 질문을 읽고 스스로 답한 뒤 `▶ 답` 을 펼쳐 확인하세요.  
> 난이도: ⭐ 기초 / ⭐⭐ 응용 / ⭐⭐⭐ 심화

---

## Q1. 토크나이저(Tokenizer)란 무엇인가? ⭐

<details>
<summary>▶ 답</summary>

문장(텍스트)을 컴퓨터가 처리할 수 있는 **숫자 ID 목록**으로 변환하는 도구.  
컴퓨터는 숫자만 이해하기 때문에, AI 모델에 텍스트를 입력하려면 반드시 숫자로 바꿔야 한다.

```
"hello"  →  토크나이저  →  [7, 6, 8, 8, 9]
```

</details>

---

## Q2. 토큰(Token)이란 무엇인가? ⭐

<details>
<summary>▶ 답</summary>

토크나이저가 문장을 쪼갤 때 나오는 **조각 하나**.  
어떤 기준으로 쪼개느냐에 따라 토큰의 단위가 달라진다.

| 방식 | "hello world" | 토큰 수 |
|---|---|---|
| 글자 단위 | h, e, l, l, o, ' ', w, o, r, l, d | 11개 |
| 단어 단위 | hello, world | 2개 |
| BPE | hel, lo, ' ', world | 4개 (예시) |

</details>

---

## Q3. 어휘집(Vocabulary)이란 무엇인가? ⭐

<details>
<summary>▶ 답</summary>

토크나이저가 알고 있는 **토큰 전체 목록**. 토큰과 숫자 ID의 대응표.

```python
char_to_id = {'h': 0, 'e': 1, 'l': 2, 'o': 3}
id_to_char = {0: 'h', 1: 'e', 2: 'l', 3: 'o'}
```

- `vocab_size`: 어휘집에 등록된 토큰의 총 개수
- 어휘집에 없는 토큰 → OOV(Out Of Vocabulary) 문제 발생

</details>

---

## Q4. 인코딩(Encoding)과 디코딩(Decoding)의 차이는? ⭐

<details>
<summary>▶ 답</summary>

| 방향 | 이름 | 사용 시점 |
|---|---|---|
| 문장 → 숫자 목록 | **인코딩** | AI 모델에 입력할 때 |
| 숫자 목록 → 문장 | **디코딩** | AI 모델의 출력을 사람이 읽을 수 있게 변환할 때 |

인코딩과 디코딩은 서로 역방향 연산이며, 같은 어휘집을 써야 왕복이 성립한다.

```
"hello"  →  [7, 6, 8, 8, 9]  →  "hello"  ✓
```

</details>

---

## Q5. 글자 단위 토크나이저의 장점과 단점은? ⭐⭐

<details>
<summary>▶ 답</summary>

**장점**
- 구현이 단순하다
- 영어 기준 ~100개의 작은 vocab_size로 모든 텍스트 처리 가능
- OOV가 사실상 발생하지 않는다 (알파벳 26자만 학습하면 됨)

**단점**
- 시퀀스가 너무 길어진다  
  "I love AI" → 9개 토큰 (글자 단위) vs 3개 (단어 단위)  
  AI는 토큰을 하나씩 처리하므로, 길수록 메모리·시간 비용 증가
- 의미 있는 단위를 잃는다  
  "apple" → 'a','p','p','l','e' — 각 글자는 아무 의미도 없음

</details>

---

## Q6. OOV(Out Of Vocabulary)란 무엇인가? ⭐⭐

<details>
<summary>▶ 답</summary>

**어휘집에 없는 토큰이 등장하는 상황**.  
학습(train)할 때 보지 못한 글자나 단어가 인코딩 단계에서 나타나면 처리할 수 없다.

- 글자 단위: "hello"로 학습 → 'z'를 만나면 OOV  
  (영문 알파벳 전체를 학습하면 사실상 해결됨)
- 단어 단위: 훨씬 심각 — "ChatGPT"처럼 학습 후 생긴 새 단어는 영구 OOV

**해결책**: UNK 토큰으로 대체하거나, BPE처럼 서브워드 단위로 쪼개는 방식을 사용

</details>

---

## Q7. Special Token 4가지(PAD, UNK, BOS, EOS)를 각각 설명하라. ⭐⭐

<details>
<summary>▶ 답</summary>

| 토큰 | ID | 이름 | 역할 |
|---|---|---|---|
| `[PAD]` | 0 | Padding | 배치 처리 시 짧은 문장의 빈 자리를 채움 |
| `[UNK]` | 1 | Unknown | 어휘집에 없는 토큰을 대체 |
| `[BOS]` | 2 | Begin Of Sentence | 문장의 시작을 모델에게 알림 |
| `[EOS]` | 3 | End Of Sentence | 문장의 끝을 모델에게 알림 |

**왜 ID가 0~3으로 고정인가?**  
모델 코드 전반에서 `PAD_ID == 0`을 직접 사용한다.  
나중에 바뀌면 학습된 모델과 어휘집 간 불일치가 발생해 모델이 망가진다.

</details>

---

## Q8. PAD 토큰이 왜 배치(batch) 처리에 필요한가? ⭐⭐

<details>
<summary>▶ 답</summary>

GPU는 여러 문장을 **같은 크기의 행렬**로 묶어 병렬 처리한다.  
문장 길이가 다르면 직사각형 행렬을 만들 수 없으므로, 짧은 문장의 끝을 PAD로 채워 길이를 맞춘다.

```
문장 A: [BOS] h e l l o [EOS]          → 길이 7
문장 B: [BOS] h i [EOS]                → 길이 4

패딩 후:
문장 A: [BOS] h e l l o [EOS] [PAD] [PAD]   → 길이 9
문장 B: [BOS] h i [EOS] [PAD] [PAD] [PAD] [PAD] [PAD] → 길이 9
```

모델은 PAD 위치를 무시하도록 **attention mask**를 함께 사용한다.

</details>

---

## Q9. vocab_size가 모델에 어떤 영향을 주는가? ⭐⭐

<details>
<summary>▶ 답</summary>

`vocab_size`는 모델의 **임베딩(Embedding) 행렬 크기**를 결정한다.

```
임베딩 행렬 크기 = vocab_size × embedding_dim
예: 50,000 × 768 = 38,400,000 개의 파라미터
```

- vocab_size가 크면 → 더 많은 토큰을 표현할 수 있지만, 메모리·연산량 증가
- vocab_size가 작으면 → 가벼운 모델이지만 OOV 위험 또는 시퀀스가 길어짐

**실제 사례**
| 모델 | vocab_size |
|---|---|
| 글자 단위 (영어) | ~100 |
| 단어 단위 | ~170,000+ |
| GPT-2 (BPE) | 50,257 |
| GPT-4 (BPE) | 100,277 |

</details>

---

## Q10. 학습한 어휘집을 파일로 저장해야 하는 이유는? ⭐⭐

<details>
<summary>▶ 답</summary>

**학습(train) 때의 ID와 추론(inference) 때의 ID가 반드시 일치해야 하기 때문**.

만약 저장하지 않고 매번 새로 어휘집을 만들면:
- 같은 단어가 학습 때는 ID=5, 추론 때는 ID=23이 될 수 있음
- 모델이 학습한 "ID=5는 'hello'를 의미함"이라는 지식이 완전히 틀린 입력을 받게 됨
- 결과: 모델이 엉터리 출력을 냄

실무에서는 `tokenizer.json`으로 어휘집을 저장하고 모델 파일과 함께 배포한다.

</details>

---

## Q11. Python의 `str`은 내부적으로 어떤 기준으로 동작하는가? ⭐⭐⭐

<details>
<summary>▶ 답</summary>

Python `str`은 **Unicode Code Point** 기준으로 동작한다.

```python
len("A")    # 1  (Code Point 1개)
len("한")   # 1  (Code Point 1개)
len("😀")   # 1  (Code Point 1개)
```

하지만 UTF-8 **바이트** 기준으로는 크기가 다르다:

```python
len("A".encode('utf-8'))    # 1바이트
len("한".encode('utf-8'))   # 3바이트
len("😀".encode('utf-8'))   # 4바이트
```

**왜 중요한가?**  
"글자 단위 토크나이저"라고 해도, Code Point 기준이냐 바이트 기준이냐에 따라 vocab_size가 달라진다.  
BPE(tiktoken)는 **바이트 기준**으로 동작해 어떤 언어도 256개 기본 토큰으로 처리할 수 있다.

</details>

---

## Q12. 인코딩 후 디코딩하면 항상 원본이 복원되는가? ⭐⭐⭐

<details>
<summary>▶ 답</summary>

**글자 단위에서는 Yes, 조건이 충족될 때만.**

조건:
1. 인코딩할 텍스트의 모든 글자가 어휘집에 있어야 함 (OOV 없을 것)
2. 인코딩과 디코딩이 동일한 어휘집을 사용해야 함

UNK 처리가 개입되면 복원 불가능:
```
"hello z" → 인코딩 → [7, 6, 8, 8, 9, 1] (z → UNK=1)
           → 디코딩 → "hello [UNK]"   ← 원본 복원 실패
```

이것이 토크나이저가 **손실 없이(lossless)** 동작해야 하는 이유이며,  
BPE가 256 바이트에서 시작해 OOV를 원천 차단하는 이유다.

</details>

---

## 빠른 복습 — 핵심 용어 카드

| 용어 | 한 줄 정의 |
|---|---|
| 토크나이저 | 텍스트 → 숫자 ID 목록으로 바꾸는 도구 |
| 토큰 | 문장을 쪼갠 조각 하나 |
| 어휘집 | 토큰 ↔ ID 대응표 |
| vocab_size | 어휘집에 등록된 토큰 수 |
| 인코딩 | 텍스트 → 숫자 |
| 디코딩 | 숫자 → 텍스트 |
| OOV | 어휘집에 없는 토큰이 등장하는 상황 |
| PAD | 배치 패딩용 토큰 (ID=0) |
| UNK | OOV 대체 토큰 (ID=1) |
| BOS | 문장 시작 토큰 (ID=2) |
| EOS | 문장 끝 토큰 (ID=3) |
| Code Point | Unicode 기준 문자 단위 (Python str) |
