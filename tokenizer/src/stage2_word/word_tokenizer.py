# -*- coding: utf-8 -*-
# =============================================================================
# word_tokenizer.py — 단어 단위 토크나이저 (Word-level Tokenizer)
#
# 목적:
#   문장을 공백 기준으로 단어 단위로 쪼개고, 각 단어에 숫자 ID를 부여합니다.
#   글자 단위 토크나이저보다 더 의미 있는 단위로 처리하지만,
#   치명적인 OOV(Out Of Vocabulary) 문제가 있습니다.
#
# 이 파일을 통해 배우는 것:
#   - 단어 어휘집이 얼마나 빠르게 커지는가 (vocab_size 폭발 문제)
#   - OOV 문제가 실제로 어떻게 발생하는가
#   - 왜 단어 단위 방식만으로는 부족한가
#   - 이 한계가 BPE 탄생으로 이어지는 흐름
# =============================================================================

import sys
import json   # 어휘집을 JSON 파일로 저장/불러오기 위해 사용
import re     # 정규식으로 단어 분리에 사용

try:
    sys.stdout.reconfigure(encoding='utf-8')  # Windows 터미널 한글/특수문자 출력 보장
except AttributeError:
    pass  # Jupyter OutStream은 reconfigure 없음


# -----------------------------------------------------------------------------
# 특수 토큰 정의 — char_tokenizer.py와 동일한 규칙을 따릅니다.
#
# 단어 단위 토크나이저에서도 특수 토큰의 ID는 고정입니다.
# 모델 코드 전반에 걸쳐 PAD=0, UNK=1이라는 사실을 직접 사용하기 때문입니다.
# -----------------------------------------------------------------------------
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

PAD_ID = 0  # 채우기 토큰 — 배치 패딩에 사용
UNK_ID = 1  # 미지 토큰 — 어휘집에 없는 단어를 대체 (이것이 OOV 문제의 핵심!)
BOS_ID = 2  # 문장 시작 토큰
EOS_ID = 3  # 문장 끝 토큰


class WordTokenizer:
    """
    단어 단위 토크나이저 클래스

    글자 단위 토크나이저와 비교했을 때의 차이점:
        - 쪼개는 단위가 '글자'에서 '단어'로 바뀜
        - 더 의미 있는 단위로 처리됨 (예: 'hello'를 하나의 토큰으로)
        - 하지만 학습 데이터에 없는 단어는 [UNK]로 대체됨 → OOV 문제

    기본 사용 예시:
        tokenizer = WordTokenizer()
        tokenizer.train("I love machine learning")
        ids = tokenizer.encode("I love AI", add_special_tokens=True)
        # 결과: 'AI'는 학습 데이터에 없으므로 [UNK]로 처리됨

    저장/불러오기 예시:
        tokenizer.save("word_tokenizer.json")
        tokenizer.load("word_tokenizer.json")
    """

    def __init__(self):
        # ---------------------------------------------------------------------
        # 어휘집 초기화: 특수 토큰을 먼저 0번부터 등록합니다.
        # char_tokenizer.py와 동일한 초기화 패턴을 따릅니다.
        # ---------------------------------------------------------------------

        # 단어 → ID 변환 사전
        # 예: {'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3, 'hello': 4, ...}
        self.word_to_id: dict[str, int] = {}

        # ID → 단어 변환 사전 (디코딩에 사용)
        # 예: {0: '[PAD]', 1: '[UNK]', 4: 'hello', ...}
        self.id_to_word: dict[int, str] = {}

        # 어휘집에 등록된 토큰 총 개수
        self.vocab_size: int = 0

        # 특수 토큰을 0번부터 순서대로 등록합니다.
        for token in SPECIAL_TOKENS:
            self._register_token(token)

    def _register_token(self, token: str):
        """
        토큰 하나를 어휘집에 등록하는 내부 헬퍼 메서드입니다.
        이미 등록된 토큰은 건너뜁니다.
        """
        if token not in self.word_to_id:
            new_id = self.vocab_size           # 현재까지 등록된 수 = 새 ID
            self.word_to_id[token] = new_id
            self.id_to_word[new_id] = token
            self.vocab_size += 1

    def _split_into_words(self, text: str) -> list[str]:
        """
        텍스트를 단어 목록으로 분리합니다.

        분리 방법:
            공백(스페이스, 탭, 줄바꿈)을 기준으로 분리합니다.
            Python의 str.split()은 연속된 공백을 자동으로 처리합니다.

        예시:
            "I love  AI"  →  ['I', 'love', 'AI']
            (연속된 공백이 있어도 올바르게 처리됩니다.)

        참고:
            더 정교한 구현에서는 구두점도 분리합니다.
            예: "hello!" → ['hello', '!']
            하지만 이 단계에서는 단순하게 공백 기준으로만 분리합니다.
        """
        # split()은 인수 없이 호출하면 모든 종류의 공백으로 분리합니다.
        # 빈 문자열은 자동으로 제외됩니다.
        return text.split()

    def train(self, text: str):
        """
        텍스트를 학습하여 단어 어휘집을 구축합니다.

        학습 과정:
            1. 텍스트를 단어 단위로 분리
            2. 중복을 제거하고 알파벳 순으로 정렬
            3. 각 단어에 ID를 부여하여 어휘집에 등록

        중요한 점:
            이 메서드로 학습한 단어만 어휘집에 들어갑니다.
            학습 후에 새로운 단어가 등장하면 [UNK]로 처리됩니다.
            이것이 바로 OOV(Out Of Vocabulary) 문제입니다.

        매개변수:
            text (str): 학습에 사용할 텍스트

        예시:
            train("I love machine learning")
            → 특수 토큰: [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3
            → 단어들:  'I'=4, 'learning'=5, 'love'=6, 'machine'=7  (알파벳 정렬)
        """
        # 텍스트를 단어 목록으로 분리합니다.
        words = self._split_into_words(text)

        # 중복을 제거하고 정렬합니다.
        # 정렬해야 같은 텍스트로 학습할 때 항상 같은 ID가 부여됩니다.
        unique_words = sorted(set(words))

        # 각 단어를 어휘집에 등록합니다.
        for word in unique_words:
            self._register_token(word)

        # 학습 결과를 사람이 보기 쉽게 출력합니다.
        print(f"학습 완료!")
        print(f"  어휘집 크기: {self.vocab_size}개 토큰")
        print(f"  특수 토큰 4개 + 단어 {self.vocab_size - 4}개")
        print(f"  등록된 단어 수: {len(unique_words)}개")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        문장을 숫자 ID 목록으로 변환합니다. (인코딩)

        핵심 동작 — OOV 처리:
            어휘집에 없는 단어는 [UNK] ID(=1)로 대체됩니다.
            이것이 단어 단위 토크나이저의 가장 큰 한계입니다.

            예시:
                train("I love machine")  # 학습
                encode("I love AI")      # 인코딩
                → 'AI'는 학습 데이터에 없으므로 [UNK](ID=1)로 대체
                → [2, 4, 6, 1, 3]  (add_special_tokens=True일 때)
                                ↑
                              [UNK]

        매개변수:
            text (str): 인코딩할 문장
            add_special_tokens (bool): True이면 앞에 [BOS], 뒤에 [EOS]를 추가합니다.

        반환값:
            list[int]: 숫자 ID 목록
        """
        ids = []

        # add_special_tokens=True이면 맨 앞에 [BOS] 토큰 ID를 추가합니다.
        if add_special_tokens:
            ids.append(BOS_ID)

        # 텍스트를 단어 단위로 분리합니다.
        words = self._split_into_words(text)

        for word in words:
            if word in self.word_to_id:
                # 어휘집에 있는 단어 → 해당 ID를 사용합니다.
                ids.append(self.word_to_id[word])
            else:
                # 어휘집에 없는 단어(OOV) → [UNK] ID(=1)로 대체합니다.
                # 이것이 OOV 문제가 실제로 발생하는 순간입니다!
                ids.append(UNK_ID)

        # add_special_tokens=True이면 맨 뒤에 [EOS] 토큰 ID를 추가합니다.
        if add_special_tokens:
            ids.append(EOS_ID)

        return ids

    def encode_with_oov_report(self, text: str) -> dict:
        """
        인코딩과 함께 OOV 단어 목록을 함께 반환합니다.

        학습 목적 메서드:
            어떤 단어가 OOV 처리되었는지 명시적으로 보여줍니다.
            실제 AI 시스템에서는 이런 식으로 OOV를 추적하기도 합니다.

        반환값:
            dict: {
                'ids': [숫자 ID 목록],
                'oov_words': [어휘집에 없었던 단어 목록],
                'oov_count': OOV 단어 개수,
                'oov_rate': OOV 비율 (0.0 ~ 1.0)
            }
        """
        words = self._split_into_words(text)
        ids = []
        oov_words = []

        for word in words:
            if word in self.word_to_id:
                # 어휘집에 있는 단어
                ids.append(self.word_to_id[word])
            else:
                # OOV 단어: UNK로 대체하면서 목록에 기록합니다.
                ids.append(UNK_ID)
                oov_words.append(word)

        # OOV 비율을 계산합니다. (전체 단어 중 몇 %가 OOV인가?)
        total_words = len(words)
        oov_count = len(oov_words)
        oov_rate = oov_count / total_words if total_words > 0 else 0.0

        return {
            'ids': ids,
            'oov_words': oov_words,
            'oov_count': oov_count,
            'oov_rate': oov_rate,
        }

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        숫자 ID 목록을 문장으로 복원합니다. (디코딩)

        단어 단위 토크나이저는 단어 사이의 공백을 자동으로 복원합니다.
        인코딩할 때 공백이 분리 기준으로 사용되었으므로,
        디코딩할 때는 단어 사이에 공백을 다시 넣어줍니다.

        매개변수:
            ids (list[int]): 숫자 ID 목록
            skip_special_tokens (bool): True이면 특수 토큰을 결과에서 제외합니다.

        반환값:
            str: 복원된 문장

        예시:
            decode([2, 4, 6, 3], skip_special_tokens=True)
            → "I love"   (BOS, EOS 제거, 단어 사이 공백 복원)
        """
        # 특수 토큰 문자열 집합 (빠른 조회를 위해 set으로)
        special_token_set = set(SPECIAL_TOKENS)

        words = []

        for id_ in ids:
            if id_ not in self.id_to_word:
                # 알 수 없는 ID는 '[UNK]' 문자열로 표시합니다.
                words.append("[UNK]")
                continue

            token = self.id_to_word[id_]

            # skip_special_tokens=True면 특수 토큰을 건너뜁니다.
            if skip_special_tokens and token in special_token_set:
                continue

            words.append(token)

        # 단어 사이에 공백을 넣어 문장을 복원합니다.
        # 글자 단위와 달리, 단어 단위는 join 시 공백이 필요합니다.
        return ' '.join(words)

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
        pad_to_max_length: bool = True,
    ) -> list[list[int]]:
        """
        여러 문장을 한 번에 인코딩합니다. (배치 인코딩)

        char_tokenizer.py의 encode_batch와 동일한 구조입니다.
        짧은 문장은 [PAD](ID=0)으로 채워 길이를 통일합니다.

        매개변수:
            texts (list[str]): 인코딩할 문장 목록
            add_special_tokens (bool): 각 문장에 [BOS], [EOS]를 추가할지 여부
            pad_to_max_length (bool): True이면 [PAD]로 길이를 통일합니다.

        반환값:
            list[list[int]]: 각 문장의 ID 목록을 담은 2차원 목록
        """
        # 1단계: 각 문장을 개별적으로 인코딩합니다.
        encoded_list = [
            self.encode(text, add_special_tokens=add_special_tokens)
            for text in texts
        ]

        if pad_to_max_length and encoded_list:
            # 2단계: 가장 긴 문장의 길이를 구합니다.
            max_length = max(len(ids) for ids in encoded_list)

            # 3단계: 짧은 문장의 뒤를 [PAD](ID=0)으로 채웁니다.
            padded_list = []
            for ids in encoded_list:
                pad_length = max_length - len(ids)
                padded_ids = ids + [PAD_ID] * pad_length
                padded_list.append(padded_ids)

            return padded_list

        return encoded_list

    def save(self, path: str):
        """
        현재 어휘집을 JSON 파일로 저장합니다.

        저장 내용:
            - vocab_size: 어휘집 크기
            - word_to_id: 단어 → ID 매핑 사전
            - special_tokens: 특수 토큰 목록

        매개변수:
            path (str): 저장할 파일 경로 (예: "word_tokenizer.json")
        """
        data = {
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
            "special_tokens": SPECIAL_TOKENS,
        }

        with open(path, "w", encoding="utf-8") as f:
            # ensure_ascii=False: 한글 등 비ASCII 문자를 이스케이프 없이 저장합니다.
            # indent=2: 사람이 읽기 쉽도록 들여쓰기를 추가합니다.
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"어휘집 저장 완료: {path}")
        print(f"  총 {self.vocab_size}개 토큰")

    def load(self, path: str):
        """
        JSON 파일에서 어휘집을 불러옵니다.

        주의: 불러오면 기존 어휘집은 파일 내용으로 교체됩니다.

        매개변수:
            path (str): 불러올 파일 경로
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 파일에서 읽은 word_to_id를 복원합니다.
        self.word_to_id = data["word_to_id"]
        self.vocab_size = data["vocab_size"]

        # id_to_word는 word_to_id에서 역으로 재구성합니다.
        # JSON에서 읽으면 value가 정수이므로 그대로 사용합니다.
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}

        print(f"어휘집 불러오기 완료: {path}")
        print(f"  총 {self.vocab_size}개 토큰")

    def get_vocab(self) -> dict:
        """현재 어휘집(단어 → ID 사전)을 반환합니다."""
        return self.word_to_id


# =============================================================================
# 아래는 이 파일을 직접 실행할 때만 동작하는 테스트 코드입니다.
# "python word_tokenizer.py" 로 실행하면 결과를 확인할 수 있습니다.
# =============================================================================

if __name__ == "__main__":

    print("=" * 65)
    print("단어 단위 토크나이저 테스트")
    print("핵심 주제: OOV 문제와 vocab_size 폭발")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # [실험 1] 기본 학습 및 인코딩
    # -------------------------------------------------------------------------
    print("\n[실험 1] 기본 학습 및 인코딩")
    print("-" * 50)

    tokenizer = WordTokenizer()

    # 간단한 영어 문장으로 학습합니다.
    train_text = "I love machine learning and natural language processing"
    print(f"학습 텍스트: '{train_text}'")
    tokenizer.train(train_text)

    # 학습 데이터에 있는 문장은 정상적으로 인코딩됩니다.
    test_text = "I love language"
    encoded = tokenizer.encode(test_text, add_special_tokens=True)
    decoded = tokenizer.decode(encoded)
    print(f"\n인코딩 대상: '{test_text}'")
    print(f"인코딩 결과: {encoded}")
    print(f"디코딩 결과: '{decoded}'")

    # -------------------------------------------------------------------------
    # [실험 2] OOV 문제 — 핵심!
    # 학습 데이터에 없는 단어가 등장하면 어떻게 되는가?
    # -------------------------------------------------------------------------
    print("\n[실험 2] OOV 문제 직접 경험하기")
    print("-" * 50)
    print("학습 데이터에 없는 단어들을 인코딩해 봅니다.")

    # 시나리오 A: 2023년 이후에 유명해진 단어들
    # → 2010년대 이전 텍스트로 학습한 모델은 이 단어들을 모름
    oov_text = "I love ChatGPT and transformer architecture"
    print(f"\n인코딩 대상: '{oov_text}'")
    print("  (학습 데이터에 없는 단어: 'ChatGPT', 'transformer', 'architecture')")

    report = tokenizer.encode_with_oov_report(oov_text)
    print(f"인코딩 결과: {report['ids']}")
    print(f"OOV 단어 목록: {report['oov_words']}")
    print(f"OOV 개수: {report['oov_count']}개 / 전체 {len(oov_text.split())}개 단어")
    print(f"OOV 비율: {report['oov_rate']:.1%}")
    print(f"결론: 전체 단어의 {report['oov_rate']:.1%}가 의미를 잃고 [UNK]로 처리됨!")

    # 시나리오 B: 디코딩하면 UNK가 들어간 채로 복원됨
    decoded_with_unk = tokenizer.decode(
        report['ids'], skip_special_tokens=False
    )
    print(f"\n디코딩 결과: '{decoded_with_unk}'")
    print("  → [UNK] 자리에 원래 단어가 무엇이었는지 알 수 없습니다.")

    # -------------------------------------------------------------------------
    # [실험 3] vocab_size 폭발 문제
    # 왜 단어 단위 어휘집은 커지지 않을 수 없는가?
    # -------------------------------------------------------------------------
    print("\n[실험 3] vocab_size 폭발 문제")
    print("-" * 50)

    # 영어 단어 수 vs 한국어 형태소 수를 수치로 보여줍니다.
    print("언어별 어휘집 크기 현실:")
    print()
    print(f"  영어 단어 사전 수:        약 170,000개")
    print(f"  한국어 기본 형태소 수:    약 200,000개")
    print(f"  한국어 활용형 포함 시:    수백만 개 이상")
    print()

    # 실제 학습 데이터 크기에 따른 vocab_size 변화를 보여줍니다.
    sample_texts = [
        "the cat sat on the mat",                                # 6 고유 단어
        "the dog ran across the park near the river",            # +5 새 단어
        "a quick brown fox jumped over the lazy dog",            # +7 새 단어
        "she sells sea shells by the sea shore",                 # +5 새 단어
        "how much wood would a woodchuck chuck if a woodchuck could chuck wood",  # +7 새 단어
    ]

    tokenizer_growing = WordTokenizer()
    print("텍스트를 추가할수록 vocab_size가 커지는 과정:")
    print()

    cumulative_text = ""
    for i, text in enumerate(sample_texts):
        cumulative_text += " " + text
        tokenizer_growing.train(cumulative_text.strip())
        print(
            f"  {i+1}번째 문장 추가 후 → vocab_size: {tokenizer_growing.vocab_size}개"
            f"  (특수 토큰 4개 + 단어 {tokenizer_growing.vocab_size - 4}개)"
        )

    print()
    print("  위 5개 문장만으로도 어휘집이 빠르게 커집니다.")
    print("  위키피디아 전체(약 60억 단어)로 학습하면?")
    print("  → 단어 수: 수백만 개 → 메모리 수십 GB 필요!")

    # -------------------------------------------------------------------------
    # [실험 4] 글자 단위 vs 단어 단위 비교
    # -------------------------------------------------------------------------
    print("\n[실험 4] 글자 단위 vs 단어 단위 비교")
    print("-" * 50)

    comparison_text = "I love natural language processing"

    # 단어 단위 토크나이저로 인코딩
    tok_word = WordTokenizer()
    tok_word.train(comparison_text)
    word_encoded = tok_word.encode(comparison_text)

    # 글자 단위로 직접 쪼개기 (비교용)
    char_encoded_len = len(comparison_text.replace(" ", ""))  # 공백 제외 글자 수
    word_encoded_len = len(word_encoded)

    print(f"비교 문장: '{comparison_text}'")
    print()
    print(f"  글자 단위: 약 {char_encoded_len}개 토큰 (글자 하나하나)")
    print(f"  단어 단위: {word_encoded_len}개 토큰 (단어 하나하나)")
    print()
    print(f"  → 단어 단위는 글자 단위보다 토큰 수가 약 {char_encoded_len // word_encoded_len}배 적습니다.")
    print(f"  → 하지만 어휘집은 훨씬 크고, OOV 문제가 있습니다.")

    # -------------------------------------------------------------------------
    # [실험 5] 저장 및 불러오기 테스트
    # -------------------------------------------------------------------------
    print("\n[실험 5] 어휘집 저장 및 불러오기 테스트")
    print("-" * 50)

    import os

    # 저장 경로를 이 파일과 같은 디렉터리로 설정합니다.
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "word_tokenizer.json")

    # 어휘집을 파일로 저장합니다.
    tok_save = WordTokenizer()
    tok_save.train("hello world foo bar baz")
    tok_save.save(save_path)

    # 새 토크나이저에 저장된 어휘집을 불러옵니다.
    tok_load = WordTokenizer()
    tok_load.load(save_path)

    # 동일하게 인코딩되는지 확인합니다.
    original_ids = tok_save.encode("hello world", add_special_tokens=True)
    loaded_ids = tok_load.encode("hello world", add_special_tokens=True)
    print(f"\n원본 토크나이저 인코딩:    {original_ids}")
    print(f"불러온 토크나이저 인코딩:  {loaded_ids}")
    print(f"결과 동일 여부: {original_ids == loaded_ids}")

    # -------------------------------------------------------------------------
    # [정리] 단어 단위 토크나이저의 두 가지 치명적 한계
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("[정리] 단어 단위 토크나이저의 두 가지 치명적 한계")
    print("=" * 65)
    print()
    print("  한계 1 - OOV(Out Of Vocabulary) 문제")
    print("    - 학습 데이터에 없는 단어는 무조건 [UNK]로 대체됨")
    print("    - 예: 'ChatGPT'처럼 새로운 단어는 처리 불가")
    print("    - 의미 있는 정보가 사라짐")
    print()
    print("  한계 2 - vocab_size 폭발 문제")
    print("    - 영어만 해도 단어 수: 약 170,000개")
    print("    - 한국어 활용형 포함 시: 수백만 개")
    print("    - vocab_size가 크면 모델 크기(메모리)도 비례해서 커짐")
    print()
    print("  -> 이 두 가지 문제를 동시에 해결한 것이 BPE입니다.")
    print("     BPE는 글자 단위의 완전성 + 단어 단위의 효율성을 모두 취합니다.")
    print("     다음 파일: basic_tokenizer.py")
