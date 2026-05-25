# -*- coding: utf-8 -*-
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Windows 터미널 한글 출력 보장
except AttributeError:
    pass  # Jupyter는 stdout이 OutStream이라 reconfigure 없음 — 무시
# =============================================================================
# char_tokenizer.py — 글자 단위 토크나이저 (Character-level Tokenizer)
#
# 목적:
#   문장을 글자 하나하나로 쪼개고, 각 글자에 숫자 ID를 부여합니다.
#   인코딩(문장 → 숫자 목록)과 디코딩(숫자 목록 → 문장)을 모두 구현합니다.
#
# 이 파일을 통해 배우는 것:
#   - 어휘집(Vocabulary)이 어떻게 만들어지는가
#   - 특수 토큰(Special Token)이 왜 필요한가
#   - 배치 처리(Batch Processing)에서 PAD가 하는 역할
#   - 어휘집을 파일로 저장하고 불러오는 이유
# =============================================================================

import json  # 어휘집을 JSON 파일로 저장/불러오기 위해 사용


# -----------------------------------------------------------------------------
# 특수 토큰(Special Token) 정의
#
# 왜 이 네 가지인가?
#   [PAD]: 배치 처리 시 문장 길이를 맞추기 위한 채우기 토큰
#   [UNK]: 어휘집에 없는 글자를 만났을 때 대체하는 미지(未知) 토큰
#   [BOS]: 문장의 시작을 모델에게 알리는 토큰 (Beginning Of Sequence)
#   [EOS]: 문장의 끝을 모델에게 알리는 토큰 (End Of Sequence)
#
# 왜 ID 0~3에 고정되어야 하는가?
#   모델 코드의 여러 곳에서 "PAD = 0번"이라는 사실을 직접 사용합니다.
#   예: Attention Mask를 만들 때 "0번 ID 위치는 무시하라"는 식으로 씁니다.
#   이 번호가 학습할 때마다 달라지면 모델 전체가 오동작합니다.
#   따라서 특수 토큰의 ID는 항상 낮은 번호에 고정해야 합니다.
# -----------------------------------------------------------------------------
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

# 각 특수 토큰의 고정 ID를 상수로 정의해 둡니다.
# 코드 어디서든 PAD_ID, UNK_ID 등으로 명확하게 참조할 수 있습니다.
PAD_ID = 0  # 채우기 토큰 — 배치 패딩에 사용
UNK_ID = 1  # 미지 토큰 — 어휘집에 없는 글자 대체
BOS_ID = 2  # 문장 시작 토큰
EOS_ID = 3  # 문장 끝 토큰


class CharTokenizer:
    """
    글자 단위 토크나이저 클래스 (Special Token 지원 버전)

    기본 사용 예시:
        tokenizer = CharTokenizer()
        tokenizer.train("hello world")
        ids = tokenizer.encode("hello", add_special_tokens=True)
        text = tokenizer.decode(ids)

    배치 처리 예시:
        results = tokenizer.encode_batch(
            ["hello", "hi"],
            add_special_tokens=True,
            pad_to_max_length=True
        )

    저장/불러오기 예시:
        tokenizer.save("tokenizer.json")
        tokenizer.load("tokenizer.json")
    """

    def __init__(self):
        # ---------------------------------------------------------------------
        # 어휘집 초기화: 특수 토큰을 먼저 등록합니다.
        #
        # 왜 특수 토큰을 먼저 등록하는가?
        #   ID 번호는 등록 순서대로 부여됩니다.
        #   특수 토큰이 항상 0, 1, 2, 3번을 차지하려면
        #   어떤 일반 글자보다 먼저 등록되어야 합니다.
        # ---------------------------------------------------------------------

        # 글자 → ID 변환 사전
        # 예: {'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3, 'a': 4, ...}
        self.char_to_id: dict[str, int] = {}

        # ID → 글자 변환 사전 (디코딩에 사용)
        # 예: {0: '[PAD]', 1: '[UNK]', 2: '[BOS]', 3: '[EOS]', 4: 'a', ...}
        self.id_to_char: dict[int, str] = {}

        # 어휘집에 등록된 토큰 총 개수
        self.vocab_size: int = 0

        # 특수 토큰을 0번부터 순서대로 등록합니다.
        for token in SPECIAL_TOKENS:
            self._register_token(token)

    def _register_token(self, token: str):
        """
        토큰 하나를 어휘집에 등록하는 내부 헬퍼 메서드입니다.
        (밑줄로 시작하는 메서드는 클래스 내부에서만 사용하는 약속입니다.)

        이미 등록된 토큰은 건너뜁니다.
        """
        if token not in self.char_to_id:
            new_id = self.vocab_size
            self.char_to_id[token] = new_id
            self.id_to_char[new_id] = token
            self.vocab_size += 1

    def train(self, text: str):
        """
        텍스트를 학습하여 어휘집(Vocabulary)에 글자를 등록합니다.

        학습이란?
            주어진 텍스트에 등장하는 모든 글자를 수집하고,
            아직 등록되지 않은 글자에 순서대로 ID를 부여하는 과정입니다.
            특수 토큰은 이미 0~3번에 고정되어 있으므로,
            일반 글자는 4번부터 시작합니다.

        매개변수:
            text (str): 학습에 사용할 텍스트

        예시:
            train("hello")
            → 특수 토큰: [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3
            → 일반 글자: 'e'=4, 'h'=5, 'l'=6, 'o'=7  (알파벳 정렬 기준)
        """
        # 텍스트에 등장하는 고유한 글자들을 수집합니다.
        # set()은 중복을 자동으로 제거합니다.
        unique_chars = set(text)

        # 정렬합니다. 같은 텍스트로 학습할 때 항상 같은 ID가 부여되도록 합니다.
        # (정렬하지 않으면 실행할 때마다 순서가 달라질 수 있습니다.)
        sorted_chars = sorted(unique_chars)

        for char in sorted_chars:
            # _register_token은 내부적으로 중복 여부를 확인합니다.
            self._register_token(char)

        print(f"학습 완료! 어휘집 크기: {self.vocab_size}개 토큰")
        print(f"  (특수 토큰 4개 + 일반 글자 {self.vocab_size - 4}개)")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        문장을 숫자 ID 목록으로 변환합니다. (인코딩)

        어휘집에 없는 글자는 ValueError 대신 [UNK] ID(=1)로 대체합니다.
        이렇게 하면 처음 보는 글자가 나와도 프로그램이 멈추지 않습니다.

        매개변수:
            text (str): 인코딩할 문장
            add_special_tokens (bool): True이면 앞에 [BOS], 뒤에 [EOS]를 자동 추가합니다.
                                       기본값은 False입니다.

        반환값:
            list[int]: 숫자 ID 목록

        예시:
            encode("hello")
            → [5, 4, 6, 6, 7]

            encode("hello", add_special_tokens=True)
            → [2, 5, 4, 6, 6, 7, 3]
               ↑              ↑
             [BOS]          [EOS]
        """
        ids = []

        # add_special_tokens=True이면 맨 앞에 [BOS] 토큰 ID를 추가합니다.
        # 모델이 "지금부터 새 문장이 시작된다"는 신호를 받습니다.
        if add_special_tokens:
            ids.append(BOS_ID)

        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                # 어휘집에 없는 글자 → [UNK] ID로 대체합니다.
                # 이전 버전에서는 ValueError를 발생시켰지만,
                # 실무에서는 에러 대신 [UNK]로 조용히 처리하는 것이 표준입니다.
                ids.append(UNK_ID)

        # add_special_tokens=True이면 맨 뒤에 [EOS] 토큰 ID를 추가합니다.
        # 모델이 "문장이 여기서 끝난다"는 신호를 받습니다.
        if add_special_tokens:
            ids.append(EOS_ID)

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        숫자 ID 목록을 문장으로 변환합니다. (디코딩)

        매개변수:
            ids (list[int]): 숫자 ID 목록
            skip_special_tokens (bool): True이면 [PAD], [BOS], [EOS] 같은
                                        특수 토큰을 결과 문자열에서 제외합니다.
                                        기본값은 True입니다.

        반환값:
            str: 복원된 문장

        예시:
            decode([2, 5, 4, 6, 6, 7, 3], skip_special_tokens=True)
            → "hello"   ← [BOS]와 [EOS]가 제거된 결과

            decode([2, 5, 4, 6, 6, 7, 3], skip_special_tokens=False)
            → "[BOS]hello[EOS]"
        """
        # 특수 토큰으로 사용되는 문자열 집합을 만듭니다.
        # skip_special_tokens=True일 때 이 목록에 있는 토큰은 건너뜁니다.
        special_token_set = set(SPECIAL_TOKENS)

        chars = []

        for id_ in ids:
            if id_ not in self.id_to_char:
                # 알 수 없는 ID는 [UNK] 문자열로 표시합니다.
                chars.append("[UNK]")
                continue

            token = self.id_to_char[id_]

            # 특수 토큰을 건너뛸지 여부를 확인합니다.
            if skip_special_tokens and token in special_token_set:
                continue

            chars.append(token)

        return ''.join(chars)

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
        pad_to_max_length: bool = False,
    ) -> list[list[int]]:
        """
        여러 문장을 한 번에 인코딩합니다. (배치 인코딩)

        AI 모델은 문장을 하나씩 처리하는 것보다 여러 개를 묶어서 처리하는 것이
        훨씬 빠릅니다. 이것을 배치 처리(Batch Processing)라고 합니다.

        배치 처리의 문제점:
            문장마다 길이가 다르면 하나의 직사각형 행렬로 묶을 수 없습니다.
            → 이때 짧은 문장의 빈 자리를 [PAD] 토큰으로 채웁니다.

        매개변수:
            texts (list[str]): 인코딩할 문장 목록
            add_special_tokens (bool): 각 문장에 [BOS], [EOS]를 추가할지 여부
            pad_to_max_length (bool): True이면 가장 긴 문장의 길이를 기준으로
                                      모든 문장을 [PAD]로 채웁니다.

        반환값:
            list[list[int]]: 각 문장의 ID 목록을 담은 2차원 목록

        예시:
            encode_batch(["hello", "hi"], add_special_tokens=True, pad_to_max_length=True)
            →
            [
              [2, 5, 4, 6, 6, 7, 3, 0],   ← "hello": [BOS] h e l l o [EOS] [PAD]
              [2, 5, 8, 3, 0, 0, 0, 0],    ← "hi":    [BOS] h i [EOS] [PAD] [PAD] [PAD] [PAD]
            ]
        """
        # 1단계: 각 문장을 개별적으로 인코딩합니다.
        encoded_list = [
            self.encode(text, add_special_tokens=add_special_tokens)
            for text in texts
        ]

        if pad_to_max_length and encoded_list:
            # 2단계: 가장 긴 문장의 길이를 구합니다.
            max_length = max(len(ids) for ids in encoded_list)

            # 3단계: 짧은 문장의 뒤쪽을 [PAD] ID(=0)로 채웁니다.
            # 왜 뒤쪽에 채우는가? 모델이 앞부분부터 읽으므로,
            # 실제 내용은 앞에, 패딩은 뒤에 두는 것이 일반적입니다.
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

        왜 저장해야 하는가?
            AI 모델을 학습할 때 사용한 어휘집과 나중에 추론할 때 사용하는 어휘집이
            반드시 동일해야 합니다. 학습할 때마다 어휘집을 새로 만들면
            ID 번호가 달라져서 모델이 틀린 답을 냅니다.
            저장된 파일을 불러오면 항상 동일한 어휘집을 사용할 수 있습니다.

        저장 형식 예시 (tokenizer.json):
            {
                "vocab_size": 12,
                "char_to_id": {"[PAD]": 0, "[UNK]": 1, ..., "h": 5, "e": 4},
                "special_tokens": ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
            }

        매개변수:
            path (str): 저장할 파일 경로 (예: "tokenizer.json")
        """
        # 저장할 데이터를 딕셔너리로 구성합니다.
        data = {
            "vocab_size": self.vocab_size,
            "char_to_id": self.char_to_id,
            # id_to_char는 char_to_id에서 복원할 수 있으므로 따로 저장하지 않아도 됩니다.
            # 하지만 JSON의 키는 문자열만 가능하므로, ID(정수)를 키로 하는
            # id_to_char는 저장하지 않고 load() 시 char_to_id에서 역으로 재구성합니다.
            "special_tokens": SPECIAL_TOKENS,
        }

        with open(path, "w", encoding="utf-8") as f:
            # ensure_ascii=False: 한글 등 비ASCII 문자를 이스케이프 없이 저장합니다.
            # indent=2: 사람이 읽기 쉽도록 들여쓰기를 추가합니다.
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"어휘집 저장 완료: {path}")
        print(f"  (총 {self.vocab_size}개 토큰)")

    def load(self, path: str):
        """
        JSON 파일에서 어휘집을 불러옵니다.

        주의사항:
            load()를 호출하면 현재 어휘집이 파일의 내용으로 완전히 교체됩니다.
            기존에 학습된 내용은 사라집니다.

        매개변수:
            path (str): 불러올 파일 경로 (예: "tokenizer.json")
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 파일에서 읽은 char_to_id를 그대로 복원합니다.
        self.char_to_id = data["char_to_id"]
        self.vocab_size = data["vocab_size"]

        # id_to_char는 char_to_id에서 역으로 재구성합니다.
        # JSON에서 읽으면 키가 문자열이므로, int()로 변환해야 합니다.
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}

        print(f"어휘집 불러오기 완료: {path}")
        print(f"  (총 {self.vocab_size}개 토큰)")

    def get_vocab(self) -> dict:
        """
        현재 어휘집(글자 → ID 사전)을 반환합니다.

        반환값:
            dict: 글자를 키로, ID를 값으로 하는 사전
        """
        return self.char_to_id


# =============================================================================
# 아래는 이 파일을 직접 실행할 때만 동작하는 테스트 코드입니다.
# "python char_tokenizer.py" 로 실행하면 결과를 확인할 수 있습니다.
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("글자 단위 토크나이저 테스트 (Special Token 지원 버전)")
    print("=" * 60)

    # --- 1단계: 기본 학습 및 인코딩 ---
    tokenizer = CharTokenizer()

    sample_text = "hello world"
    print(f"\n[1] 학습 텍스트: '{sample_text}'")
    tokenizer.train(sample_text)
    print(f"어휘집: {tokenizer.get_vocab()}")

    # --- 2단계: 특수 토큰 없이 인코딩 ---
    print("\n[2] 기본 인코딩 (특수 토큰 없음)")
    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)
    print(f"  입력:  'hello'")
    print(f"  인코딩: {encoded}")
    print(f"  디코딩: '{decoded}'")

    # --- 3단계: 특수 토큰 포함 인코딩 ---
    print("\n[3] 특수 토큰 포함 인코딩 (add_special_tokens=True)")
    encoded_with_special = tokenizer.encode("hello", add_special_tokens=True)
    decoded_with_special = tokenizer.decode(encoded_with_special, skip_special_tokens=False)
    decoded_clean = tokenizer.decode(encoded_with_special, skip_special_tokens=True)
    print(f"  인코딩 결과:                    {encoded_with_special}")
    print(f"  디코딩 (특수 토큰 포함):         '{decoded_with_special}'")
    print(f"  디코딩 (특수 토큰 제거):         '{decoded_clean}'")

    # --- 4단계: UNK 처리 테스트 ---
    print("\n[4] UNK 처리 테스트 (어휘집에 없는 글자 'z')")
    encoded_unk = tokenizer.encode("hz")
    print(f"  입력:  'hz'")
    print(f"  인코딩: {encoded_unk}  ← 'z'가 UNK(ID=1)로 대체됨")
    print(f"  디코딩: '{tokenizer.decode(encoded_unk, skip_special_tokens=False)}'")

    # --- 5단계: 배치 인코딩 테스트 ---
    print("\n[5] 배치 인코딩 테스트 (encode_batch)")
    texts = ["hello", "hi", "world"]
    batch_result = tokenizer.encode_batch(
        texts,
        add_special_tokens=True,
        pad_to_max_length=True,
    )
    print(f"  입력 문장: {texts}")
    print(f"  배치 인코딩 결과 (PAD로 길이 통일):")
    for text, ids in zip(texts, batch_result):
        print(f"    '{text}': {ids}")

    # --- 6단계: 저장 및 불러오기 테스트 ---
    print("\n[6] 어휘집 저장 및 불러오기 테스트")

    import os
    save_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")

    # 저장
    tokenizer.save(save_path)

    # 새 토크나이저에 불러오기
    new_tokenizer = CharTokenizer()
    new_tokenizer.load(save_path)

    # 불러온 어휘집으로 인코딩이 동일한지 확인
    original_ids = tokenizer.encode("hello", add_special_tokens=True)
    loaded_ids = new_tokenizer.encode("hello", add_special_tokens=True)
    print(f"  원본 토크나이저 인코딩:    {original_ids}")
    print(f"  불러온 토크나이저 인코딩:  {loaded_ids}")
    print(f"  결과 동일 여부: {original_ids == loaded_ids}")
