# -*- coding: utf-8 -*-
# =============================================================================
# basic_tokenizer.py — BPE 기본 토크나이저 (Basic BPE Tokenizer)
#
# 목적:
#   Byte Pair Encoding(BPE) 알고리즘을 처음부터 직접 구현합니다.
#   minbpe(Andrej Karpathy)의 BasicTokenizer를 참고하되,
#   모든 로직을 한국어 주석과 함께 처음부터 재구현했습니다.
#
# BPE의 핵심 아이디어:
#   1. 텍스트를 UTF-8 바이트 단위로 쪼갭니다. (0~255, 총 256가지)
#   2. 가장 자주 붙어 나오는 바이트 쌍(pair)을 찾습니다.
#   3. 그 쌍을 새로운 단일 토큰으로 합칩니다. (병합, merge)
#   4. 원하는 vocab_size에 도달할 때까지 2~3을 반복합니다.
#
# 왜 256 바이트에서 시작하는가?
#   UTF-8로 인코딩하면 어떤 언어, 어떤 문자도 결국 0~255 사이의
#   바이트들로 표현됩니다. 따라서 256개 기본 토큰으로 시작하면
#   절대로 [UNK]가 발생하지 않습니다.
#
# 참고: minbpe by Andrej Karpathy
#   https://github.com/karpathy/minbpe
#   단, 이 구현은 복붙이 아닌 한국어 주석과 함께 직접 재구현한 것입니다.
# =============================================================================

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Windows 터미널 한글/특수문자 출력 보장
except AttributeError:
    pass  # Jupyter OutStream은 reconfigure 없음


# =============================================================================
# 헬퍼 함수들 (Helper Functions)
# 클래스 외부의 독립 함수로 정의해, 어디서든 단독으로 테스트할 수 있게 합니다.
# =============================================================================

def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """
    ID 목록에서 인접한 쌍(pair)의 등장 횟수를 셉니다.

    BPE 알고리즘의 핵심 연산입니다.
    어떤 두 토큰이 가장 자주 붙어 나오는지 파악하기 위해 사용합니다.

    매개변수:
        ids (list[int]): 현재 토큰 ID들의 목록

    반환값:
        dict[tuple[int, int], int]:
            키: (앞 토큰 ID, 뒤 토큰 ID) 쌍
            값: 그 쌍이 등장한 횟수

    예시:
        ids = [1, 2, 1, 2, 3]
        get_stats(ids)
        →  {(1, 2): 2,   ← (1, 2) 쌍이 2번 나옴
             (2, 1): 1,   ← (2, 1) 쌍이 1번 나옴
             (2, 3): 1}   ← (2, 3) 쌍이 1번 나옴

    시각화:
        ids = [1, 2, 1, 2, 3]
               ↑↑  ↑↑  ↑↑
              (1,2)(2,1)(1,2)(2,3)  ← 슬라이딩 윈도우로 인접 쌍 추출
    """
    # 쌍 → 횟수를 저장할 딕셔너리입니다.
    counts = {}

    # zip(ids, ids[1:])은 인접한 두 원소를 쌍으로 묶어줍니다.
    # 예: [1, 2, 3] → (1,2), (2,3)
    for pair in zip(ids, ids[1:]):
        # 딕셔너리에 없는 쌍이면 0을 기본값으로 두고, 1을 더합니다.
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """
    ID 목록에서 특정 쌍(pair)을 찾아 새로운 단일 ID(new_id)로 교체합니다.

    BPE 병합(merge) 연산의 핵심 함수입니다.
    get_stats()로 찾은 가장 빈번한 쌍을 실제로 합치는 역할을 합니다.

    매개변수:
        ids (list[int]): 현재 토큰 ID 목록
        pair (tuple[int, int]): 합칠 쌍 (앞 ID, 뒤 ID)
        new_id (int): 합친 결과에 부여할 새 ID

    반환값:
        list[int]: pair가 new_id로 교체된 새로운 ID 목록

    예시:
        ids = [1, 2, 3, 1, 2]
        merge(ids, pair=(1, 2), new_id=4)
        → [4, 3, 4]   ← (1,2)가 나타난 두 곳이 모두 4로 교체됨

    중요:
        원본 ids를 수정하지 않고 새로운 목록을 반환합니다.
        (함수형 프로그래밍 스타일: 부작용 없음)
    """
    # 병합 결과를 담을 새 목록입니다.
    new_ids = []

    # 인덱스를 직접 관리하면서 순회합니다.
    # (pair를 만나면 두 원소를 건너뛰어야 하므로 for-each 대신 while 사용)
    i = 0
    while i < len(ids):
        # 현재 위치에서 pair와 일치하는지 확인합니다.
        # 조건 1: 마지막 원소가 아니어야 합니다. (다음 원소가 있어야 pair 비교 가능)
        # 조건 2: 현재 원소와 다음 원소가 pair와 정확히 같아야 합니다.
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            # pair를 발견했습니다 → new_id 하나로 교체하고 두 칸 전진합니다.
            new_ids.append(new_id)
            i += 2  # pair는 두 원소이므로 두 칸 건너뜁니다.
        else:
            # pair가 아닙니다 → 현재 원소를 그대로 유지하고 한 칸 전진합니다.
            new_ids.append(ids[i])
            i += 1

    return new_ids


# =============================================================================
# BasicTokenizer 클래스
# =============================================================================

class BasicTokenizer:
    """
    BPE(Byte Pair Encoding) 기본 토크나이저

    작동 원리:
        1. 텍스트를 UTF-8 바이트 목록으로 변환합니다.
        2. 가장 자주 나오는 바이트 쌍을 찾아 새 ID로 합칩니다.
        3. 원하는 vocab_size에 도달할 때까지 반복합니다.
        4. 학습된 병합 규칙을 encode() 시에 순서대로 적용합니다.

    어휘집 구조:
        - ID   0 ~ 255: 원시 바이트 토큰 (초기 고정)
        - ID 256 ~ ...: 병합으로 생성된 새 토큰

    한계 (BasicTokenizer):
        단어 경계를 무시합니다. "New York"에서 'k'와 'N'이 합쳐질 수도 있습니다.
        이 문제는 RegexTokenizer(다음 단계)에서 해결합니다.
    """

    def __init__(self):
        # -----------------------------------------------------------------------
        # 병합 규칙 사전: {(앞 ID, 뒤 ID): 새 ID}
        #
        # 학습(train) 시 채워지고, 인코딩(encode) 시 순서대로 적용됩니다.
        # 순서가 매우 중요합니다 — 먼저 학습된 병합이 먼저 적용되어야 합니다.
        # -----------------------------------------------------------------------
        self.merges: dict[tuple[int, int], int] = {}

        # -----------------------------------------------------------------------
        # 어휘집 사전: {ID: bytes}
        #
        # ID를 실제 바이트열(bytes)로 변환하는 표입니다.
        # 디코딩 시 ID → bytes → UTF-8 문자열 순서로 변환합니다.
        # -----------------------------------------------------------------------
        self.vocab: dict[int, bytes] = {}

        # 초기 어휘집(0~255)을 즉시 구축합니다.
        self._build_vocab()

    def _build_vocab(self):
        """
        초기 어휘집을 구축합니다.

        ID 0~255: 각각 1바이트짜리 bytes 객체로 설정합니다.
        예: 0 → b'\x00', 65 → b'A', 104 → b'h'

        병합 토큰은 나중에 train()에서 추가됩니다.
        병합 토큰의 bytes 값 = 두 부모 토큰의 bytes를 이어 붙인 것입니다.
        예: 'he' 토큰(ID=256) = vocab[104] + vocab[101] = b'h' + b'e' = b'he'
        """
        # 0~255 기본 바이트 토큰을 초기화합니다.
        # bytes([n])은 n을 값으로 갖는 1바이트짜리 bytes 객체를 만듭니다.
        for i in range(256):
            self.vocab[i] = bytes([i])

        # 이미 저장된 병합 규칙이 있으면 어휘집에 반영합니다.
        # (load() 후 _build_vocab()을 다시 호출하는 경우에 대비)
        for (p0, p1), new_id in self.merges.items():
            # 병합 토큰의 bytes = 두 부모 bytes를 이어 붙임
            self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        텍스트로 BPE 병합 규칙을 학습합니다.

        학습 과정:
            1. text를 UTF-8 바이트 목록으로 변환합니다.
            2. vocab_size에 도달할 때까지 다음을 반복합니다:
               a. 가장 빈번한 인접 쌍을 get_stats()로 찾습니다.
               b. 그 쌍에 새 ID를 부여합니다. (256번부터 시작)
               c. 병합 규칙을 self.merges에 저장합니다.
               d. 어휘집에 새 토큰을 추가합니다.
               e. merge()로 현재 ID 목록에 병합을 적용합니다.

        매개변수:
            text (str): 학습에 사용할 텍스트
            vocab_size (int): 최종 어휘집 크기 (최소 256 이상이어야 합니다)
            verbose (bool): True이면 병합 과정을 단계별로 출력합니다.

        주의:
            vocab_size < 256이면 BPE 병합을 한 번도 하지 않습니다.
            실용적인 범위: 256 + 수십 ~ 수천 (GPT-2는 50,257)
        """
        # 최소 vocab_size 검증
        if vocab_size < 256:
            raise ValueError(
                f"vocab_size는 최소 256이어야 합니다. (현재: {vocab_size})\n"
                f"이유: 0~255 기본 바이트 토큰이 반드시 필요합니다."
            )

        # 수행할 병합 횟수를 계산합니다.
        # 예: vocab_size=300이면 300-256=44번 병합합니다.
        num_merges = vocab_size - 256

        # -----------------------------------------------------------------------
        # 1단계: 텍스트를 UTF-8 바이트 목록으로 변환합니다.
        #
        # text.encode("utf-8")은 str → bytes 변환입니다.
        # list(...)는 bytes를 각 바이트의 정수값 목록으로 변환합니다.
        #
        # 예: "hello" → b'hello' → [104, 101, 108, 108, 111]
        # 예: "안" → b'\xec\x95\x88' → [236, 149, 136]  (3바이트!)
        # -----------------------------------------------------------------------
        ids = list(text.encode("utf-8"))

        if verbose:
            print(f"학습 시작")
            print(f"  텍스트 길이: {len(text)}글자 → {len(ids)}바이트")
            print(f"  현재 vocab_size: 256")
            print(f"  목표 vocab_size: {vocab_size}")
            print(f"  수행할 병합 횟수: {num_merges}회")
            print()

        # -----------------------------------------------------------------------
        # 2단계: 병합을 num_merges번 반복합니다.
        # -----------------------------------------------------------------------
        for i in range(num_merges):

            # a. 현재 ID 목록에서 인접 쌍의 빈도수를 셉니다.
            stats = get_stats(ids)

            if not stats:
                # 더 이상 합칠 쌍이 없으면 (텍스트가 너무 짧으면) 중단합니다.
                if verbose:
                    print(f"  병합 {i+1}회차: 더 이상 합칠 쌍이 없습니다. 학습 종료.")
                break

            # b. 가장 자주 나오는 쌍을 선택합니다.
            # max()에 key=stats.get을 전달하면 값(빈도수)이 가장 큰 키(쌍)를 반환합니다.
            best_pair = max(stats, key=stats.get)
            best_count = stats[best_pair]

            # c. 새 ID를 부여합니다. (256번부터 순서대로)
            new_id = 256 + i

            # d. 병합 규칙을 저장합니다.
            self.merges[best_pair] = new_id

            # e. 어휘집에 새 토큰을 추가합니다.
            # 새 토큰의 bytes = 두 부모 토큰의 bytes를 이어 붙인 것입니다.
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # f. 현재 ID 목록에 병합을 실제로 적용합니다.
            ids = merge(ids, best_pair, new_id)

            if verbose:
                # 새 토큰이 어떤 문자(열)인지 사람이 읽기 쉽게 출력합니다.
                # errors='replace': 출력 불가능한 바이트는 '?'로 대체합니다.
                token_str = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"  병합 {i+1:4d}회차: "
                    f"({best_pair[0]:3d}, {best_pair[1]:3d}) → {new_id:5d}  "
                    f"| 빈도: {best_count:5d}회  "
                    f"| 새 토큰: '{token_str}'  "
                    f"| 남은 ID 수: {len(ids)}"
                )

        if verbose:
            print()
            print(f"학습 완료!")
            print(f"  최종 vocab_size: {len(self.vocab)}")
            print(f"  병합 규칙 수: {len(self.merges)}개")
            print(f"  최종 ID 목록 길이: {len(ids)} (원본 {len(text.encode('utf-8'))}바이트 → 압축됨)")

    def encode(self, text: str) -> list[int]:
        """
        텍스트를 ID 목록으로 변환합니다. (인코딩)

        인코딩 과정:
            1. text를 UTF-8 바이트 목록으로 변환합니다.
            2. 학습된 병합 규칙(self.merges)을 등록된 순서대로 적용합니다.
               순서가 중요합니다 — 먼저 학습된 병합을 먼저 적용해야 합니다.
            3. 더 이상 적용할 수 있는 병합이 없으면 종료합니다.

        왜 순서대로 적용해야 하는가?
            병합 규칙 A: (104, 101) → 256  ('h'+'e' → 'he')
            병합 규칙 B: (256, 108) → 257  ('he'+'l' → 'hel')

            만약 B를 먼저 적용하려 해도, 256번 토큰이 아직 없으므로 적용 안 됨.
            A를 먼저 적용해야 256이 생기고, 그 다음 B를 적용할 수 있습니다.

        매개변수:
            text (str): 인코딩할 문자열

        반환값:
            list[int]: 토큰 ID 목록
        """
        # 1단계: text → UTF-8 바이트 목록
        ids = list(text.encode("utf-8"))

        # 2단계: 병합 규칙이 없으면 (학습 전이면) 바이트 목록 그대로 반환합니다.
        if not self.merges:
            return ids

        # 3단계: 병합 규칙을 순서대로 적용합니다.
        # 병합이 적용될 수 없을 때까지(ID 목록 길이가 줄어드는 동안) 반복합니다.
        while len(ids) >= 2:
            # 현재 ID 목록의 모든 인접 쌍의 빈도를 셉니다.
            stats = get_stats(ids)

            # 현재 남아있는 쌍들 중 self.merges에 등록된 것을 찾습니다.
            # 그 중 가장 먼저 학습된 것(ID 번호가 가장 작은 것)을 선택합니다.
            #
            # 왜 가장 먼저 학습된 것을 선택하는가?
            #   학습 순서대로 적용해야 올바른 결과가 나옵니다.
            #   ID 번호가 작을수록 먼저 학습된 병합입니다. (256번이 첫 번째)
            pair_to_merge = min(
                stats,
                key=lambda p: self.merges.get(p, float("inf"))
                # 등록된 쌍은 해당 ID, 등록 안 된 쌍은 무한대로 처리
            )

            # 선택된 쌍이 self.merges에 없으면 더 이상 적용할 병합이 없습니다.
            if pair_to_merge not in self.merges:
                break

            # 선택된 쌍을 실제로 병합합니다.
            new_id = self.merges[pair_to_merge]
            ids = merge(ids, pair_to_merge, new_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        ID 목록을 텍스트로 변환합니다. (디코딩)

        디코딩 과정:
            1. 각 ID를 self.vocab에서 bytes로 변환합니다.
            2. 모든 bytes를 이어 붙입니다.
            3. bytes를 UTF-8 문자열로 변환합니다.

        UTF-8 안전성:
            바이트를 이어 붙인 후 한 번에 디코딩해야 합니다.
            한글처럼 다바이트(multi-byte) 문자는 바이트가 여러 ID에 걸쳐
            나뉘어 있을 수 있어서, 각 ID를 따로 디코딩하면 오류가 납니다.

        매개변수:
            ids (list[int]): 토큰 ID 목록

        반환값:
            str: 복원된 문자열
        """
        # 1단계: 각 ID를 bytes로 변환하고 모두 이어 붙입니다.
        # b"".join(...)은 bytes 목록을 하나의 bytes로 이어 붙입니다.
        raw_bytes = b"".join(self.vocab[i] for i in ids)

        # 2단계: bytes를 UTF-8 문자열로 변환합니다.
        # errors="replace": 유효하지 않은 바이트 시퀀스는 '?'로 대체합니다.
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, file_prefix: str):
        """
        학습된 병합 규칙과 어휘집을 파일로 저장합니다.

        저장 파일:
            {file_prefix}.model — 병합 규칙 (텍스트 형식, 핵심 파일)
            {file_prefix}.vocab — 어휘집 (사람이 읽기 위한 참고용)

        .model 파일 형식:
            첫 번째 줄: "minbpe v1"  (버전 표시)
            이후 각 줄: "앞ID 뒤ID"  (병합 규칙, 학습 순서대로)

        .vocab 파일 형식:
            각 줄: "[토큰 문자열] ID"

        매개변수:
            file_prefix (str): 저장 파일 이름의 접두사
                예: "my_tokenizer" → "my_tokenizer.model", "my_tokenizer.vocab"
        """
        # .model 파일 저장 (병합 규칙 — 로드 시 필요한 핵심 정보)
        model_file = file_prefix + ".model"
        with open(model_file, "w", encoding="utf-8") as f:
            # 버전 표시 (minbpe 형식 호환)
            f.write("minbpe v1\n")

            # 병합 규칙을 학습 순서대로 저장합니다.
            # self.merges는 Python 3.7+ 딕셔너리라 삽입 순서가 보장됩니다.
            for (p0, p1), _ in self.merges.items():
                f.write(f"{p0} {p1}\n")

        # .vocab 파일 저장 (어휘집 — 사람이 확인하기 위한 참고용)
        vocab_file = file_prefix + ".vocab"
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token_id, token_bytes in self.vocab.items():
                # bytes를 사람이 읽기 쉬운 문자열로 변환합니다.
                token_str = token_bytes.decode("utf-8", errors="replace")
                f.write(f"{token_str!r} {token_id}\n")

        print(f"저장 완료:")
        print(f"  병합 규칙: {model_file}  ({len(self.merges)}개 규칙)")
        print(f"  어휘집:    {vocab_file}  ({len(self.vocab)}개 토큰)")

    def load(self, model_file: str):
        """
        .model 파일에서 병합 규칙을 불러옵니다.

        불러온 후에는 encode()와 decode()를 바로 사용할 수 있습니다.

        매개변수:
            model_file (str): 불러올 .model 파일 경로
        """
        # 기존 병합 규칙과 어휘집을 초기화합니다.
        self.merges = {}
        self.vocab = {}

        # 0~255 기본 어휘집을 먼저 복원합니다.
        for i in range(256):
            self.vocab[i] = bytes([i])

        with open(model_file, "r", encoding="utf-8") as f:
            # 첫 번째 줄: 버전 확인 ("minbpe v1")
            version = f.readline().strip()
            assert version == "minbpe v1", f"알 수 없는 파일 형식: {version}"

            # 이후 줄들: 병합 규칙 (학습 순서대로)
            new_id = 256  # 병합 토큰 ID는 256번부터 시작합니다.
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 빈 줄은 건너뜁니다.

                # "p0 p1" 형식으로 저장된 줄을 파싱합니다.
                p0, p1 = map(int, line.split())

                # 병합 규칙 복원
                self.merges[(p0, p1)] = new_id

                # 어휘집에 병합 토큰 추가
                self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]

                new_id += 1

        print(f"불러오기 완료: {model_file}")
        print(f"  병합 규칙: {len(self.merges)}개")
        print(f"  vocab_size: {len(self.vocab)}")


# =============================================================================
# 아래는 이 파일을 직접 실행할 때만 동작하는 테스트 코드입니다.
# "python basic_tokenizer.py" 로 실행하면 결과를 확인할 수 있습니다.
# =============================================================================

if __name__ == "__main__":

    print("=" * 65)
    print("BPE 기본 토크나이저 (BasicTokenizer) 테스트")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # [실험 1] get_stats() 헬퍼 함수 동작 확인
    # -------------------------------------------------------------------------
    print("\n[실험 1] get_stats() — 인접 쌍 빈도 계산")
    print("-" * 50)

    sample_ids = [1, 2, 1, 2, 3, 1, 2]
    stats = get_stats(sample_ids)
    print(f"입력: {sample_ids}")
    print(f"쌍 빈도: {stats}")
    print(f"가장 빈번한 쌍: {max(stats, key=stats.get)}  (빈도: {max(stats.values())})")

    # -------------------------------------------------------------------------
    # [실험 2] merge() 헬퍼 함수 동작 확인
    # -------------------------------------------------------------------------
    print("\n[실험 2] merge() — 쌍을 새 ID로 교체")
    print("-" * 50)

    ids_before = [1, 2, 3, 1, 2, 4]
    merged = merge(ids_before, pair=(1, 2), new_id=10)
    print(f"병합 전: {ids_before}")
    print(f"병합 규칙: (1, 2) → 10")
    print(f"병합 후: {merged}")
    print(f"  → (1, 2)가 나타난 위치가 모두 10으로 교체됨")

    # -------------------------------------------------------------------------
    # [실험 3] BPE 병합 과정 단계별 시각화
    # "aaabdaaabac" 예제 (BPE 설명에 자주 쓰이는 고전 예시)
    # -------------------------------------------------------------------------
    print("\n[실험 3] BPE 병합 과정 단계별 시각화")
    print("-" * 50)
    print("예제 문자열: 'aaabdaaabac'")
    print("(각 글자를 UTF-8 바이트로 변환한 뒤 병합 과정을 추적합니다.)\n")

    # 텍스트를 바이트로 변환합니다.
    demo_text = "aaabdaaabac"
    demo_ids = list(demo_text.encode("utf-8"))

    # 'a'=97, 'b'=98, 'c'=99, 'd'=100
    print(f"초기 ID 목록: {demo_ids}")
    print(f"  (a=97, b=98, c=99, d=100)\n")

    # 각 단계를 직접 수행하며 출력합니다.
    current_ids = demo_ids[:]

    for step in range(3):  # 3번의 병합만 수행합니다.
        stats = get_stats(current_ids)
        if not stats:
            break

        best = max(stats, key=stats.get)
        new_id = 256 + step

        # 어떤 바이트(문자)인지 보여줍니다.
        char0 = chr(best[0]) if best[0] < 128 else f"#{best[0]}"
        char1 = chr(best[1]) if best[1] < 128 else f"#{best[1]}"

        print(f"  {step+1}번째 병합:")
        print(f"    가장 빈번한 쌍: ({best[0]}, {best[1]}) = ('{char0}', '{char1}')  "
              f"→ 빈도: {stats[best]}회")
        print(f"    새 ID 부여: {new_id} = '{char0}{char1}'")

        current_ids = merge(current_ids, best, new_id)
        print(f"    병합 후 ID 목록: {current_ids}")
        print(f"    목록 길이: {len(current_ids)}\n")

    # -------------------------------------------------------------------------
    # [실험 4] 실제 텍스트로 BPE 학습 (verbose=True)
    # -------------------------------------------------------------------------
    print("\n[실험 4] 실제 텍스트로 BPE 학습 (vocab_size=280)")
    print("-" * 50)

    # 짧은 영어 단락으로 학습합니다.
    training_text = (
        "the quick brown fox jumps over the lazy dog. "
        "the dog barked at the fox. the fox ran away quickly. "
        "a quick brown dog jumps over the lazy fox. "
        "learning is the key to understanding language models. "
        "language models learn from large amounts of text data. "
        "the more text they see, the better they understand language."
    )

    tokenizer = BasicTokenizer()
    tokenizer.train(training_text, vocab_size=280, verbose=True)

    # -------------------------------------------------------------------------
    # [실험 5] 인코딩 → 디코딩 왕복 테스트 (완벽 복원 확인)
    # -------------------------------------------------------------------------
    print("\n[실험 5] 인코딩 → 디코딩 왕복 테스트")
    print("-" * 50)

    test_sentences = [
        "the quick brown fox",           # 학습 데이터와 동일한 문장
        "the lazy dog jumps quickly",    # 학습 데이터 단어 조합
        "hello world",                   # 학습 데이터에 없는 단어
        "언어 모델",                      # 한국어 (다바이트 UTF-8)
    ]

    all_passed = True
    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)

        # 원본과 복원된 문자열이 완전히 같아야 합니다.
        is_correct = (decoded == sentence)
        if not is_correct:
            all_passed = False

        status = "통과" if is_correct else "실패"
        print(f"  [{status}] '{sentence}'")
        print(f"         인코딩: {encoded[:8]}{'...' if len(encoded) > 8 else ''}")
        print(f"         토큰 수: {len(encoded)}개  "
              f"(원본 바이트: {len(sentence.encode('utf-8'))}개)")
        print()

    if all_passed:
        print("  모든 왕복 테스트 통과! encode → decode가 완벽히 복원됩니다.")
    else:
        print("  일부 테스트 실패. 위 결과를 확인해 주세요.")

    # -------------------------------------------------------------------------
    # [실험 6] BPE의 압축 효과 측정
    # -------------------------------------------------------------------------
    print("\n[실험 6] BPE의 압축 효과 측정")
    print("-" * 50)
    print("학습 텍스트를 직접 인코딩했을 때 길이 변화:\n")

    original_bytes = len(training_text.encode("utf-8"))
    encoded_ids = tokenizer.encode(training_text)
    compressed_len = len(encoded_ids)
    compression_ratio = original_bytes / compressed_len

    print(f"  원본 바이트 수:   {original_bytes:5d}개")
    print(f"  BPE 인코딩 후:    {compressed_len:5d}개 토큰")
    print(f"  압축 비율:        {compression_ratio:.2f}배")
    print(f"  (vocab_size가 클수록 압축 비율이 높아집니다.)")

    # -------------------------------------------------------------------------
    # [실험 7] 저장 및 불러오기 테스트
    # -------------------------------------------------------------------------
    print("\n[실험 7] 저장 및 불러오기 테스트")
    print("-" * 50)

    import os

    # 저장 경로를 이 파일과 같은 디렉터리로 설정합니다.
    save_prefix = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "basic_tokenizer"
    )

    # 저장
    tokenizer.save(save_prefix)

    # 새 토크나이저에 불러오기
    tokenizer2 = BasicTokenizer()
    tokenizer2.load(save_prefix + ".model")

    # 동일하게 인코딩되는지 확인합니다.
    test_text = "the quick brown fox"
    ids_original = tokenizer.encode(test_text)
    ids_loaded = tokenizer2.encode(test_text)
    print(f"\n원본 토크나이저 인코딩:    {ids_original}")
    print(f"불러온 토크나이저 인코딩:  {ids_loaded}")
    print(f"결과 동일 여부: {ids_original == ids_loaded}")

    # -------------------------------------------------------------------------
    # [정리] BPE vs 글자 단위 vs 단어 단위
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("[정리] 세 가지 토크나이저 비교")
    print("=" * 65)
    print()
    compare_text = "the quick brown fox jumps over the lazy dog"
    char_count = len(compare_text)                       # 글자 단위
    word_count = len(compare_text.split())               # 단어 단위
    bpe_count = len(tokenizer.encode(compare_text))      # BPE

    print(f"비교 문장: '{compare_text}'")
    print()
    print(f"  글자 단위:  {char_count:3d}개 토큰  (공백 포함 글자 수)")
    print(f"  단어 단위:  {word_count:3d}개 토큰  (공백으로 분리)")
    print(f"  BPE:        {bpe_count:3d}개 토큰  (vocab_size=280 기준)")
    print()
    print(f"  BPE는 글자 단위보다 훨씬 짧고, 단어 단위와 비슷한 효율을 냅니다.")
    print(f"  동시에 OOV 문제가 없습니다. (어떤 텍스트도 바이트로 표현 가능)")
    print()
    print(f"  다음 단계: RegexTokenizer — 단어 경계를 인식하는 개선된 BPE")
    print(f"  (docs/03b_regex_tokenizer_preview.md 참고)")
