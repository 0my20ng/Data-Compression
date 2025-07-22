"""
Universal Decoder (RLE · Huffman · LZ77)
=======================================
압축 프로그램과 완벽히 호환되는 해제기.
압축 파일 포맷
--------------
• `.rle`  : [value 1][count 4]...
• `.huff` : [pad 1][code_len 256][sym_order N][payload bits]
• `.lz77` : [offset 2][len 4bit(‑3)][next 1]
"""

from pathlib import Path
import sys

# ---------- RLE ----------

def rle_decode(buf: bytes) -> bytes:
    out, i = bytearray(), 0
    while i < len(buf):
        val = buf[i]
        cnt = int.from_bytes(buf[i+1:i+5], 'big')
        out.extend([val]*cnt)
        i += 5
    return bytes(out)

# ---------- Huffman ----------
class Node:
    __slots__ = ("l", "r", "s")

    def __init__(self):
        self.l = self.r = None
        self.s = None  # 리프 노드일 경우 심볼

    def walk(self, b):
        """비트 문자열 한 글자('0' 또는 '1')를 따라 다음 노드로 이동"""
        return self.l if b == '0' else self.r



def build_tree(code_len, sym_order):
    """code_len(256) + sym_order(N) → Canonical Huffman 트리 복원"""
    # 1) code_len > 0 인 심볼만 취사선택 후 Canonical 규칙 정렬
    symbols = [(code_len[s], s) for s in sym_order if code_len[s]]
    symbols.sort(key=lambda x: (x[0], x[1]))  # (길이↑, 심볼값↑)

    root = Node()
    code = 0
    prev_len = 0

    for length, sym in symbols:
        # 길이 변화만큼 코드 좌시프트 → Canonical 증분 방식
        code <<= (length - prev_len)
        bits = f"{code:0{length}b}"
        node = root
        for b in bits:
            if b == "0":
                if node.l is None:
                    node.l = Node()
                node = node.l
            else:
                if node.r is None:
                    node.r = Node()
                node = node.r
        node.s = sym  # 리프에 심볼 기록
        code += 1
        prev_len = length

    return root

def huff_decode(buf: bytes) -> bytes:
    pad = buf[0]
    code_len = list(buf[1:257])
    sym_cnt = sum(1 for l in code_len if l)
    sym_order = list(buf[257:257+sym_cnt])
    payload = buf[257+sym_cnt:]
    bits = bin(int.from_bytes(payload,'big'))[2:].zfill(len(payload)*8)
    if pad: bits = bits[:-pad]
    root = build_tree(code_len, sym_order)
    out, node = bytearray(), root
    for b in bits:
        node = node.walk(b)
        if node.s is not None:
            out.append(node.s)
            node = root
    return bytes(out)

# ---------- LZ77 ----------

def lz77_decode(buf: bytes) -> bytes:
    out, i = bytearray(), 0
    while i < len(buf):
        token = int.from_bytes(buf[i:i+2],'big')
        length = (token & 0xF) + 3
        off = token >> 4
        nxt = buf[i+2]
        i += 3
        if off == 0:
            out.append(nxt)
        else:
            start = len(out) - off
            for _ in range(length):
                out.append(out[start]); start += 1
            out.append(nxt)
    return bytes(out.rstrip(b"\x00"))

# ---------- LZW ----------

def lzw_decode(buf: bytes) -> bytes:
    """
    ↳ 2 바이트 고정폭 코드를 그대로 읽어 들이는 간단 LZW 복원기
       ‑ 인코더와 동일하게 초기 사전 크기 256, 최대 65 535
    """
    if not buf:
        return b''

    # 1) 2‑byte 코드 스트림을 정수 리스트로 변환
    if len(buf) % 2:
        raise ValueError("LZW 스트림 길이가 2의 배수가 아닙니다.")
    codes = [int.from_bytes(buf[i:i+2], 'big') for i in range(0, len(buf), 2)]

    # 2) 표준 LZW 디코딩
    dict_size = 256
    dictionary = {i: bytes([i]) for i in range(dict_size)}

    w = dictionary[codes[0]]
    out = bytearray(w)

    for k in codes[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:          # 특수 case: (KwKwK) 패턴
            entry = w + w[:1]
        else:
            raise ValueError(f"잘못된 LZW 코드: {k}")

        out.extend(entry)

        # 새 항목 추가
        dictionary[dict_size] = w + entry[:1]
        dict_size += 1
        if dict_size > 0xFFFF:        # 65 535 코드 한계
            dict_size = 256           # 사전 리셋(간단 처리)

        w = entry

    return bytes(out)

# ---------- CLI ----------

def main():
    path = Path(input('압축 파일(.rle/.huff/.lz77/.lzw) 경로: ').strip())
    if not path.is_file():
        print('[오류] 파일 없음'); return
    raw = path.read_bytes()
    ext = path.suffix.lower()
    try:
        if ext == '.rle':
            data = rle_decode(raw)
        elif ext == '.huff':
            data = huff_decode(raw)
        elif ext == '.lz77':
            data = lz77_decode(raw)
        elif ext == '.lzw':
            data = lzw_decode(raw)
        else:
            print('[오류] 지원 안함'); return
    except Exception as e:
        print('[해제 실패]', e); return

    out = path.with_suffix('.restored')
    out.write_bytes(data)
    print('✅ 복원 완료 →', out)

if __name__ == '__main__':
    main()
