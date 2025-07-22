import math, heapq, argparse, json
from collections import Counter, namedtuple
from pathlib import Path
from typing import Dict

# -------------------------------------------------
# 0‑A. RLE ENCODER
# -------------------------------------------------

def rle_encode(data: bytes) -> bytes:
    """Run‑Length Encoding (value + 4‑byte count)"""
    if not data:
        return b''
    out, prev, cnt = bytearray(), data[0], 1
    for b in data[1:]:
        if b == prev and cnt < 2**32 - 1:
            cnt += 1
        else:
            out.append(prev)
            out.extend(cnt.to_bytes(4, 'big'))
            prev, cnt = b, 1
    out.append(prev)
    out.extend(cnt.to_bytes(4, 'big'))
    return bytes(out)

# -------------------------------------------------
# 0‑B. HUFFMAN ENCODER (고정 포맷, 순서 보존)
# -------------------------------------------------
Node = namedtuple('Node', 'freq sym left right')
class HeapNode(Node):
    def __lt__(self, other):
        return self.freq < other.freq

def _build_tree(data: bytes):
    freq = Counter(data)
    heap = [HeapNode(f, b, None, None) for b, f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:  # 단일 심볼 특수 처리
        node = heapq.heappop(heap)
        return HeapNode(node.freq, None, node, None)
    while len(heap) > 1:
        l, r = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, HeapNode(l.freq + r.freq, None, l, r))
    return heap[0]

def _gen_codes(node, prefix='', cmap=None, order=None):
    if cmap is None:
        cmap = {}
    if order is None:
        order = []
    if node.sym is not None:  # 리프
        cmap[node.sym] = prefix or '0'  # 최소 1비트 보장
        order.append(node.sym)
    else:
        _gen_codes(node.left, prefix + '0', cmap, order)
        _gen_codes(node.right, prefix + '1', cmap, order)
    return cmap, order

def huff_encode(data: bytes) -> bytes:
    if not data:
        return b''

    # 1) 트리 → 임시 codes
    tree = _build_tree(data)
    codes, _ = _gen_codes(tree)

    # 2) Canonical 정렬 리스트
    canon_list = sorted(
        ((len(c), sym) for sym, c in codes.items()),
        key=lambda x: (x[0], x[1])
    )

    # 3) code_len 테이블
    code_len = bytearray(256)
    for length, sym in canon_list:
        code_len[sym] = length

    # 4‑a) sym_order (캐논컬 순) 저장
    sym_order = bytes(sym for _, sym in canon_list)

    # 4‑b) ***캐논컬 코드맵 재생성***
    canon_codes = {}
    code = 0
    prev_len = 0
    for length, sym in canon_list:
        code <<= (length - prev_len)
        canon_codes[sym] = f"{code:0{length}b}"
        code += 1
        prev_len = length

    # 5) 페이로드 비트열 생성 (캐논컬 코드로!)
    bits = ''.join(canon_codes[b] for b in data)
    pad = (8 - len(bits) % 8) % 8
    bits += '0' * pad
    payload = int(bits, 2).to_bytes(len(bits) // 8, 'big')

    return bytes([pad]) + bytes(code_len) + sym_order + payload

# -------------------------------------------------
# 0‑C. LZ77 ENCODER (간단 구현)
# -------------------------------------------------
WINDOW = 4095  # 12비트 오프셋
MAXLEN = 15   # 4비트 길이 (실제 길이 = value+3)

def lz77_encode(data: bytes) -> bytes:
    if not data:
        return b''
    out = bytearray()
    i = 0
    while i < len(data):
        best_off = 0
        best_len = 0
        start = max(0, i - WINDOW)
        for j in range(start, i):
            length = 0
            while (
                length < MAXLEN and
                i + length < len(data) and
                data[j + length] == data[i + length]
            ):
                length += 1
            if length > best_len:
                best_off = i - j
                best_len = length
            if best_len == MAXLEN:
                break
        if best_len >= 3:
            token = (best_off << 4) | (best_len - 3)
            out.extend(token.to_bytes(2, 'big'))
            nxt = data[i + best_len] if i + best_len < len(data) else 0
            out.append(nxt)
            i += best_len + 1
        else:
            out.extend((0).to_bytes(2, 'big'))
            out.append(data[i])
            i += 1
    return bytes(out)

# -------------------------------------------------
# 0‑D. LZW ENCODER (간단 구현)
# -------------------------------------------------

def lzw_encode(data: bytes) -> bytes:
    if not data:
        return b''
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(dict_size)}
    w = b''
    result = []
    for c in data:
        wc = w + bytes([c])
        if wc in dictionary:
            w = wc
        else:
            if w:
                result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = bytes([c])
    if w:
        result.append(dictionary[w])

    out = bytearray()
    for code in result:
        out.extend(code.to_bytes(2, 'big'))
    return bytes(out)

# -------------------------------------------------
# 1. 데이터 특성 분석 (선택 기능; 여기서는 사용 안 함)
# -------------------------------------------------

def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    total = len(data)
    cnt = Counter(data)
    return -sum((f / total) * math.log2(f / total) for f in cnt.values())

# -------------------------------------------------
# 3. CLI (수동 선택)
# -------------------------------------------------

EXT_MAP = {
    'rle': '.rle',
    'huffman': '.huff',
    'lz77': '.lz77',
    'lzw': '.lzw',
}

def main():
    print("=== 간이 압축기 ===")
    print("지원 알고리즘: RLE / HUFFMAN / LZ77 / LZW\n")

    # ① 입력 경로
    in_path = Path(input("압축할 파일 경로를 입력하세요: ").strip())
    if not in_path.is_file():
        print("[오류] 파일을 찾을 수 없습니다."); return
    data = in_path.read_bytes()
    print(f"원본 크기: {len(data):,} bytes")

    # ② 압축 방식
    algo = input("압축 방식을 선택하세요: ").strip().lower()
    if algo not in EXT_MAP:
        print("[오류] 지원하지 않는 알고리즘입니다."); return

    # ③ 출력 경로 (알고리즘별 확장자 사용)
    default_out = in_path.with_suffix(in_path.suffix + EXT_MAP[algo])
    out_str = input(
        f"출력 파일 경로를 입력하세요\n(Enter 입력 시 기본값 → {default_out.name}): "
    ).strip()
    out_path = Path(out_str) if out_str else default_out

    # ④ 압축 수행
    if algo == 'rle':
        comp = rle_encode(data)
    elif algo == 'huffman':
        comp = huff_encode(data)
    elif algo == 'lz77':
        comp = lz77_encode(data)
    elif algo == 'lzw':
        comp = lzw_encode(data)
    else:
        print("[오류] 내부 분기 실패"); return

    # ⑤ 결과 저장/출력
    out_path.write_bytes(comp)
    print(f"압축 완료 → {out_path} ({len(comp):,} bytes)")
    if len(data):
        print(f"압축률: {len(data)/len(comp):.2f}")
        print(f"압축 효율: {len(comp)/len(data)*100:.2f}%")

if __name__ == "__main__":
    main()
