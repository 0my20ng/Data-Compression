import math, heapq, argparse, json
from collections import Counter, namedtuple
from pathlib import Path
from typing import Dict

# -------------------------------------------------
# 0‑A. RLE ENCODER
# -------------------------------------------------

def rle_encode(data: bytes) -> bytes:
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
    if len(heap) == 1:
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
    if node.sym is not None:
        cmap[node.sym] = prefix or '0'
        order.append(node.sym)
    else:
        _gen_codes(node.left, prefix + '0', cmap, order)
        _gen_codes(node.right, prefix + '1', cmap, order)
    return cmap, order

def huff_encode(data: bytes) -> bytes:
    if not data:
        return b''
    tree = _build_tree(data)
    codes, sym_order = _gen_codes(tree)
    code_len = bytearray(256)
    for sym, c in codes.items():
        code_len[sym] = len(c)
    bits = ''.join(codes[b] for b in data)
    pad = (8 - len(bits) % 8) % 8
    bits += '0' * pad
    payload = int(bits, 2).to_bytes(len(bits) // 8, 'big')
    return bytes([pad]) + bytes(code_len) + bytes(sym_order) + payload

# -------------------------------------------------
# 0‑C. LZ77 ENCODER (간단 구현)
# -------------------------------------------------
WINDOW = 4095
MAXLEN = 15

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
            while length < MAXLEN and i + length < len(data) and data[j + length] == data[i + length]:
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
    w = b""
    result = []
    for c in data:
        wc = w + bytes([c])
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = bytes([c])
    if w:
        result.append(dictionary[w])
    # 2바이트씩 저장 (최대 65535 코드까지)
    out = bytearray()
    for code in result:
        out.extend(code.to_bytes(2, 'big'))
    return bytes(out)

# -------------------------------------------------
# 1. 데이터 특성 분석
# -------------------------------------------------

def entropy(data: bytes) -> float:
    if not data: return 0.0
    total = len(data)
    cnt = Counter(data)
    return -sum((f / total) * math.log2(f / total) for f in cnt.values())

def run_stats(data: bytes):
    if not data: return 0.0, 0.0
    runs, prev, cnt = [], data[0], 1
    for b in data[1:]:
        if b == prev:
            cnt += 1
        else:
            runs.append(cnt)
            prev, cnt = b, 1
    runs.append(cnt)
    return sum(runs) / len(runs), max(runs)

def cond_entropy(data: bytes):
    if len(data) < 2: return 0.0
    trans, ctx = Counter(), Counter()
    for i in range(len(data) - 1):
        pair = (data[i], data[i + 1])
        trans[pair] += 1
        ctx[data[i]] += 1
    total = len(data) - 1
    H = 0.0
    for (a, b), f in trans.items():
        p = f / ctx[a]
        H += (f / total) * (-math.log2(p))
    return H

def analyze(data: bytes) -> Dict[str, float]:
    a, m = run_stats(data)
    return {
        'entropy': entropy(data),
        'avg_run': a,
        'max_run': m,
        'conditional_entropy': cond_entropy(data)
    }

# -------------------------------------------------
# 2. 알고리즘 선택
# -------------------------------------------------

def choose(metrics):
    if metrics['max_run'] >= 20 or metrics['avg_run'] >= 5:
        return 'RLE'
    if metrics['conditional_entropy'] <= 2.0:
        return 'Huffman'
    if metrics['entropy'] <= 5.5:
        return 'LZ77'
    # LZW 조건 추가 (엔트로피가 6.0 초과면 LZW)
    if metrics['entropy'] > 6.0:
        return 'LZW'
    return 'STORED'

# -------------------------------------------------
# 3. CLI
# -------------------------------------------------

# ...existing code...

def main():
    ap = argparse.ArgumentParser(description="자동 압축기 (RLE·Huffman·LZ77·LZW)")
    ap.add_argument('input', nargs='?', help='압축할 파일 경로')
    ap.add_argument('--show-metrics', action='store_true')
    args = ap.parse_args()

    path = Path(args.input) if args.input else Path(input('파일 경로 입력: ').strip())
    if not path.is_file():
        print('[오류] 파일을 찾을 수 없습니다.'); return
    data = path.read_bytes()
    m = analyze(data)
    ent = m['entropy']
    if args.show_metrics:
        print(json.dumps(m, indent=2, ensure_ascii=False))
    algo = choose(m)
    print('선택된 압축 방식 →', algo)
    print(f"원본 크기: {len(data):,} bytes")
    print(f"엔트로피: {ent:.4f} bits/byte")

    if algo == 'RLE':
        comp = rle_encode(data)
        out = path.with_suffix(path.suffix + '.rle')
    elif algo == 'Huffman':
        comp = huff_encode(data)
        out = path.with_suffix(path.suffix + '.huff')
    elif algo == 'LZ77':
        comp = lz77_encode(data)
        out = path.with_suffix(path.suffix + '.lz77')
    elif algo == 'LZW':
        comp = lzw_encode(data)
        out = path.with_suffix(path.suffix + '.lzw')
    else:
        print('무압축(STORED) 권장: 추가 작업 없음')
        return

    out.write_bytes(comp)
    print(f"압축 후 크기: {len(comp):,} bytes")
    if len(data) > 0:
        print(f"압축률: {len(data)/len(comp):.2f}")
        print(f"압축 효율: {len(comp)/len(data)*100:.2f}%")
    # 압축 후 크기가 원본보다 크면 저장하지 않음
    if len(comp) >= len(data):
        print("압축 결과가 원본보다 크거나 같으므로 압축 파일을 저장하지 않습니다.")
        out.unlink(missing_ok=True)  # 파일 삭제 (Python 3.8+)
    else:
        print('압축 완료 →', out)

if __name__ == '__main__':
    main()