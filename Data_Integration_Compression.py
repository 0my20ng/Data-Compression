"""compression_benchmark.py
CLI 스크립트: 하나의 텍스트 파일을 RLE·Huffman·LZ77·LZW 알고리즘으로 압축→복원하여
압축률·복원 정확도·소요 시간·용량을 표로 출력한다.

필수 파일:
  - Data_Incoder.py
  - Data_Decoder.py
같은 디렉터리에 두면 하위 모듈 import 없이 subprocess 로 호출 가능.
"""

from __future__ import annotations
import subprocess, time, hashlib, json, sys
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

INCODER_CMD = [sys.executable, "Data_Incoder.py"]
DECODER_CMD = [sys.executable, "Data_Decoder.py"]

# 인코더 내부 확장자 매핑과 동일하게 설정
EXT_MAP = {
    "rle": ".rle",
    "huffman": ".huff",   # 주의: .huffman 아님!
    "lz77": ".lz77",
    "lzw": ".lzw",
}
ALGOS = list(EXT_MAP.keys())  # 유지보수 용이

# ────────────────────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    """파일의 SHA‑256 해시 반환"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_subprocess(cmd: List[str], stdin_bytes: bytes) -> None:
    """stdin 전달하면서 subprocess 실행 (stderr → stdout)"""
    subprocess.run(cmd, input=stdin_bytes, check=True)


def benchmark(src_path: Path) -> List[Dict]:
    raw_size = src_path.stat().st_size
    baseline_hash = sha256(src_path)
    results: List[Dict] = []

    for algo in ALGOS:
        # ── 압축 ───────────────────────────────────────────────────────────
        ext = EXT_MAP[algo]
        default_out = src_path.with_suffix(src_path.suffix + ext)
        enc_stdin = f"{src_path}\n{algo}\n{default_out}\n".encode()
        t0 = time.perf_counter()
        run_subprocess(INCODER_CMD, enc_stdin)
        enc_ms = (time.perf_counter() - t0) * 1000

        if not default_out.exists():
            raise FileNotFoundError(f"[실패] {algo.upper()} 압축 파일 생성 안됨")
        comp_size = default_out.stat().st_size

        # ── 복원 ───────────────────────────────────────────────────────────
        dec_stdin = f"{default_out}\n".encode()
        t0 = time.perf_counter()
        try:
            run_subprocess(DECODER_CMD, dec_stdin)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"[실패] {algo.upper()} 복원 오류: {e.stderr}") from e
        dec_ms = (time.perf_counter() - t0) * 1000

        # 디코더는 *.restored 로만 출력하므로 이름 충돌 방지
        restored_generic = default_out.with_suffix('.restored')
        if not restored_generic.exists():
            raise FileNotFoundError(f"[실패] {algo.upper()} 복원 파일 생성 안됨")
        restored_unique = src_path.with_suffix(src_path.suffix + f"{ext}.restored")
        restored_generic.rename(restored_unique)

        ok = sha256(restored_unique) == baseline_hash

        results.append({
            "algo": algo.upper(),
            "enc_ms": round(enc_ms, 1),
            "dec_ms": round(dec_ms, 1),
            "size": comp_size,
            "ratio": round(comp_size / raw_size * 100, 1),
            "restored_ok": ok,
            "enc_file": str(default_out.name),
            "restored_file": str(restored_unique.name),
        })
    return results


def main():
    src = Path(input("압축할 텍스트 파일 경로: ").strip())
    if not src.is_file():
        print("[오류] 파일을 찾을 수 없습니다."); return

    try:
        results = benchmark(src)
    except Exception as e:
        print(e); return

    # 그래프 시각화
    algos = [r['algo'] for r in results]
    ratios = [r['ratio'] for r in results]
    enc_times = [r['enc_ms'] for r in results]
    dec_times = [r['dec_ms'] for r in results]
    ok_flags = ['OK' if r['restored_ok'] else 'FAIL' for r in results]

    # 한글 폰트 설정 (필요시)
    plt.rcParams['font.family'] = 'Malgun Gothic'

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # 압축률
    axs[0].bar(algos, ratios, color='skyblue')
    axs[0].set_title('압축률(%)')
    axs[0].set_ylabel('압축률(%)')
    axs[0].set_ylim(0, max(ratios)*1.2)

    # 인코딩/디코딩 시간
    width = 0.35
    x = range(len(algos))
    axs[1].bar([i - width/2 for i in x], enc_times, width, label='인코딩(ms)', color='orange')
    axs[1].bar([i + width/2 for i in x], dec_times, width, label='디코딩(ms)', color='green')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(algos)
    axs[1].set_title('인코딩/디코딩 시간')
    axs[1].set_ylabel('시간(ms)')
    axs[1].legend()

    # 무결성
    colors = ['green' if ok == 'OK' else 'red' for ok in ok_flags]
    axs[2].bar(algos, [1]*len(algos), color=colors)
    axs[2].set_title('복원 무결성')
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(['FAIL', 'OK'])

    plt.tight_layout()
    plt.show()

    # JSON 저장
    Path("benchmark_result.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n세부 결과 → benchmark_result.json 저장 완료")


if __name__ == "__main__":
    main()
