#!/usr/bin/env python3
"""
Omar (2001) 全23ケースの Chebyshev bias を計算しプロット生成。

使い方:
    # 全23ケース (10^9, 150並列)
    python compute_omar_all.py --x-max 1e9 -w 150

    # 特定のケースだけ
    python compute_omar_all.py --cases 1,2,14,15 --x-max 1e9 -w 150

    # プロットだけ再生成（JSON が既にある場合）
    python compute_omar_all.py --plot-only

    # 10^12 (3000チャンク)
    python compute_omar_all.py --x-max 1e12 -w 150 --chunks 3000 --timeout 14400
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 同じディレクトリの compute_bias_parallel.py をインポート
sys.path.insert(0, str(Path(__file__).parent))
from compute_bias_parallel import compute_bias_parallel
from omar_cases import OMAR_CASES


def parse_poly_to_coeffs(poly_str):
    """GP多項式文字列 → 係数リスト [a0, a1, ..., a8]"""
    import re
    coeffs = [0] * 9
    # x^8 の係数は常に1
    poly_str = poly_str.replace(" ", "")
    # 各項をパース
    # +/- で分割
    terms = re.findall(r'[+-]?[^+-]+', poly_str)
    for term in terms:
        term = term.strip()
        if not term:
            continue
        if 'x' not in term:
            coeffs[0] = int(term)
        elif '^' not in term:
            # x or N*x
            if term == 'x' or term == '+x':
                coeffs[1] = 1
            elif term == '-x':
                coeffs[1] = -1
            else:
                coeffs[1] = int(term.replace('*x', ''))
        else:
            m = re.match(r'([+-]?\d*)\*?x\^(\d+)', term)
            if m:
                coeff_str = m.group(1)
                if coeff_str in ('', '+'):
                    coeff_str = '1'
                elif coeff_str == '-':
                    coeff_str = '-1'
                deg = int(m.group(2))
                coeffs[deg] = int(coeff_str)
    return coeffs


def get_m_for_case(case):
    """Omar の n 値から理論的な m を返す"""
    return case["n"]


def make_field_info(case, coeffs):
    """plot_bias.py 用の field_info を作る"""
    # LMFDB ラベル候補
    LMFDB_LABELS = {
        1: "8.8.1340095640625.1",
        2: "8.0.1340095640625.1",
        3: "8.8.74220378765625.1",
        14: "8.8.12230590464.1",
        15: "8.0.12230590464.1",
        16: "8.8.29721861554176.1",
        17: "8.8.789298907447296.1",
        22: "8.0.343064484000000.1",
    }
    cid = case["id"]
    return {
        "index": cid,
        "lmfdb_label": f"Omar Case {cid}",
        "lmfdb_secondary": LMFDB_LABELS.get(cid, "N/A"),
        "coeffs": coeffs,
        "polynomial": case["poly"].replace("*", ""),
        "polynomial_str": case["poly"],
        "L_half": 0.0 if case["W"] == -1 else None,  # 後で計算値で上書き
        "L_prime": None,
        "L_double_prime": None,
        "root_number": float(case["W"]),
        "m": case["n"],
        "description": f"Omar Case {case['id']} (W={case['W']:+d}, n={case['n']})",
        "disc": case["disc"],
        "RI": case["RI"],
        "quad": case["quad"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute bias for all 23 Omar cases")
    parser.add_argument("--x-max", type=float, default=1e9)
    parser.add_argument("-w", "--workers", type=int, default=150)
    parser.add_argument("--chunks", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case IDs (e.g. 1,2,14,15). Default: all")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only regenerate plots from existing JSON files")
    parser.add_argument("--outdir", default="omar_results",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ケース選択
    if args.cases:
        case_ids = [int(x) for x in args.cases.split(",")]
        cases = [c for c in OMAR_CASES if c["id"] in case_ids]
    else:
        cases = OMAR_CASES

    if args.plot_only:
        # 既存 JSON からプロットだけ再生成
        from plot_bias import plot_bias, theory_coefficients, fit_constant_C
        import numpy as np

        for case in cases:
            json_path = os.path.join(args.outdir, f"omar_{case['id']:02d}.json")
            if not os.path.exists(json_path):
                print(f"Skip Case {case['id']}: {json_path} not found")
                continue
            with open(json_path) as f:
                data = json.load(f)
            coeffs = parse_poly_to_coeffs(case["poly"])
            field_info = make_field_info(case, coeffs)
            # JSON のメタデータで上書き
            for k in ["L_half", "L_prime", "L_double_prime"]:
                if k in data and data[k] is not None:
                    field_info[k] = data[k]
            png_path = os.path.join(args.outdir, f"omar_{case['id']:02d}.png")
            plot_bias(data, field_info, png_path)
        return

    # 全ケース計算
    t0_total = time.time()
    for i, case in enumerate(cases):
        case_id = case["id"]
        coeffs = parse_poly_to_coeffs(case["poly"])
        field_info = make_field_info(case, coeffs)

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(cases)}] Omar Case {case_id}: "
              f"W={case['W']:+d}, n={case['n']}, {case['RI']}")
        print(f"  f(x) = {case['poly']}")
        print(f"  disc = {case['disc']}")
        print(f"  Q(√{case['quad'][0]}), Q(√{case['quad'][1]})")
        print(f"{'='*70}")

        bias_data = compute_bias_parallel(
            coeffs, args.x_max,
            n_workers=args.workers,
            n_chunks=args.chunks or args.workers,
            n_samples_total=args.n_points,
            timeout=args.timeout,
        )

        if not bias_data["x_values"]:
            print(f"  ERROR: No data for Case {case_id}")
            continue

        # メタデータを統合
        result = {k: v for k, v in field_info.items()}
        result.update(bias_data)

        json_path = os.path.join(args.outdir, f"omar_{case_id:02d}.json")
        with open(json_path, "w") as f:
            json.dump(result, f)
        print(f"  Saved: {json_path}")

        # プロットも生成
        from plot_bias import plot_bias
        png_path = os.path.join(args.outdir, f"omar_{case_id:02d}.png")
        plot_bias(bias_data, field_info, png_path)

    elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"All done! {len(cases)} cases in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Results in: {args.outdir}/")
    print(f"  JSON: omar_XX.json")
    print(f"  Plots: omar_XX.png")


if __name__ == "__main__":
    main()
