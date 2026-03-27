# 四元数拡大における素数の偏り（Chebyshev bias）

Q8（四元数群）ガロア拡大体における Artin L 関数と Chebyshev bias の数値計算データ。

## 概要

有理数体 Q 上のガロア群が四元数群 Q₈ と同型なガロア拡大体 L/Q に対し、2次元既約表現 ρ₀ に付随する Artin L 関数 L(s, ρ₀) の中心値 s = 1/2 における零点の位数 m(ρ₀) と、素数の偏り（Chebyshev bias）の関係を数値的に検証するプロジェクトです。

青木・小山の予想 [1, 2] によれば、フロベニウス共役類 σ に対する偏り関数

```
S_σ(x) = π_{1/2}(x) − (|G|/|c_σ|) π_{1/2}(x; σ)
```

は、漸近的に

```
S_σ(x) = (M(σ) + m(σ)) log log x + c + o(1)    (x → ∞)
```

と振る舞います。ここで M(σ) はガロア群の構造で定まる代数的な値、m(σ) は L 関数の零点の位数で定まる解析的な値です。

## データ

### Omar (2001) の23例

Omar [3] が論文で扱った23個の四元数拡大体について、10¹⁰ までの素数に対する偏り関数 S₁〜S₅ を計算しプロットしました。

| ディレクトリ | 内容 |
|------------|------|
| `omar_results/` | 23ケースの JSON データと PNG プロット |

各ケースについて以下を含みます：
- `omar_XX.json` — 偏り関数の数値データ（x, S₁〜S₅）
- `omar_XX.png` — 数値データと理論曲線の比較プロット

#### W(χ₀) = +1, m(ρ₀) = 0 の例（8ケース）

Cases 1, 5, 11, 14, 16, 17, 19, 22。S₁ の log log x の係数は +1/2、S₂ は +5/2。

#### W(χ₀) = −1, m(ρ₀) = 1 の例（15ケース）

Cases 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 15, 18, 20, 21, 23。S₁ の log log x の係数は +5/2、S₂ は +1/2。m(ρ₀) = 0 の場合と比べ、S₁ と S₂ の係数が入れ替わります。

### LMFDB 全 Q8 拡大体の L 値データ

LMFDB に収録されている8次 Q8 拡大体（Galois group 8T5）全50,257件について、L(1/2, ρ₀)、L'(1/2, ρ₀)、L''(1/2, ρ₀) を系統的に計算しました。

| ファイル | 内容 |
|---------|------|
| `lmfdb_lhalf_complete.tsv` | 50,257件の L 値データ（エラー0件） |

主な結果：
- W = +1: 24,595件、W = −1: 25,662件
- |L(1/2)| < 10⁻⁸ の体が2件存在（index 39973, 9710）
- 全ケースで Omar の予想 m(ρ₀) = 0 (W=+1) または 1 (W=−1) と整合

## 計算手法

### 高速フロベニウス分類

Q8 拡大体 L は3つの二次部分体 Q(√d₁), Q(√d₂), Q(√d₃) を含みます。素数 p のフロベニウス共役類は、Kronecker 記号 (d₁/p), (d₂/p) の値で大部分が分類できます。

| (d₁/p) | (d₂/p) | フロベニウス | 計算コスト |
|---------|---------|-----------|----------|
| +1 | −1 | {i, −i} | Kronecker のみ |
| −1 | +1 | {j, −j} | Kronecker のみ |
| −1 | −1 | {k, −k} | Kronecker のみ |
| +1 | +1 | {1} or {−1} | 多項式冪剰余 |

素数の約3/4は Kronecker 記号のみで分類でき、残り約1/4だけ x^p mod f(x) mod p の計算が必要です。

### 実装

- **Julia 版** (`compute_bias_fast.jl`): セグメンテッド篩 + Kronecker + マルチスレッド。150コアで 10¹⁰ まで1体あたり数分。
- **Python/GP 版** (`compute_bias_parallel.py`): PARI/GP の `factormod` を使った並列計算。

## 使い方

### 依存関係

- Julia 1.9+ （Primes.jl, JSON.jl）
- Python 3.8+ （numpy, matplotlib）
- PARI/GP 2.17+

### 偏り関数の計算

```bash
# Julia 高速版
julia --threads=150 compute_bias_fast.jl --all --x-max 1e10

# プロット生成
python plot_9cases.py --indir 9cases_results
```

### L 値の計算

```bash
# PARI/GP を使用
python compute_omar_lvalues.py
```

## 参考文献

[1] M. Aoki and S. Koyama, "Chebyshev's Bias against Splitting and Principal Primes in Global Fields", J. Number Theory **245**, 233–262 (2023).

[2] 青木美穂, 「代数体の素イデアルの偏り（その２）」, 現代数学 2026年5月号, 現代数学社, 2026年.

[3] S. Omar, "On Artin L-functions for octic quaternion fields", Experiment. Math. **10**, No. 2, 237–245 (2001).

## 著者

- 横山重俊（国立情報学研究所）

## 謝辞

本プロジェクトの計算データは、青木美穂氏（[2]の著者）の研究に使用されています。青木美穂先生、小山信也先生、黒川信重先生に感謝いたします。
