#!/usr/bin/env julia
#
# compute_bias_fast.jl — Q8 Chebyshev bias (v2: segmented sieve)
#
# 150コア完全活用。セグメンテッド篩 + Kronecker で全コアを使い切る。
# Primes.jl 不要（自前篩）。
#
# 依存: JSON.jl のみ
#   ] add JSON
#
# 使い方:
#   julia --threads=150 compute_bias_fast.jl --all --x-max 1e10
#   julia --threads=150 compute_bias_fast.jl --case A1 --x-max 1e10
#

using JSON
using Printf
using Base.Threads

# =====================================================================
# 1. Segmented Sieve
# =====================================================================

function simple_sieve(limit::Int)
    is_prime = trues(limit + 1)
    is_prime[1] = false
    if limit >= 1
        for i in 2:isqrt(limit)
            if is_prime[i]
                for j in (i*i):i:limit
                    is_prime[j] = false
                end
            end
        end
    end
    return findall(is_prime)
end

function process_segment!(callback::F, lo::Int64, hi::Int64,
                          small_primes::Vector{Int}) where F
    seg_size = hi - lo + 1
    is_prime = trues(seg_size)

    if lo <= 1
        for i in 1:min(2 - lo, seg_size)
            is_prime[i] = false
        end
    end

    for sp in small_primes
        start = lo + mod(-lo, sp)
        if start == lo && lo != sp
            # lo is a multiple of sp but not sp itself
        elseif start == lo && lo == sp
            start += sp
        end
        # More robust: find smallest multiple of sp >= max(sp*sp, lo)
        first_mul = max(Int64(sp) * Int64(sp), lo + mod(-lo, Int64(sp)))
        if first_mul == lo && lo % sp == 0 && lo != sp
            # lo is divisible by sp and lo != sp → mark it
        elseif first_mul < lo
            first_mul += sp
        end
        
        j = first_mul
        while j <= hi
            idx = j - lo + 1
            is_prime[idx] = false
            j += sp
        end
    end

    for i in 1:seg_size
        if is_prime[i]
            p = lo + i - 1
            if p >= 2
                callback(Int64(p))
            end
        end
    end
end

# =====================================================================
# 2. Kronecker Symbol
# =====================================================================

@inline function kronecker_symbol(a::Int64, p::Int64)::Int8
    if p == 2
        iseven(a) && return Int8(0)
        r = a & 7
        r < 0 && (r += 8)
        return (r == 1 || r == 7) ? Int8(1) : Int8(-1)
    end
    a_j = mod(a, p)
    a_j < 0 && (a_j += p)
    n = p
    result = Int8(1)
    while a_j != 0
        while a_j & 1 == 0
            a_j >>= 1
            r = n & 7
            (r == 3 || r == 5) && (result = -result)
        end
        a_j, n = n, a_j
        (a_j & 3 == 3 && n & 3 == 3) && (result = -result)
        a_j = mod(a_j, n)
    end
    return n == 1 ? result : Int8(0)
end

# =====================================================================
# 3. x^p mod f(x) mod p
# =====================================================================

function xp_equals_x(p::Int64, coeffs::Vector{Int64})::Bool
    deg = length(coeffs) - 1  # 8
    pm = p

    # f_red: coefficients of f mod p (degree 0..7, leading x^8 is implicit)
    f_red = Vector{Int64}(undef, deg)
    for i in 1:deg
        v = mod(coeffs[i], pm)
        f_red[i] = v < 0 ? v + pm : v
    end

    # result = 1
    res = zeros(Int64, deg)
    res[1] = 1
    # base = x
    bas = zeros(Int64, deg)
    deg >= 2 && (bas[2] = 1)

    # Temp buffer for multiplication
    prod_buf = Vector{Int128}(undef, 2 * deg - 1)
    
    exp = p
    while exp > 0
        if exp & 1 == 1
            polymul!(res, bas, f_red, pm, deg, prod_buf)
        end
        exp >>= 1
        if exp > 0
            bas_copy = copy(bas)
            polymul!(bas, bas_copy, f_red, pm, deg, prod_buf)
        end
    end

    # Check res == x
    res[2] != 1 && return false
    for i in 1:deg
        i == 2 && continue
        res[i] != 0 && return false
    end
    return true
end

function polymul!(a::Vector{Int64}, b::Vector{Int64}, f_red::Vector{Int64},
                  pm::Int64, deg::Int, buf::Vector{Int128})
    # a = a * b mod f mod p, in-place into a
    prod_len = 2 * deg - 1
    fill!(buf, Int128(0))

    @inbounds for i in 1:deg
        ai = a[i]
        ai == 0 && continue
        ai128 = Int128(ai)
        for j in 1:deg
            bj = b[j]
            bj == 0 && continue
            k = i + j - 1
            buf[k] = mod(buf[k] + ai128 * Int128(bj), Int128(pm))
        end
    end

    # Reduce mod f (monic x^deg)
    @inbounds for k in prod_len:-1:(deg+1)
        c = buf[k]
        c == 0 && continue
        offset = k - deg
        for i in 1:deg
            buf[offset + i - 1] = mod(buf[offset + i - 1] - c * Int128(f_red[i]), Int128(pm))
        end
        buf[k] = 0
    end

    @inbounds for i in 1:deg
        v = Int64(mod(buf[i], Int128(pm)))
        a[i] = v < 0 ? v + pm : v
    end
end

# =====================================================================
# 4. Main Computation
# =====================================================================

struct Acc
    wt::Float64; w0::Float64; w1::Float64
    w2::Float64; w3::Float64; w4::Float64
end
Acc() = Acc(0,0,0,0,0,0)
Base.:+(a::Acc,b::Acc) = Acc(a.wt+b.wt,a.w0+b.w0,a.w1+b.w1,a.w2+b.w2,a.w3+b.w3,a.w4+b.w4)

function compute_segment_direct(lo::Int64, hi::Int64, small_primes::Vector{Int},
                                 coeffs::Vector{Int64}, d1::Int64, d2::Int64)
    # Inline sieve + classification (no closure → no boxing)
    seg_size = Int(hi - lo + 1)
    is_prime = trues(seg_size)

    if lo <= 1
        is_prime[1] = false
        if seg_size >= 2; is_prime[2] = (lo + 1 >= 2); end
    end

    @inbounds for sp in small_primes
        sp64 = Int64(sp)
        first_mul = max(sp64 * sp64, lo + mod(-lo, sp64))
        if first_mul < lo; first_mul += sp64; end
        j = first_mul
        while j <= hi
            is_prime[Int(j - lo + 1)] = false
            j += sp64
        end
    end

    lwt = 0.0; lw0 = 0.0; lw1 = 0.0
    lw2 = 0.0; lw3 = 0.0; lw4 = 0.0

    @inbounds for i in 1:seg_size
        is_prime[i] || continue
        p = lo + Int64(i - 1)
        p < 2 && continue

        w = 1.0 / sqrt(Float64(p))
        lwt += w

        kr1 = kronecker_symbol(d1, p)
        kr2 = kronecker_symbol(d2, p)

        if kr1 == 0 || kr2 == 0
            # ramified → skip
        elseif kr1 == Int8(1) && kr2 == Int8(-1)
            lw2 += w
        elseif kr1 == Int8(-1) && kr2 == Int8(1)
            lw3 += w
        elseif kr1 == Int8(-1) && kr2 == Int8(-1)
            lw4 += w
        else  # both +1: need x^p mod f test
            if xp_equals_x(p, coeffs)
                lw0 += w
            else
                lw1 += w
            end
        end
    end

    return Acc(lwt, lw0, lw1, lw2, lw3, lw4)
end

function compute_bias(coeffs::Vector{Int64}, d1::Int64, d2::Int64, x_max::Int64;
                      n_samples::Int=5000)
    nt = nthreads()
    println("  Threads: $nt")
    if nt < 10
        @warn "Low thread count! Start julia with: julia --threads=150"
    end

    sqrt_x = isqrt(x_max) + 1
    small_primes = simple_sieve(sqrt_x)
    println("  Small primes for sieve: $(length(small_primes)) (up to $sqrt_x)")

    # ===== Phase 1: x <= transition (single-threaded, prime-by-prime) =====
    # Record at EVERY prime for maximum density (like reference plot)
    transition = min(Int64(1_000_000), x_max)
    all_primes_small = simple_sieve(transition)
    println("  Phase 1: $(length(all_primes_small)) primes up to $transition (every prime recorded)")

    xv = Int64[]; s1v = Float64[]; s2v = Float64[]
    s3v = Float64[]; s4v = Float64[]; s5v = Float64[]

    wt = 0.0; w0 = 0.0; w1 = 0.0; w2 = 0.0; w3 = 0.0; w4 = 0.0

    for p in all_primes_small
        w = 1.0 / sqrt(Float64(p))
        wt += w
        kr1 = kronecker_symbol(d1, Int64(p))
        kr2 = kronecker_symbol(d2, Int64(p))
        if kr1 == 0 || kr2 == 0
            # ramified
        elseif kr1 == Int8(1) && kr2 == Int8(-1)
            w2 += w
        elseif kr1 == Int8(-1) && kr2 == Int8(1)
            w3 += w
        elseif kr1 == Int8(-1) && kr2 == Int8(-1)
            w4 += w
        else
            if xp_equals_x(Int64(p), coeffs)
                w0 += w
            else
                w1 += w
            end
        end

        # Record at every prime
        push!(xv, Int64(p))
        push!(s1v, wt - 8*w0); push!(s2v, wt - 8*w1)
        push!(s3v, wt - 4*w2); push!(s4v, wt - 4*w3); push!(s5v, wt - 4*w4)
    end

    cum_phase1 = Acc(wt, w0, w1, w2, w3, w4)
    println("  Phase 1 done: $(length(xv)) data points")

    if x_max <= transition
        println("  Data points: $(length(xv))")
        @printf("  Final: S1=%.4f S2=%.4f S3=%.4f S4=%.4f S5=%.4f\n",
                s1v[end], s2v[end], s3v[end], s4v[end], s5v[end])
        return Dict("x_values"=>xv,"S1"=>s1v,"S2"=>s2v,"S3"=>s3v,"S4"=>s4v,"S5"=>s5v)
    end

    # ===== Phase 2: x > transition (multi-threaded segmented sieve) =====
    # Adaptive segment size: smaller near 10^6 for density on log scale
    first_seg_lo = transition + 1
    segments_p2 = Tuple{Int64,Int64}[]
    x = first_seg_lo
    while x <= x_max
        if x < 10_000_000        # 10^6 ~ 10^7: 5K segments → ~1800 pts
            sz = Int64(5_000)
        elseif x < 100_000_000   # 10^7 ~ 10^8: 10K segments → ~9000 pts
            sz = Int64(10_000)
        elseif x < 1_000_000_000 # 10^8 ~ 10^9: 50K segments → ~18000 pts
            sz = Int64(50_000)
        else                     # 10^9+: 100K segments → ~90000 pts
            sz = Int64(100_000)
        end
        hi = min(x + sz - 1, x_max)
        push!(segments_p2, (x, hi))
        x = hi + 1
    end
    n_segments = length(segments_p2)
    println("  Phase 2: $n_segments segments (adaptive 50K→500K, every segment recorded)")

    seg_results = Vector{Acc}(undef, n_segments)

    t0 = time()
    progress = Atomic{Int}(0)

    @threads :dynamic for seg_idx in 1:n_segments
        lo, hi = segments_p2[seg_idx]

        seg_results[seg_idx] = compute_segment_direct(
            lo, hi, small_primes, coeffs, d1, d2)

        done = atomic_add!(progress, 1) + 1
        if done % 1000 == 0 || done == n_segments
            elapsed = time() - t0
            rate = done / elapsed
            eta = (n_segments - done) / max(rate, 0.001)
            @printf("\r    [%d/%d] %.0f seg/s, ETA %.0fs   ",
                    done, n_segments, rate, eta)
        end
    end
    println()
    @printf("  Phase 2 compute: %.1fs\n", time() - t0)

    # Record at every segment boundary
    cum = cum_phase1
    for seg_idx in 1:n_segments
        cum = cum + seg_results[seg_idx]
        _, hi = segments_p2[seg_idx]
        push!(xv, hi)
        push!(s1v, cum.wt-8*cum.w0); push!(s2v, cum.wt-8*cum.w1)
        push!(s3v, cum.wt-4*cum.w2); push!(s4v, cum.wt-4*cum.w3)
        push!(s5v, cum.wt-4*cum.w4)
    end

    println("  Total data points: $(length(xv))")
    @printf("  Final: S1=%.4f S2=%.4f S3=%.4f S4=%.4f S5=%.4f\n",
            s1v[end], s2v[end], s3v[end], s4v[end], s5v[end])
    return Dict("x_values"=>xv,"S1"=>s1v,"S2"=>s2v,"S3"=>s3v,"S4"=>s4v,"S5"=>s5v)
end

# =====================================================================
# 5. GP discriminant helper
# =====================================================================

function compute_discs_gp(coeffs::Vector{Int64})
    ts = String[]
    for (i,c) in enumerate(coeffs)
        c==0 && continue; d=i-1
        d==0 ? push!(ts,string(c)) : d==1 ? push!(ts,"$(c)*x") : push!(ts,"$(c)*x^$d")
    end
    ps = join(ts," + "); ps = replace(ps, "+ -"=>"- ")
    gp = """default(parisize,"1G");default(parisizemax,"8G");
{my(nf=nfinit($ps));my(sf=nfsubfields(nf,2));for(i=1,3,print(coredisc(poldisc(sf[i][1]))));}
\\q\n"""
    tmp = tempname()*".gp"
    write(tmp, gp)
    out = try; read(`gp -q $tmp`,String); finally; rm(tmp,force=true); end
    ls = [strip(l) for l in split(strip(out),"\n") if !isempty(strip(l))&&!startswith(strip(l),"***")]
    ds = [parse(Int64,l) for l in ls if tryparse(Int64,l)!==nothing]
    length(ds)==3 || error("Got $(length(ds)) discs: $ls")
    return ds
end

# =====================================================================
# 6. Cases
# =====================================================================

const CASES = Dict(
    "A1"=>Dict("label"=>"Omar Case 14","m"=>0,"coeffs"=>Int64[9,0,-36,0,36,0,-12,0,1],
        "discs"=>Int64[12,24,8],"polynomial"=>"x^8 - 12x^6 + 36x^4 - 36x^2 + 9",
        "lmfdb_label"=>"8.8.12230590464.1","L_half"=>0.708769134170847,
        "L_prime"=>1.06385000194906,"L_double_prime"=>-3.09575306095333,"root_number"=>1.0),
    "A2"=>Dict("label"=>"Omar Case 1","m"=>0,
        "coeffs"=>Int64[-395,1345,-1090,-305,361,29,-34,-1,1],"discs"=>Int64[5,21,105],
        "polynomial"=>"x^8 - x^7 - 34x^6 + 29x^5 + 361x^4 - 305x^3 - 1090x^2 + 1345x - 395",
        "lmfdb_label"=>"8.8.1340095640625.1","L_half"=>0.742867011959717,
        "L_prime"=>0.533544225239188,"L_double_prime"=>-3.12667448783679,"root_number"=>1.0),
    "A3"=>Dict("label"=>"Omar Case 19","m"=>0,
        "coeffs"=>Int64[22500,0,-9000,0,1170,0,-60,0,1],"discs"=>Int64[5,6,30],
        "polynomial"=>"x^8 - 60x^6 + 1170x^4 - 9000x^2 + 22500",
        "lmfdb_label"=>"8.8.47775744000000.?","L_half"=>nothing,
        "L_prime"=>nothing,"L_double_prime"=>nothing,"root_number"=>1.0),
    "B1"=>Dict("label"=>"Omar Case 15","m"=>1,
        "coeffs"=>Int64[9,0,36,0,36,0,12,0,1],"discs"=>Int64[12,24,8],
        "polynomial"=>"x^8 + 12x^6 + 36x^4 + 36x^2 + 9",
        "lmfdb_label"=>"8.0.12230590464.1","L_half"=>0.0,
        "L_prime"=>3.40290313280938,"L_double_prime"=>-11.1656754867106,"root_number"=>-1.0),
    "B2"=>Dict("label"=>"Omar Case 2","m"=>1,
        "coeffs"=>Int64[22325625,0,1488375,0,34020,0,315,0,1],"discs"=>Int64[5,21,105],
        "polynomial"=>"x^8 + 315x^6 + 34020x^4 + 1488375x^2 + 22325625",
        "lmfdb_label"=>"8.0.1340095640625.1","L_half"=>0.0,
        "L_prime"=>3.83724034069741,"L_double_prime"=>-18.5981030602912,"root_number"=>-1.0),
    "B3"=>Dict("label"=>"Omar Case 18","m"=>1,
        "coeffs"=>Int64[900,0,-1800,0,810,0,-60,0,1],"discs"=>Int64[5,6,30],
        "polynomial"=>"x^8 - 60x^6 + 810x^4 - 1800x^2 + 900",
        "lmfdb_label"=>"8.8.47775744000000.?","L_half"=>0.0,
        "L_prime"=>nothing,"L_double_prime"=>nothing,"root_number"=>-1.0),
    "C1"=>Dict("label"=>"LMFDB idx 630","m"=>2,
        "coeffs"=>Int64[22256760969,0,399224412,0,1790244,0,2676,0,1],"discs"=>Int64[12,24,8],
        "polynomial"=>"x^8 + 2676x^6 + 1790244x^4 + 399224412x^2 + 22256760969",
        "lmfdb_label"=>"8.0.30245925385219866624.1","L_half"=>2.45130549570495e-6,
        "L_prime"=>-1.72762667901460e-5,"L_double_prime"=>67.4912209501730,"root_number"=>1.0),
    "C2"=>Dict("label"=>"LMFDB idx 9710","m"=>2,
        "coeffs"=>Int64[-10666078889,-50880621137,-24292293418,-2290205,27916561,8015,-9472,-1,1],
        "discs"=>Int64[],"polynomial"=>"x^8 - x^7 - 9472x^6 + ...",
        "lmfdb_label"=>"N/A","L_half"=>4.32841930643136e-11,
        "L_prime"=>-2.12432553361299e-10,"L_double_prime"=>37.3991924803517,"root_number"=>1.0),
    "C3"=>Dict("label"=>"LMFDB idx 6356","m"=>2,
        "coeffs"=>Int64[3863865600,0,-6761764800,0,15804180,0,-7770,0,1],"discs"=>Int64[],
        "polynomial"=>"x^8 - 7770x^6 + 15804180x^4 - 6761764800x^2 + 3863865600",
        "lmfdb_label"=>"N/A","L_half"=>5.74310772818824e-6,
        "L_prime"=>-2.85552893764771e-5,"L_double_prime"=>51.5521715854009,"root_number"=>1.0),
)

# =====================================================================
# 7. Main
# =====================================================================

function run_case(key::String, x_max::Int64, outdir::String)
    info = CASES[key]; coeffs = info["coeffs"]
    println("\n"*"="^60)
    println("[$key] $(info["label"]) (m=$(info["m"]))")
    println("  f(x) = $(info["polynomial"])")
    println("="^60)
    discs = info["discs"]
    if isempty(discs)
        println("  Computing discs via GP..."); discs = compute_discs_gp(coeffs)
    end
    println("  Discs: $discs")
    t0 = time()
    bias = compute_bias(coeffs, Int64(discs[1]), Int64(discs[2]), x_max)
    elapsed = time()-t0
    @printf("  TOTAL: %.1fs (%.1f min)\n", elapsed, elapsed/60)
    result = Dict{String,Any}()
    for (k,v) in info; result[k]=v; end
    for s in ["x_values","S1","S2","S3","S4","S5"]; result[s]=bias[s]; end
    result["computation_time"]=elapsed
    jp = joinpath(outdir,"$key.json")
    open(jp,"w") do f; JSON.print(f,result); end
    println("  Saved: $jp")
    return elapsed
end

function main()
    x_max=Int64(1e10); cases=String[]; outdir="9cases_results"
    i=1
    while i<=length(ARGS)
        a=ARGS[i]
        if a=="--x-max"&&i<length(ARGS); x_max=Int64(parse(Float64,ARGS[i+1])); i+=2
        elseif a=="--case"&&i<length(ARGS); push!(cases,ARGS[i+1]); i+=2
        elseif a=="--all"; cases=sort(collect(keys(CASES))); i+=1
        elseif a=="--outdir"&&i<length(ARGS); outdir=ARGS[i+1]; i+=2
        else; i+=1; end
    end
    if isempty(cases)
        println("Usage: julia --threads=150 compute_bias_fast.jl --all --x-max 1e10")
        println("Cases: A1,A2,A3 (m=0)  B1,B2,B3 (m=1)  C1,C2,C3 (m~2)")
        return
    end
    mkpath(outdir)
    println("=== Q8 Chebyshev Bias (Julia v2: segmented sieve) ===")
    @printf("X_max = %.0e, Threads = %d\n", Float64(x_max), nthreads())
    t0=time()
    for k in cases; haskey(CASES,k)||continue; run_case(k,x_max,outdir); end
    @printf("\nAll done! %d cases in %.0fs (%.1f min)\n", length(cases), time()-t0, (time()-t0)/60)
    println("Plot: python plot_9cases.py --indir $outdir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
