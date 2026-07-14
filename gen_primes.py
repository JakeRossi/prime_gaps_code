"""
Generate primes up to N via segmented sieve, and build the prime-gap /
residue-class table used by the rest of the pipeline.

Usage:
    python gen_primes.py --N 10000000 --out data/primes_1e7.csv

Uses the `primesieve` package if installed (faster for N >= 10^8), else
falls back to a pure-Python segmented sieve.
"""

import argparse
import math
import time
import numpy as np
import pandas as pd

try:
    import primesieve
    HAVE_PRIMESIEVE = True
except ImportError:
    HAVE_PRIMESIEVE = False


def simple_sieve(limit):
    """Sieve of Eratosthenes for small limit (base primes up to sqrt(N))."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i * i:: i] = False
    return np.nonzero(is_prime)[0]


def segmented_sieve(N, segment_size=1_000_000):
    """Segmented sieve of Eratosthenes: bounds memory to ~segment_size
    regardless of N. Returns a sorted array of primes <= N."""
    if N < 2:
        return np.array([], dtype=np.int64)

    limit = int(math.isqrt(N)) + 1
    base_primes = simple_sieve(limit)

    primes_list = [base_primes]
    low = limit + 1
    total_span = max(N - low, 1)
    last_report = time.time()
    while low <= N:
        high = min(low + segment_size - 1, N)
        segment = np.ones(high - low + 1, dtype=bool)
        for p in base_primes:
            start = max(p * p, ((low + p - 1) // p) * p)
            if start > high:
                continue
            segment[start - low:: p] = False
        primes_list.append(np.nonzero(segment)[0] + low)
        low = high + 1

        if time.time() - last_report > 5:
            pct = 100 * min(low - limit, total_span) / total_span
            print(f"[gen_primes] segmented sieve progress: {pct:.1f}% "
                  f"(up to {min(low, N):,} of {N:,})")
            last_report = time.time()

    return np.concatenate(primes_list)


def generate_primes(N, segment_size=1_000_000):
    """Generate all primes <= N, using primesieve if available."""
    if N >= 10**8 and not HAVE_PRIMESIEVE:
        print(f"[gen_primes] WARNING: N={N:,} without `primesieve` installed. "
              f"Pure-Python sieve is correct but may be slow at this scale.")
    t0 = time.time()
    if HAVE_PRIMESIEVE:
        primes = np.array(primesieve.primes(N), dtype=np.int64)
    else:
        primes = segmented_sieve(N, segment_size=segment_size)
    elapsed = time.time() - t0
    print(f"[gen_primes] Generated {len(primes):,} primes up to {N:,} "
          f"in {elapsed:.1f}s using "
          f"{'primesieve' if HAVE_PRIMESIEVE else 'pure-Python segmented sieve'}.")
    return primes


def build_gap_table(primes, moduli=(6, 30)):
    """
    Build a dataframe with, for each prime p_n: p_n, gap to p_{n+1}, and
    the residue class of both p_n and p_{n+1} mod each value in `moduli`.

    Uses compact dtypes (int32 for p/gap, int8 for residues) to keep
    memory manageable at N=10^9 scale (~50M rows).
    """
    p = primes[:-1].astype(np.int32)
    p_next = primes[1:].astype(np.int32)
    gaps = (primes[1:] - primes[:-1]).astype(np.int32)

    df = pd.DataFrame({"p": p, "gap": gaps})
    for q in moduli:
        df[f"res_mod{q}"] = (p % q).astype(np.int8)
        df[f"next_res_mod{q}"] = (p_next % q).astype(np.int8)
    return df


def save_table(df, path):
    """Save to Parquet if the path ends in .parquet and pyarrow is
    available, else CSV."""
    if path.endswith(".parquet"):
        try:
            df.to_parquet(path, index=False)
            print(f"[gen_primes] Saved {len(df):,} rows to {path} (Parquet).")
            return
        except ImportError:
            fallback = path.replace(".parquet", ".csv")
            print(f"[gen_primes] pyarrow not installed; falling back to CSV: {fallback}")
            path = fallback
    df.to_csv(path, index=False)
    print(f"[gen_primes] Saved {len(df):,} rows to {path} (CSV).")


def run(N, out_path, moduli=(6, 30), segment_size=1_000_000):
    primes = generate_primes(N, segment_size=segment_size)
    df = build_gap_table(primes, moduli=moduli)
    save_table(df, out_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate primes and the prime-gap/residue table.")
    parser.add_argument("--N", type=int, required=True, help="Upper limit for prime generation.")
    parser.add_argument("--out", type=str, required=True, help="Output path (.csv or .parquet).")
    parser.add_argument("--moduli", type=int, nargs="+", default=[6, 30])
    args = parser.parse_args()
    run(args.N, args.out, moduli=tuple(args.moduli))
