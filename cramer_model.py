"""
Residue-restricted Cramer model: a random baseline for comparison against
real class-conditional gap statistics.

Each integer n coprime to q is treated as "prime" independently with
probability q / (phi(q) * log n) -- scaled (vs. the vanilla Cramer
probability 1/log n) so the expected count per admissible residue class
matches the Prime Number Theorem for arithmetic progressions.
"""

import numpy as np
import pandas as pd
from sympy import totient

from gen_primes import build_gap_table


def simulate_cramer_model(N, q, seed=None, n_start=None):
    """
    Generate a synthetic "prime-like" sequence up to N.

    Processes each admissible residue class in fixed-size chunks (like
    the segmented sieve) to bound peak memory regardless of N or q.

    Returns: 1D numpy array of synthetic "primes" (sorted).
    """
    rng = np.random.default_rng(seed)
    phi_q = int(totient(q))
    admissible = [r for r in range(1, q) if np.gcd(r, q) == 1]
    chunk_size = 20_000_000

    if n_start is None:
        n_start = q + 1

    chunks = []
    for r in admissible:
        start = n_start + ((r - n_start) % q)
        if start > N:
            continue

        n_positions = (N - start) // q + 1
        step_span = chunk_size * q

        pos = start
        for _ in range(0, n_positions, chunk_size):
            chunk_end = min(pos + step_span - q, N)
            n_r = np.arange(pos, chunk_end + 1, q, dtype=np.int64)
            if len(n_r) == 0:
                break
            log_n = np.log(np.maximum(n_r, 2)).astype(np.float32)
            prob = np.minimum(1.0, q / (phi_q * log_n)).astype(np.float32)
            draws = rng.random(len(n_r), dtype=np.float32)
            chunks.append(n_r[draws < prob].astype(np.int32))
            del n_r, log_n, prob, draws
            pos = chunk_end + q

    n = np.concatenate(chunks) if chunks else np.array([], dtype=np.int64)
    n.sort()
    return n


def build_synthetic_gap_table(N, q, seed=None):
    """Runs the simulation and builds the gap/residue table via the
    same build_gap_table used for real primes, for apples-to-apples
    comparison."""
    synthetic = simulate_cramer_model(N, q, seed=seed)
    return build_gap_table(synthetic, moduli=(q,))
