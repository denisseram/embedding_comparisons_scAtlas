"""Scale check for pipeline.fuzzy_upset: synthetic kNN graphs, n cells x M models, reports time and peak RSS.

Graphs are generated one model at a time (never all in memory): cells sit in 20 label blocks and each model
draws a model-specific share of neighbours from a random other block. Usage:
    python -m pipeline.fuzzy_upset_scale [--n 1000000 --models 20 --k 15 --labels 20]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from pipeline.common import get_logger, peak_rss_mb
from pipeline.fuzzy_upset import _tables, label_frequency, memberships, ranked, signatures

log = get_logger("fuzzy_scale")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--models", type=int, default=20)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--labels", type=int, default=20)
    ap.add_argument("--normalize", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    codes = np.sort(rng.integers(0, a.labels, a.n)).astype(np.int32)
    codes[rng.random(a.n) < 0.01] = -1  # 1% unlabelled
    bounds = np.searchsorted(codes.clip(0), np.arange(a.labels + 1))
    freq = label_frequency(codes, a.labels)
    levels = [f"L{i}" for i in range(a.labels)]
    t_gen = t_mem = t_sig = 0.0
    per = {}
    for m in range(a.models):
        t0 = time.perf_counter()
        blk = codes.clip(0)
        lo, hi = bounds[blk], bounds[blk + 1]
        knn = (lo[:, None] + (rng.random((a.n, a.k)) * (hi - lo)[:, None]).astype(np.int64)).astype(np.int32)
        mix = rng.random((a.n, a.k)) < 0.05 * m / max(a.models - 1, 1) * 4  # model m mixes 0..20% of neighbours
        other = rng.integers(0, a.n, mix.sum()).astype(np.int32)
        knn[mix] = other
        del mix, other, lo, hi
        t1 = time.perf_counter()
        P = memberships(knn, codes, a.labels)
        del knn
        t2 = time.perf_counter()
        per[f"m{m}"] = signatures(P, 0.1, 3, a.normalize, freq)
        del P
        t3 = time.perf_counter()
        t_gen, t_mem, t_sig = t_gen + t1 - t0, t_mem + t2 - t1, t_sig + t3 - t2
        log.info(f"model {m + 1}/{a.models}: memberships {t2 - t1:.1f}s, signatures {t3 - t2:.1f}s "
                 f"(peak RSS {peak_rss_mb():.0f} MB)")
    t0 = time.perf_counter()
    cmp = _tables(per, levels, [])
    r = ranked(cmp)
    t_tab = time.perf_counter() - t0
    log.info(f"n={a.n:,} models={a.models} k={a.k} labels={a.labels}: graph generation {t_gen:.0f}s (not part of "
             f"the method), memberships {t_mem:.1f}s, signatures {t_sig:.1f}s, tables+ranking {t_tab:.1f}s; "
             f"{len(cmp.fuzzy)} intersections; peak RSS {peak_rss_mb():.0f} MB")
    print(r.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
