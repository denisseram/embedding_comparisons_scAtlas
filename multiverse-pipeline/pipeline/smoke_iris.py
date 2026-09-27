"""30-second plumbing smoke test on Iris (NOT the main system: no batches, 4 features).

Decisions: standardize {no, yes} x n_pcs {2, 3} x UMAP n_neighbors {5, 15}, seeds {0, 1, 2}
(seed = UMAP random_state). Runs kNN, Δ, seed noise, matched pairs, E and agreement with the
same functions as the main pipeline and asserts shapes and finiteness.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline.knn import knn
from pipeline.measures import consensus, delta, matched_pairs, smooth


def main():
    t0 = time.perf_counter()
    import umap
    X = load_iris().data
    factors = ["standardize", "n_pcs", "umap_nn"]
    rows, lat = [], []
    for std, npc, nn in itertools.product([False, True], [2, 3], [5, 15]):
        Z0 = StandardScaler().fit_transform(X) if std else X - X.mean(0)
        Z0 = PCA(npc, random_state=0).fit_transform(Z0)
        for s in range(3):
            lat.append(umap.UMAP(n_neighbors=nn, random_state=s, n_jobs=1).fit_transform(Z0))
            rows.append({"standardize": std, "n_pcs": npc, "umap_nn": nn, "seed": s,
                         "config_id": f"std{int(std)}.pc{npc}.nn{nn}"})
    models = pd.DataFrame(rows)
    k = 10
    nbrs = np.stack([knn(Z, k, 0, "exact")[0] for Z in lat])
    cons, stab = consensus(nbrs, k, 0.8)
    configs = models.drop_duplicates("config_id")[["config_id", *factors]]
    pairs = matched_pairs(configs, factors)
    by = {c: np.flatnonzero(models.config_id == c) for c in configs.config_id}
    seed_d = {c: np.stack([delta(nbrs[a], nbrs[b]) for a, b in itertools.combinations(i, 2)]) for c, i in by.items()}
    E = {}
    for F in factors:
        zs = []
        for r in pairs[pairs.factor == F].itertuples():
            A, B = by[r.config_a], by[r.config_b]
            d = np.mean([delta(nbrs[a], nbrs[b]) for a in A for b in B if a % 3 != b % 3], 0)
            mu = smooth((seed_d[r.config_a].mean(0) + seed_d[r.config_b].mean(0)) / 2, cons)
            sd = np.sqrt(smooth((seed_d[r.config_a].var(0, ddof=1) + seed_d[r.config_b].var(0, ddof=1)) / 2, cons))
            zs.append((d - mu) / np.maximum(sd, 0.05))
        E[F] = np.mean(zs, 0)
    assert nbrs.shape == (24, 150, k) and len(pairs) == 3 * 4
    assert all(np.isfinite(v).all() for v in E.values()) and np.isfinite(stab).all()
    print("iris smoke OK:", {F: round(float(np.median(np.abs(v))), 2) for F, v in E.items()},
          f"in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
