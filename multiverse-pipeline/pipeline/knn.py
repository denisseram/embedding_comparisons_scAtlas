"""kNN graphs in each embedding's own latent space (idea I1).

k_large (50) neighbours are computed once per embedding with pynndescent (fixed
random_state); the k_small (15) graph is the first 15 columns of that sorted list.
Self is removed. Stored as one stacked array  knn50.npy : int32 [n_models, n_cells, 50]
in models.csv order, plus knn50_dist.npy (float32, same shape).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from pipeline.common import append_runtime, get_logger, load_config, out_dir

log = get_logger("knn")


def knn(X: np.ndarray, k: int, seed: int, backend: str = "pynndescent") -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, distances) of the k nearest neighbours of every row, self excluded."""
    n = X.shape[0]
    if backend == "exact" or n <= 2000:
        from sklearn.neighbors import NearestNeighbors
        d, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    else:
        from pynndescent import NNDescent
        index = NNDescent(X, n_neighbors=k + 1, random_state=seed, n_jobs=1, low_memory=True)
        idx, d = index.neighbor_graph
    # drop self wherever it appears (ties can put self off column 0)
    is_self = idx == np.arange(n)[:, None]
    out_i = np.empty((n, k), np.int32)
    out_d = np.empty((n, k), np.float32)
    has_self = is_self.any(1)
    out_i[has_self] = idx[has_self][~is_self[has_self]].reshape(-1, k)
    out_d[has_self] = d[has_self][~is_self[has_self]].reshape(-1, k)
    out_i[~has_self] = idx[~has_self, :k]
    out_d[~has_self] = d[~has_self, :k]
    return out_i, out_d


def load_knn(cfg: dict, k: int | None = None, mmap: bool = True) -> np.ndarray:
    arr = np.load(out_dir(cfg) / "knn50.npy", mmap_mode="r" if mmap else None)
    return arr if k is None else arr[:, :, :k]


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    models = pd.read_csv(out_dir(cfg) / "models.csv")
    kc = cfg["knn"]
    lat_dir = out_dir(cfg, "latent")
    n = np.load(lat_dir / f"{models.model_id[0]}.npy", mmap_mode="r").shape[0]
    M, K = len(models), kc["k_large"]
    I = np.lib.format.open_memmap(out_dir(cfg) / "knn50.npy", mode="w+", dtype=np.int32, shape=(M, n, K))
    D = np.lib.format.open_memmap(out_dir(cfg) / "knn50_dist.npy", mode="w+", dtype=np.float32, shape=(M, n, K))
    seen: dict[str, int] = {}
    for m, mid in enumerate(models.model_id):
        X = np.load(lat_dir / f"{mid}.npy")
        h = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()
        if h in seen:  # identical latents (e.g. pca with batch-unaware HVGs across batch_key)
            I[m], D[m] = I[seen[h]], D[seen[h]]
        else:
            seen[h] = m
            I[m], D[m] = knn(X, K, cfg["global_seed"], kc["backend"])
        if (m + 1) % 24 == 0:
            log.info(f"{m + 1}/{M} kNN graphs")
    I.flush(), D.flush()
    append_runtime(cfg, "knn", time.perf_counter() - t0)
    log.info(f"kNN done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
