"""Per-embedding UMAPs for visual QC of each integration (the standard scanpy recipe).

For every model: sc.pp.neighbors(use_rep=<its latent>, n_neighbors) + sc.tl.umap, fixed seed.
These maps are for *looking at* an integration (batch mixing, cell-type separation). No
multiverse measure is computed from them. Identical latents (e.g. `pca` with batch-unaware
HVGs across batch_key) are computed once. Output: umap_per_model.npy float32 [n_models, n_cells, 2].
"""
from __future__ import annotations

import hashlib
import time
import warnings

import numpy as np
import pandas as pd

from pipeline.common import append_runtime, get_logger, load_config, log_decision, out_dir

log = get_logger("umaps")
warnings.filterwarnings("ignore")


def umap_of(latent: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(np.zeros((latent.shape[0], 1), np.float32))
    a.obsm["X_emb"] = latent
    sc.pp.neighbors(a, use_rep="X_emb", n_neighbors=n_neighbors, random_state=seed)
    sc.tl.umap(a, random_state=seed)
    return a.obsm["X_umap"].astype(np.float32)


def _work(args):
    path, n_neighbors, seed = args
    return umap_of(np.load(path), n_neighbors, seed)


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    uc = cfg.get("per_model_umap", {"enabled": False})
    if not uc.get("enabled", False):
        log_decision(cfg, "per-model UMAPs disabled in config; the embedding viewer will be unavailable.")
        return
    t0 = time.perf_counter()
    models = pd.read_csv(out_dir(cfg) / "models.csv")
    lat_dir = out_dir(cfg, "latent")
    # deduplicate identical latents
    first: dict[str, int] = {}
    owner = np.empty(len(models), int)
    for i, mid in enumerate(models.model_id):
        h = hashlib.sha1(np.load(lat_dir / f"{mid}.npy").tobytes()).hexdigest()
        owner[i] = first.setdefault(h, i)
    unique = sorted(set(owner.tolist()))
    jobs = [(str(lat_dir / f"{models.model_id[i]}.npy"), uc.get("n_neighbors", 15), cfg["global_seed"]) for i in unique]
    n_jobs = int(cfg.get("n_jobs", 1))
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_jobs) as ex:
            res = list(ex.map(_work, jobs))
    else:
        res = [_work(j) for j in jobs]
    by_owner = dict(zip(unique, res))
    n = res[0].shape[0]
    out = np.lib.format.open_memmap(out_dir(cfg) / "umap_per_model.npy", mode="w+", dtype=np.float32, shape=(len(models), n, 2))
    for i in range(len(models)):
        out[i] = by_owner[owner[i]]
    out.flush()
    append_runtime(cfg, "umaps", time.perf_counter() - t0)
    log.info(f"{len(unique)} unique UMAPs for {len(models)} models in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
