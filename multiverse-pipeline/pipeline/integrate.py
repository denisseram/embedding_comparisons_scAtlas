"""Train one embedding per (configuration, seed) of the full factorial design.

Seed variation for near-deterministic methods (modelling decision, see README):
  * PCA is FIT on a seed-specific random subsample (config: integration.pca_fit_fraction,
    default 90%) of the cells and then applied to all cells.
  * Harmony additionally receives random_state=seed.
  * ComBat itself is deterministic; its seed variation comes from the PCA subsample.
The subsample depends only on the seed index, so seed i uses the same cells in every
configuration. Matched contrasts therefore only pair different seed indices (see measures.py).
"""
from __future__ import annotations

import itertools
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA

from pipeline.common import append_runtime, get_logger, load_config, log_decision, out_dir
from pipeline.preprocess import hvg_key, load_lognorm

log = get_logger("integrate")
warnings.filterwarnings("ignore", category=FutureWarning)

FACTORS = ["n_hvg", "hvg_batch_aware", "exclude_igx", "batch_key", "method"]


def config_id(c: dict) -> str:
    return (f"{c['method']}.n{c['n_hvg']}.ba{int(c['hvg_batch_aware'])}"
            f".ex{int(c['exclude_igx'])}.{c['batch_key']}")


def model_id(c: dict, seed: int) -> str:
    return f"{config_id(c)}.s{seed}"


def design(cfg: dict) -> pd.DataFrame:
    d = cfg["decisions"]
    methods = list(d["method"])
    if cfg["integration"].get("use_scvi"):
        try:
            import scvi  # noqa: F401
            methods.append("scvi")
        except Exception as e:  # pragma: no cover - optional dependency
            log_decision(cfg, f"scvi requested but not importable ({e!r}); scVI runs skipped.")
    rows = []
    for n, ba, ex, bk, m in itertools.product(d["n_hvg"], d["hvg_batch_aware"], d["exclude_igx"],
                                              d["batch_key"], methods):
        c = dict(n_hvg=n, hvg_batch_aware=bool(ba), exclude_igx=bool(ex), batch_key=bk, method=m)
        for s in d["seeds"]:
            rows.append({"model_id": model_id(c, s), "config_id": config_id(c), **c, "seed": s})
    return pd.DataFrame(rows)


def fit_pca(X: np.ndarray, n_pcs: int, frac: float, seed: int, global_seed: int) -> np.ndarray:
    rng = np.random.default_rng([global_seed, seed])
    fit_idx = rng.choice(X.shape[0], int(round(frac * X.shape[0])), replace=False)
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=global_seed)
    pca.fit(X[fit_idx])
    return pca.transform(X).astype(np.float32)


def run_harmony(Z: np.ndarray, obs: pd.DataFrame, key: str, seed: int) -> np.ndarray:
    import harmonypy
    ho = harmonypy.run_harmony(Z, obs[[key]].copy(), [key], random_state=seed, verbose=False)
    out = np.asarray(ho.Z_corr)
    if out.shape[0] != Z.shape[0]:
        out = out.T
    return out.astype(np.float32)


def run_scvi(adata, genes, key, seed, cfg):  # pragma: no cover - optional path
    import scvi
    sub = adata[:, genes].copy()
    sub.X = sub.layers["counts"]
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(sub, batch_key=key)
    m = scvi.model.SCVI(sub, n_latent=cfg["integration"]["n_pcs"])
    m.train(max_epochs=cfg["integration"]["scvi_epochs"], accelerator="cpu")
    return m.get_latent_representation().astype(np.float32)


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    t_all = time.perf_counter()
    import json
    sets = json.loads((out_dir(cfg) / "hvg_sets.json").read_text())
    adata = load_lognorm(cfg)
    obs = adata.obs
    ic, gs = cfg["integration"], cfg["global_seed"]
    lat_dir = out_dir(cfg, "latent")
    models = design(cfg)
    log.info(f"{len(models)} embeddings, {models.config_id.nunique()} configurations")

    scaled_cache: dict[str, np.ndarray] = {}
    combat_cache: dict[tuple, np.ndarray] = {}

    def scaled(hk: str) -> np.ndarray:
        if hk not in scaled_cache:
            scaled_cache.clear()  # keep memory low: only one HVG set resident at a time
            combat_cache.clear()
            sub = adata[:, sets[hk]].copy()
            sub.X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
            scaled_cache[hk] = sub
        return scaled_cache[hk]

    def matrix(hk: str, method: str, key: str) -> np.ndarray:
        sub = scaled(hk)
        if method == "combat_pca":
            ck = (hk, key)
            if ck not in combat_cache:
                c = sub.copy()
                sc.pp.combat(c, key=key)
                c.X = np.array(c.X, dtype=np.float32, copy=True)  # combat output is read-only (pandas CoW)
                sc.pp.scale(c, max_value=10)
                combat_cache[ck] = np.asarray(c.X, dtype=np.float32)
            return combat_cache[ck]
        ck = (hk, "scaled")
        if ck not in combat_cache:
            c = sub.copy()
            sc.pp.scale(c, max_value=10)
            combat_cache[ck] = np.asarray(c.X, dtype=np.float32)
        return combat_cache[ck]

    # group by HVG set so each scaled matrix is built once
    models["hvg_set"] = [hvg_key(r.n_hvg, r.hvg_batch_aware, r.exclude_igx, r.batch_key)
                         for r in models.itertuples()]
    runtimes = {}
    pcs_cache: dict[tuple, np.ndarray] = {}
    for hk, grp in models.groupby("hvg_set", sort=True):
        pcs_cache.clear()
        for r in grp.itertuples():
            t0 = time.perf_counter()
            if r.method == "scvi":
                Z = run_scvi(adata, sets[hk], r.batch_key, r.seed, cfg)
            else:
                pre = "combat" if r.method == "combat_pca" else "scaled"
                pk = (pre, r.batch_key if pre == "combat" else "", r.seed)
                if pk not in pcs_cache:
                    X = matrix(hk, r.method, r.batch_key)
                    pcs_cache[pk] = fit_pca(X, ic["n_pcs"], ic["pca_fit_fraction"], r.seed, gs)
                Z = pcs_cache[pk]
                if r.method == "harmony":
                    Z = run_harmony(Z, obs, r.batch_key, r.seed)
            np.save(lat_dir / f"{r.model_id}.npy", Z)
            runtimes[r.model_id] = round(time.perf_counter() - t0, 3)
        log.info(f"{hk}: {len(grp)} embeddings done")
    models["runtime_s"] = models.model_id.map(runtimes)
    models.drop(columns="hvg_set").to_csv(out_dir(cfg) / "models.csv", index=False)
    append_runtime(cfg, "integrate", time.perf_counter() - t_all)
    return models


if __name__ == "__main__":
    main()
