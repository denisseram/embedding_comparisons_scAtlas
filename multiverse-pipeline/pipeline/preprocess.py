"""Normalisation and HVG selection for every distinct feature-selection decision.

HVG sets depend on (n_hvg, hvg_batch_aware, exclude_igx) and, only when batch-aware,
on batch_key. Each distinct set is computed once and cached in hvg_sets.json.
"""
from __future__ import annotations

import itertools
import json
import time
import warnings

import anndata as ad
import numpy as np
import scanpy as sc

from pipeline.common import append_runtime, get_logger, load_config, out_dir, p

log = get_logger("preprocess")
warnings.filterwarnings("ignore", category=FutureWarning)


def hvg_key(n_hvg: int, batch_aware: bool, exclude_igx: bool, batch_key: str) -> str:
    bk = batch_key if batch_aware else "none"
    return f"n{n_hvg}_ba-{bk}_igx-{'excl' if exclude_igx else 'incl'}"


def load_lognorm(cfg: dict) -> ad.AnnData:
    adata = ad.read_h5ad(p(cfg, "data"))
    adata.X = adata.layers["counts"].astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def select_hvgs(adata: ad.AnnData, n_hvg: int, batch_aware: bool, exclude_igx: bool,
                batch_key: str) -> np.ndarray:
    keep = ~adata.var["igx"].to_numpy(bool) if exclude_igx else np.ones(adata.n_vars, bool)
    sub = adata[:, keep]
    hv = sc.pp.highly_variable_genes(sub, flavor="seurat", n_top_genes=n_hvg, inplace=False,
                                     batch_key=batch_key if batch_aware else None)
    if batch_aware:  # seurat flavor with batch_key ranks by n_batches, then mean dispersion
        order = np.lexsort((-hv["dispersions_norm"].fillna(-np.inf).to_numpy(),
                            -hv["highly_variable_nbatches"].to_numpy()))
        chosen = order[:n_hvg]
    else:
        chosen = np.flatnonzero(hv["highly_variable"].to_numpy())
    genes = np.flatnonzero(keep)[chosen]
    return np.sort(genes)


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    adata = load_lognorm(cfg)
    d = cfg["decisions"]
    sets = {}
    for n, ba, ex, bk in itertools.product(d["n_hvg"], d["hvg_batch_aware"], d["exclude_igx"], d["batch_key"]):
        key = hvg_key(n, ba, ex, bk)
        if key not in sets:
            sets[key] = select_hvgs(adata, n, ba, ex, bk).tolist()
    for k, v in sets.items():
        n_igx = int(adata.var["igx"].to_numpy(bool)[v].sum())
        grp = adata.var["gene_group"].to_numpy(dtype=str)[v]
        log.info(f"{k}: {len(v)} genes, IGX={n_igx}, P1 weak markers={int(np.isin(grp, ['p1_TA', 'p1_TB']).sum())}")
    (out_dir(cfg) / "hvg_sets.json").write_text(json.dumps(sets))
    append_runtime(cfg, "preprocess", time.perf_counter() - t0)
    return sets


if __name__ == "__main__":
    main()
