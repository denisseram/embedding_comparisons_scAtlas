"""Local variants: how each model structures each region.

For region R and model m, on m's own k-NN graph (k = measures.k):
  n_sub        number of Leiden sub-clusters of R's cells (induced subgraph, resolution
               variants.sub_resolution) holding at least variants.min_subcluster_frac of R
  outside_frac fraction of R-cells' neighbours in m that lie outside R
  joined_to    the region receiving most of those outside neighbours
Label rules (in order):
  joined   outside_frac >= variants.joined_min_outside
  split    n_sub > modal n_sub over models
  merged   n_sub < modal n_sub over models
  typical  otherwise
"""
from __future__ import annotations

import json
import time

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd

from pipeline.common import append_runtime, get_logger, load_config, out_dir
from pipeline.knn import load_knn

log = get_logger("variants")


def local_structure(nbrs: np.ndarray, members: np.ndarray, regions: np.ndarray, r: int,
                    res: float, min_frac: float, seed: int) -> tuple[int, float, int]:
    nb = nbrs[members]
    inside = regions[nb] == r
    outside_frac = float(1 - inside.mean())
    other = regions[nb][~inside]
    joined_to = int(np.bincount(other).argmax()) if other.size else -1
    local = np.full(regions.size, -1, np.int64)
    local[members] = np.arange(members.size)
    src = np.repeat(np.arange(members.size), nb.shape[1])[inside.ravel()]
    dst = local[nb.ravel()[inside.ravel()]]
    g = ig.Graph(n=members.size, edges=np.column_stack([src, dst]), directed=False).simplify()
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=res, seed=seed)
    sizes = np.bincount(part.membership)
    n_sub = int((sizes >= min_frac * members.size).sum())
    return max(n_sub, 1), outside_frac, joined_to


def main(cfg: dict | None = None, mode: str | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    modes = cfg["measures"].get("modes", ["jaccard"])
    mode = mode or modes[0]
    sfx = "" if mode == modes[0] else f"_{mode}"
    t0 = time.perf_counter()
    vc, k = cfg["variants"], cfg["measures"]["k"]
    od = out_dir(cfg, "measures")
    regions = np.load(od / f"regions{sfx}.npy")
    models = pd.read_csv(out_dir(cfg) / "models.csv")
    knn = load_knn(cfg)
    rows = []
    members = [np.flatnonzero(regions == r) for r in range(regions.max() + 1)]
    for m, mid in enumerate(models.model_id):
        nbrs = np.asarray(knn[m, :, :k])
        for r, mem in enumerate(members):
            n_sub, out_f, jt = local_structure(nbrs, mem, regions, r, vc["sub_resolution"],
                                               vc["min_subcluster_frac"], cfg["global_seed"])
            rows.append({"model_id": mid, "region": r, "n_sub": n_sub, "outside_frac": out_f, "joined_to": jt})
    df = pd.DataFrame(rows)
    modal = df.groupby("region").n_sub.agg(lambda x: x.mode().min())
    df["modal_n_sub"] = df.region.map(modal)
    df["variant"] = np.select(
        [df.outside_frac >= vc["joined_min_outside"], df.n_sub > df.modal_n_sub, df.n_sub < df.modal_n_sub],
        ["joined", "split", "merged"], "typical")
    df.to_csv(od / f"variants{sfx}.csv", index=False)
    summary = df.groupby("region").variant.value_counts().unstack(fill_value=0)
    log.info("variant counts per region:\n" + summary.to_string())
    append_runtime(cfg, f"variants_{mode}", time.perf_counter() - t0)
    return df


if __name__ == "__main__":
    import sys
    main(mode=sys.argv[1] if len(sys.argv) > 1 else None)
