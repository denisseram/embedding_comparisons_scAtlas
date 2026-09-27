"""Regions: groups of cells with a similar decision-effect profile that are also neighbours.

Graph = consensus kNN graph (spatial constraint: only consensus neighbours can be linked),
edge weight = exp(-||p_i - p_j||^2 / median) on the standardised profile p = [E(c,F) for F]
(seed pseudo-factor excluded). Leiden (RB configuration) is run over a resolution grid and the
first resolution giving min_regions..max_regions regions (after merging regions smaller than
min_region_size into their best-connected neighbour) is kept.
"""
from __future__ import annotations

import json
import time
import warnings

import anndata as ad
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import scanpy as sc

from pipeline.common import append_runtime, get_logger, load_config, log_decision, out_dir, p
from pipeline.preprocess import load_lognorm

log = get_logger("regions")
warnings.filterwarnings("ignore", category=FutureWarning)


def weighted_graph(cons: np.ndarray, prof: np.ndarray) -> ig.Graph:
    n, k = cons.shape
    src = np.repeat(np.arange(n), k)
    dst = cons.ravel()
    d2 = ((prof[src] - prof[dst]) ** 2).sum(1)
    w = np.exp(-d2 / max(np.median(d2), 1e-9))
    g = ig.Graph(n=n, edges=np.column_stack([src, dst]), directed=False)
    g.es["weight"] = w
    return g.simplify(combine_edges="sum")


def merge_small(labels: np.ndarray, cons: np.ndarray, min_size: int) -> np.ndarray:
    labels = labels.copy()
    while True:
        sizes = np.bincount(labels)
        small = [l for l in np.flatnonzero(sizes) if sizes[l] < min_size]
        if not small:
            break
        l = small[0]
        m = labels == l
        nb = labels[cons[m]].ravel()
        nb = nb[nb != l]
        labels[m] = np.bincount(nb).argmax() if nb.size else np.argmax(sizes)
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int32)


def find_regions(cons: np.ndarray, E: np.ndarray, rc: dict, seed: int) -> tuple[np.ndarray, float]:
    prof = (E - E.mean(0)) / (E.std(0) + 1e-9)
    g = weighted_graph(cons, prof)
    fallback = None
    for res in rc["resolutions"]:
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                        resolution_parameter=res, seed=seed)
        lab = merge_small(np.asarray(part.membership), cons, rc["min_region_size"])
        n = lab.max() + 1
        if rc["min_regions"] <= n <= rc["max_regions"]:
            return lab, res
        if n < rc["min_regions"]:
            fallback = (lab, res)
    return fallback if fallback else (lab, res)


def region_table(adata: ad.AnnData, labels: np.ndarray, E: np.ndarray, H: np.ndarray,
                 factors: list[str], stab_all, stab_seed, n_top: int) -> list[dict]:
    obs = adata.obs
    ad_ = adata.copy()
    ad_.obs["region"] = pd.Categorical(labels.astype(str))
    sc.tl.rank_genes_groups(ad_, "region", method="wilcoxon", n_genes=n_top)
    rg = ad_.uns["rank_genes_groups"]
    qc_cols = ["total_counts", "pct_mito", "doublet_score"]
    glob = {c: (obs[c].mean(), obs[c].std()) for c in qc_cols}
    out = []
    for r in range(labels.max() + 1):
        m = labels == r
        rec = {"region": int(r), "size": int(m.sum())}
        for col in ["cell_type", "sample", "study", "tissue", "planted_effect"]:
            vc = obs[col][m].value_counts()
            rec[f"comp_{col}"] = {str(k): int(v) for k, v in vc.items() if v > 0}
        rec["qc"] = {}
        for c in qc_cols:
            x = obs[c].to_numpy(float)[m]
            rec["qc"][c] = {"mean": float(x.mean()), "median": float(np.median(x)),
                            "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75)),
                            "z_vs_all": float((x.mean() - glob[c][0]) / glob[c][1])}
        rec["qc_deviation"] = float(np.sqrt(sum(rec["qc"][c]["z_vs_all"] ** 2 for c in qc_cols)))
        rec["E_mean"] = {F: float(E[m, i].mean()) for i, F in enumerate(factors)}
        rec["H_mean"] = {F: float(H[m, i].mean()) for i, F in enumerate(factors)}
        rec["total_effect"] = float(sum(abs(v) for F, v in rec["E_mean"].items() if F != "seed"))
        rec["stability_all"] = float(stab_all[m].mean())
        rec["stability_seed"] = float(stab_seed[m].mean())
        names = [rg["names"][i][str(r)] for i in range(n_top)]
        lfc = [float(rg["logfoldchanges"][i][str(r)]) for i in range(n_top)]
        padj = [float(rg["pvals_adj"][i][str(r)]) for i in range(n_top)]
        rec["top_genes"] = [{"gene": g, "logfc": round(l, 3), "padj": pa} for g, l, pa in zip(names, lfc, padj)]
        top_ct = max(rec["comp_cell_type"], key=rec["comp_cell_type"].get)
        rec["label"] = f"R{r} {top_ct} ({100 * rec['comp_cell_type'][top_ct] / rec['size']:.0f}%)"
        out.append(rec)
    return out


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    od = out_dir(cfg, "measures")
    meta = json.loads((od / "meta.json").read_text())
    factors = meta["factors"]
    E, H = np.load(od / "E.npy"), np.load(od / "H.npy")
    cons = np.load(od / "consensus_knn.npy")
    real = [i for i, F in enumerate(factors) if F != "seed"]
    labels, res = find_regions(cons, E[:, real], cfg["regions"], cfg["global_seed"])
    n = labels.max() + 1
    if not (cfg["regions"]["min_regions"] <= n <= cfg["regions"]["max_regions"]):
        log_decision(cfg, f"regions: no resolution gave {cfg['regions']['min_regions']}-"
                          f"{cfg['regions']['max_regions']} regions; using {n} at resolution {res}.")
    log.info(f"{n} regions at resolution {res}")
    adata = load_lognorm(cfg)
    table = region_table(adata, labels, E, H, factors, np.load(od / "stability_all.npy"),
                         np.load(od / "stability_seed.npy"), cfg["regions"]["n_top_genes"])
    np.save(od / "regions.npy", labels)
    (od / "regions.json").write_text(json.dumps({"resolution": res, "regions": table}, indent=1))
    for r in sorted(table, key=lambda r: -r["total_effect"]):
        log.info(f"{r['label']:28s} n={r['size']:4d} total|E|={r['total_effect']:.2f} "
                 f"stab={r['stability_all']:.3f} qc_dev={r['qc_deviation']:.2f} "
                 f"planted={max(r['comp_planted_effect'], key=r['comp_planted_effect'].get)}")
    append_runtime(cfg, "regions", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
