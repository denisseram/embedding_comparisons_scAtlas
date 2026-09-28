"""Write compact, web-ready files into the Astro site's public/multiverse-data/ folder.

Every binary file is little-endian, row-major, without header; its dtype and shape are listed in
manifest.json['files'] and documented in the generated README.md next to the data.
Also computes the two remaining layouts: the fixed cell map (UMAP of the consensus kNN graph,
used only for location) and the model map (UMAP with precomputed A(a,b), and classical MDS).
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import time
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pipeline.common import append_runtime, get_logger, load_config, out_dir, p
from pipeline.integrate import FACTORS

log = get_logger("export")
warnings.filterwarnings("ignore")
SCHEMA_VERSION = "1.2.0"  # 1.1: models.json layout gains "tsne"; 1.2: optional umap_models.bin


def classical_mds(D: np.ndarray, dim: int = 2) -> np.ndarray:
    n = D.shape[0]
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:dim]
    return (V[:, order] * np.sqrt(np.maximum(w[order], 0))).astype(np.float32)


def model_umap(D: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    import umap
    reducer = umap.UMAP(metric="precomputed", n_neighbors=n_neighbors, random_state=seed, init="random")
    return reducer.fit_transform(D.astype(np.float64)).astype(np.float32)


def model_tsne(D: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    perp = min(perplexity, (D.shape[0] - 1) / 3)  # t-SNE requires perplexity < n_samples
    tsne = TSNE(n_components=2, metric="precomputed", init="random", perplexity=perp, random_state=seed)
    return tsne.fit_transform(D.astype(np.float64)).astype(np.float32)


def cell_map(cons: np.ndarray, seed: int) -> np.ndarray:
    """UMAP of cells from the consensus kNN graph (binary, symmetrised). Location only."""
    import scanpy as sc
    n, k = cons.shape
    W = sp.csr_matrix((np.ones(n * k, np.float32), (np.repeat(np.arange(n), k), cons.ravel())), shape=(n, n))
    W = ((W + W.T) > 0).astype(np.float32)
    a = ad.AnnData(X=np.zeros((n, 1), np.float32))
    a.obsp["connectivities"] = W.tocsr()
    a.obsp["distances"] = W.tocsr()
    a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"method": "umap", "n_neighbors": k}}
    sc.tl.umap(a, random_state=seed, init_pos="spectral")
    return a.obsm["X_umap"].astype(np.float32)


def rnd(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def encode_cat(s: pd.Series) -> dict:
    c = pd.Categorical(s.astype(str))
    return {"codes": c.codes.astype(int).tolist(), "levels": [str(x) for x in c.categories]}


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    gs = cfg["global_seed"]
    od, md = out_dir(cfg), out_dir(cfg, "measures")
    ex = p(cfg, "export_dir")
    if ex.exists():
        shutil.rmtree(ex)
    ex.mkdir(parents=True)
    meta = json.loads((md / "meta.json").read_text())
    modes, primary = meta["modes"], meta["modes"][0]
    sfx = {m: ("" if m == primary else f"_{m}") for m in modes}
    models = pd.read_csv(od / "models.csv")
    M = len(models)
    obs = ad.read_h5ad(p(cfg, "data"), backed="r").obs.copy()
    N = len(obs)
    metrics = pd.read_csv(od / "metrics.csv", index_col=0).loc[models.model_id]
    seed_sd = pd.read_csv(od / "metrics_seed_sd.csv", index_col=0).loc[models.model_id]
    info = json.loads((od / "metric_info.json").read_text())
    files: dict[str, dict] = {}

    def write_bin(name: str, arr: np.ndarray, dtype, desc: str, order: str):
        arr = np.ascontiguousarray(arr, dtype=dtype)
        (ex / name).write_bytes(arr.astype(arr.dtype.newbyteorder("<")).tobytes())
        files[name] = {"dtype": np.dtype(dtype).name, "shape": list(arr.shape), "description": desc, "order": order}

    def write_json(name: str, obj, desc: str):
        (ex / name).write_text(json.dumps(obj, separators=(",", ":")))
        files[name] = {"dtype": "json", "description": desc}

    # ---- layouts --------------------------------------------------------------------------
    agreement = {m: np.load(md / f"agreement{sfx[m]}.npy") for m in modes}
    layouts = {m: {"umap": model_umap(agreement[m], cfg["layouts"]["model_umap_n_neighbors"], gs),
                   "tsne": model_tsne(agreement[m], cfg["layouts"].get("model_tsne_perplexity", 15), gs),
                   "mds": classical_mds(agreement[m])} for m in modes}
    cons = np.load(md / "consensus_knn.npy")
    xy = cell_map(cons, gs)

    # ---- models.json ------------------------------------------------------------------------
    names = [d["name"] for d in info]
    share = np.load(md / "model_consensus_share.npy")
    recs = []
    for i, r in enumerate(models.itertuples()):
        mm = metrics.loc[r.model_id]
        sd = seed_sd.loc[r.model_id]
        recs.append({
            "model_id": r.model_id, "config_id": r.config_id, "seed": int(r.seed),
            "factors": {f: (bool(getattr(r, f)) if isinstance(getattr(r, f), (bool, np.bool_)) else
                            (int(getattr(r, f)) if f == "n_hvg" else str(getattr(r, f)))) for f in FACTORS},
            "runtime_s": rnd(r.runtime_s, 3),
            "raw": {n: rnd(mm[n]) for n in names},
            "scaled": {n: rnd(mm[f"{n}_scaled"]) for n in names},
            "overall": rnd(mm.overall), "bio": rnd(mm.bio), "batch": rnd(mm.batch),
            "seed_sd": {**{n: rnd(sd[n], 5) for n in names}, **{f"{n}_scaled": rnd(sd[f"{n}_scaled"], 5) for n in names},
                        "overall": rnd(sd.overall, 5), "bio": rnd(sd.bio, 5), "batch": rnd(sd.batch, 5)},
            "layout": {m: {"umap": [rnd(v, 3) for v in layouts[m]["umap"][i]],
                           "tsne": [rnd(v, 3) for v in layouts[m]["tsne"][i]],
                           "mds": [rnd(v, 4) for v in layouts[m]["mds"][i]]} for m in modes},
            "summary": {"consensus_share": rnd(share[i], 5),
                        **{f"mean_agreement_delta_{m}": rnd(np.delete(agreement[m][i], i).mean()) for m in modes}},
        })
    write_json("models.json", recs, "Per-model factor levels, metrics (raw/scaled), aggregates, seed-replicate sd, "
                                    "2D model-map coordinates per measure mode, summary measures. Order = manifest.model_order.")

    for m in modes:
        write_bin(f"agreement{sfx[m]}.bin", agreement[m], np.float32,
                  f"A(a,b) = mean Δ over a stratified sample of {len(np.load(md / 'agreement_cells.npy'))} cells "
                  f"(measure: {m})", "[model_a, model_b], model order = manifest.model_order")

    # ---- cells.json -------------------------------------------------------------------------
    gt = pd.read_csv(p(cfg, "ground_truth"), index_col=0).loc[obs.index]
    cells = {
        "id": obs.index.tolist(),
        "x": [rnd(v, 3) for v in xy[:, 0]], "y": [rnd(v, 3) for v in xy[:, 1]],
        "categorical": {c: encode_cat(obs[c]) for c in ["cell_type", "study", "sample", "tissue", "planted_effect"]},
        "numeric": {c: [rnd(v, 4) for v in obs[c].to_numpy(float)] for c in ["total_counts", "pct_mito", "doublet_score"]},
        "planted_membership": {c: gt[c].to_numpy(bool).astype(int).tolist()
                               for c in gt.columns if c != "planted_effect"},
    }
    write_json("cells.json", cells, "Per cell: id, fixed-map x/y (UMAP of the consensus kNN graph; location only), "
                                    "dictionary-encoded obs columns, numeric QC columns, planted-effect membership.")

    # ---- cell_measures.bin ------------------------------------------------------------------
    factors = meta["factors"]
    arrays, layout = [], []

    def add(name, arr, desc):
        arrays.append(np.asarray(arr, np.float32))
        layout.append({"name": name, "description": desc})

    add("stability_all", np.load(md / "stability_all.npy"), "consensus stability over all models")
    add("stability_seed", np.load(md / "stability_seed.npy"), "consensus stability over seed replicates only (mean over configurations)")
    add("majority_vote_disagreement", np.load(od / "cell_majority_disagreement.npy"), "paper-style baseline")
    add("paper_graph_dissimilarity", np.load(md / "paper_graph_dissim.npy"), "paper-style baseline (mean |d| over matched pairs)")
    for m in modes:
        E, H, Er = (np.load(md / f"{n}{sfx[m]}.npy") for n in ["E", "H", "E_rank"])
        for i, F in enumerate(factors):
            add(f"E.{m}.{F}", E[:, i], f"effect E(c,{F}), measure {m}")
            add(f"H.{m}.{F}", H[:, i], f"interaction H(c,{F}), measure {m}")
            add(f"Erank.{m}.{F}", Er[:, i], f"rank-based effect, measure {m}")
        add(f"region.{m}", np.load(md / f"regions{sfx[m]}.npy"), f"region id (measure {m})")
    write_bin("cell_measures.bin", np.stack(arrays), np.float32, "per-cell measure arrays; array names in manifest.cell_measures",
              "[array, cell]")

    # ---- pair_delta.bin ---------------------------------------------------------------------
    pairs = pd.read_csv(md / "pairs.csv")
    for m in modes:
        write_bin(f"pair_delta{sfx[m]}.bin", np.load(md / f"pair_delta{sfx[m]}.npy"), np.float32,
                  f"seed-averaged Δ̄ per cell and matched one-factor pair (measure {m})", "[cell, pair], pair order = manifest.pairs")

    # ---- z_vs_ref.bin -----------------------------------------------------------------------
    refs = meta["references"]
    z = np.load(md / "z_vs_ref.npy")
    write_bin("z_vs_ref.bin", z, np.float32, f"calibrated change z(c, ref, model), measure {primary}",
              "[reference, cell, model], reference order = manifest.references")
    for m in modes[1:]:
        write_bin(f"z_vs_ref{sfx[m]}.bin", np.load(md / f"z_vs_ref{sfx[m]}.npy")[:1], np.float32,
                  f"calibrated change z(c, ref, model) for the first reference only, measure {m}",
                  "[reference, cell, model]")

    # ---- regions.json -----------------------------------------------------------------------
    reg_out = {}
    vcode = ["typical", "merged", "split", "joined"]
    for m in modes:
        tbl = json.loads((md / f"regions{sfx[m]}.json").read_text())
        var = pd.read_csv(md / f"variants{sfx[m]}.csv")
        for rec in tbl["regions"]:
            v = var[var.region == rec["region"]].set_index("model_id").loc[models.model_id]
            rec["variants"] = {"code": [vcode.index(x) for x in v.variant], "n_sub": v.n_sub.astype(int).tolist(),
                               "outside_frac": [rnd(x, 3) for x in v.outside_frac],
                               "joined_to": v.joined_to.astype(int).tolist(), "modal_n_sub": int(v.modal_n_sub.iloc[0])}
        reg_out[m] = tbl
    write_json("regions.json", {"variant_levels": vcode, "by_mode": reg_out},
               "Per measure mode: regions with size, compositions, QC summaries, top genes, mean E/H, and the local "
               "variant of every model (arrays in manifest.model_order).")

    # ---- neighbors_sample.bin ---------------------------------------------------------------
    knn = np.load(od / "knn50.npy", mmap_mode="r")
    ref = refs[0]
    rrow = models.set_index("model_id").loc[ref]
    base = f".n{rrow.n_hvg}.ba{int(rrow.hvg_batch_aware)}.ex{int(rrow.exclude_igx)}.{rrow.batch_key}.s{rrow.seed}"
    chosen = [ref] + [f"{meth}{base}" for meth in ["pca", "harmony", "combat_pca"] if f"{meth}{base}" != ref]
    chosen = [c for c in chosen if c in set(models.model_id)][: cfg["export"]["n_neighbor_models"]]
    idx = [int(np.flatnonzero(models.model_id == c)[0]) for c in chosen]
    k = meta["k"]
    write_bin("neighbors_sample.bin", np.asarray(knn[idx, :, :k]), np.uint32,
              f"k={k} nearest-neighbour indices for the models in manifest.neighbor_models", "[model, cell, neighbour]")

    # ---- umap_models.bin (per-embedding UMAPs for visual QC, optional) ------------------------
    umap_info = None
    upath = od / "umap_per_model.npy"
    if upath.exists():
        U = np.load(upath)
        lo, hi = U.min(axis=1), U.max(axis=1)                    # [M, 2]
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        q = np.round((U - lo[:, None, :]) / span[:, None, :] * 65535).astype(np.uint16)
        write_bin("umap_models.bin", q, np.uint16,
                  "per-embedding UMAP (scanpy neighbors on the model's latent + tl.umap), quantised: "
                  "x = min + q/65535*(max-min) with min/max per model in manifest.umap_models; visual QC only",
                  "[model, cell, (x, y)]")
        umap_info = {"min": lo.round(4).tolist(), "max": hi.round(4).tolist(),
                     "recipe": f"scanpy pp.neighbors(use_rep=latent, n_neighbors={cfg['per_model_umap'].get('n_neighbors', 15)}) + tl.umap, random_state={gs}"}

    # ---- manifest ---------------------------------------------------------------------------
    for name in files:
        files[name]["bytes"] = (ex / name).stat().st_size
    d = cfg["decisions"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"name": cfg["name"], "n_cells": N, "n_genes": int(ad.read_h5ad(p(cfg, "data"), backed="r").n_vars),
                    "n_models": M, "n_configs": int(models.config_id.nunique()), "n_pairs": len(pairs),
                    "seeds": [int(s) for s in sorted(models.seed.unique())]},
        "factors": [{"name": f, "levels": [str(x) if not isinstance(x, bool) else x for x in
                                          (d[f] if f != "method" else sorted(models.method.unique(), key=list(models.method.unique()).index))]}
                    for f in FACTORS],
        "measure_factors": factors,
        "measure": {"k": k, "modes": modes, "primary_mode": primary,
                    "mode_labels": {"jaccard": "Neighbour identity (pre-registered)",
                                    "composition": "Neighbour composition (exploratory)"}},
        "metrics": info,
        "aggregates": [{"name": "overall", "group": "aggregate"}, {"name": "bio", "group": "aggregate"},
                       {"name": "batch", "group": "aggregate"}],
        "weights": cfg["metrics"]["weights"],
        "model_order": models.model_id.tolist(),
        "configs": meta["configs"],
        "references": refs,
        "neighbor_models": chosen,
        "pairs": pairs.to_dict("records"),
        "cell_measures": [{"index": i, **l} for i, l in enumerate(layout)],
        "umap_models": umap_info,
        "files": files,
    }
    (ex / "manifest.json").write_text(json.dumps(manifest, indent=1))
    files["manifest.json"] = {"bytes": (ex / "manifest.json").stat().st_size, "dtype": "json",
                              "description": "this index"}
    (ex / "README.md").write_text(readme(manifest, files))
    total = sum(f.stat().st_size for f in ex.iterdir())
    log.info(f"exported {len(files)} files, {total / 1e6:.2f} MB total to {ex}")
    if total > cfg["export"]["budget_mb"] * 1e6:
        log.warning(f"export exceeds budget of {cfg['export']['budget_mb']} MB")
    append_runtime(cfg, "export", time.perf_counter() - t0)
    return manifest


def readme(manifest: dict, files: dict) -> str:
    lines = ["# multiverse-data", "",
             "Generated by `multiverse-pipeline/pipeline/export.py` — do not edit by hand.", "",
             f"Schema version {manifest['schema_version']}, generated {manifest['generated_at']}. "
             f"{manifest['dataset']['n_cells']} cells, {manifest['dataset']['n_models']} models, "
             f"{manifest['dataset']['n_pairs']} matched pairs.", "",
             "All binary files are little-endian, row-major, headerless. Shapes and orders reference "
             "lists in `manifest.json` (`model_order`, `pairs`, `references`, `cell_measures`, `neighbor_models`).", "",
             "| file | dtype | shape | order | bytes | description |", "|---|---|---|---|---|---|"]
    for name, f in sorted(files.items()):
        lines.append(f"| `{name}` | {f.get('dtype', '')} | {f.get('shape', '')} | {f.get('order', '')} | "
                     f"{f.get('bytes', '')} | {f.get('description', '')} |")
    lines += ["", "## cell_measures.bin arrays", "", "| index | name | description |", "|---|---|---|"]
    lines += [f"| {c['index']} | `{c['name']}` | {c['description']} |" for c in manifest["cell_measures"]]
    lines += ["", "Region ids are stored as floats in `cell_measures.bin`; cast to integers.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
