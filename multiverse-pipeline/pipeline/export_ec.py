"""Export the integration embeddings for the `embedding-comparison` app (the Iris-based
embedding-stability framework: consensus kNN, magnitude, severity, agreement patterns, meta-map).

Written to <paths.ec_export_dir> (default ../embedding-comparison/public/data/integration/):
  meta.json      cells, categories, embeddings, ensembles (presets), neutral spaces, file index
  knn.bin        Uint16 [E, n, k_max]  kNN *within the cell subsample*, in each model's latent space
  layouts.bin    Float32 [E, n, 2]     each model's own UMAP (display only), normalised to the unit square
  neutral_<key>.bin Float32 [n, d]     neutral spaces for severity (no embedding sees them)

Choices (see README "Embedding-comparison framework"):
  * Cells: a stratified subsample (cell_type × planted_effect, proportional, n = ec.n_cells) so the
    framework's per-point SVG views stay responsive. kNN is recomputed within the subsample.
  * Neighbours come from the latent space the integration produces (what clustering uses);
    the per-model UMAP is only the drawing.
  * Severity smoothing: in a space with exact ties (identical true profiles), severity uses
    (dd + f)/(cc + f) with f = 5% of the median pairwise distance; otherwise the app's original formula.
  * Neutral spaces: "truth" = the simulator's noise-free, batch-free biological profile (20 PCs;
    simulation only); "expression" = PCA of the observed log-normalised expression of all genes,
    uncorrected (available on real data, but contains batch effects).
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import time

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from pipeline.common import append_runtime, get_logger, load_config, out_dir, p
from pipeline.measures import stratified_sample
from pipeline.preprocess import load_lognorm

log = get_logger("export_ec")
SCHEMA = "ec-1.0.0"
FEATS = ["n_hvg", "hvg_batch_aware", "exclude_igx"]


def unit_square(P: np.ndarray) -> np.ndarray:
    """Same as the app's lib/linalg normalize(): fit into [0,1]², preserving aspect ratio."""
    lo, hi = P.min(0), P.max(0)
    span = float((hi - lo).max()) or 1.0
    off = (1 - (hi - lo) / span) / 2
    return ((P - lo) / span + off).astype(np.float32)


def short_name(r) -> str:
    return (f"{r.method} · {r.batch_key} · {r.n_hvg} · {'BA' if r.hvg_batch_aware else 'noBA'} · "
            f"{'−IGX' if r.exclude_igx else '+IGX'} · s{r.seed}")


def presets(models: pd.DataFrame, ref_id: str) -> list[dict]:
    ref = models.set_index("model_id").loc[ref_id]
    same_feats = np.logical_and.reduce([models[f] == ref[f] for f in FEATS])
    out = [
        {"key": "methods_x_batchkey", "name": "3 methods × 2 batch keys (reference features, seed 0)",
         "description": f"pca / harmony / combat_pca × study / sample, with the reference model's features "
                        f"(n_hvg={ref.n_hvg}, batch-aware={ref.hvg_batch_aware}, exclude IGX={ref.exclude_igx}), seed 0.",
         "ids": models[same_feats & (models.seed == 0)].model_id.tolist()},
        {"key": "reference_seeds", "name": "3 methods × 3 seeds (reference setup)",
         "description": f"Reference features and batch_key={ref.batch_key}; every seed, so rerun noise sits next to method differences.",
         "ids": models[same_feats & (models.batch_key == ref.batch_key)].model_id.tolist()},
        {"key": "harmony_feature_choices", "name": "Harmony, all feature choices (batch_key=sample, seed 0)",
         "description": "n_hvg × batch-aware HVG × exclude IGX for one method: sensitivity to feature selection.",
         "ids": models[(models.method == "harmony") & (models.batch_key == "sample") & (models.seed == 0)].model_id.tolist()},
        {"key": "all_seed0", "name": "All 48 configurations (seed 0)",
         "description": "The whole decision space, one seed each. Agreement patterns become long; use the meta-map and magnitude/severity.",
         "ids": models[models.seed == 0].model_id.tolist()},
    ]
    return [q for q in out if q["ids"]]


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    ec = cfg.get("ec", {})
    n_cells, k_max = ec.get("n_cells", 1500), ec.get("k_max", 30)
    exd = p(cfg, "ec_export_dir")
    if exd.exists():
        shutil.rmtree(exd)
    exd.mkdir(parents=True)
    od = out_dir(cfg)
    models = pd.read_csv(od / "models.csv")
    adata = ad.read_h5ad(p(cfg, "data"))
    obs = adata.obs
    gs = cfg["global_seed"]

    strat = pd.Categorical(obs["cell_type"].astype(str) + "|" + obs["planted_effect"].astype(str)).codes
    sub = stratified_sample(np.asarray(strat), n_cells, gs)
    n = len(sub)
    files: dict[str, dict] = {}

    def write_bin(name, arr, dtype, desc):
        arr = np.ascontiguousarray(arr, dtype=dtype)
        (exd / name).write_bytes(arr.astype(arr.dtype.newbyteorder("<")).tobytes())
        files[name] = {"dtype": np.dtype(dtype).name, "shape": list(arr.shape), "bytes": arr.nbytes, "description": desc}

    # kNN within the subsample, in each model's latent space (exact; ties broken by index)
    lat_dir = out_dir(cfg, "latent")
    knn = np.empty((len(models), n, k_max), np.uint16)
    for i, mid in enumerate(models.model_id):
        Z = np.load(lat_dir / f"{mid}.npy")[sub]
        idx = NearestNeighbors(n_neighbors=k_max + 1).fit(Z).kneighbors(Z, return_distance=False)
        rows = []
        for r in range(n):  # drop self wherever it appears
            row = idx[r][idx[r] != r][:k_max]
            rows.append(row)
        knn[i] = np.asarray(rows)
    write_bin("knn.bin", knn, np.uint16, "kNN indices within the subsample, latent space, ascending distance")

    U = np.load(od / "umap_per_model.npy", mmap_mode="r")
    layouts = np.stack([unit_square(np.asarray(U[i])[sub]) for i in range(len(models))])
    write_bin("layouts.bin", layouts, np.float32, "each model's own UMAP (scanpy recipe) of the subsample, unit square")

    truth = adata.obsm["X_truth"][sub]
    write_bin("neutral_truth.bin", truth, np.float32, "ground-truth biological profile, 20 PCs (simulation only)")
    ln = load_lognorm(cfg)
    from sklearn.decomposition import PCA
    Xl = ln.X.toarray() if hasattr(ln.X, "toarray") else np.asarray(ln.X)
    expr = PCA(n_components=20, random_state=gs).fit_transform((Xl - Xl.mean(0)) / (Xl.std(0) + 1e-6))[sub]
    write_bin("neutral_expression.bin", expr.astype(np.float32), np.float32,
              "PCA (20) of observed log-normalised expression, all genes, z-scored, no batch correction")

    # severity smoothing for spaces with exact ties: 5% of the median pairwise distance
    def floor_of(Xn: np.ndarray) -> float | None:
        rng = np.random.default_rng(gs)
        a, b = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
        d = np.linalg.norm(Xn[a] - Xn[b], axis=1)
        ties = float((np.linalg.norm(Xn[:, None, :] - Xn[None, :200, :], axis=2) == 0).sum() - 200) / (200 * n)
        return round(0.05 * float(np.median(d)), 4) if ties > 0.001 else None
    floors = {"truth": floor_of(truth), "expression": floor_of(expr.astype(np.float32))}
    log.info(f"severity floors (None = original formula): {floors}")

    met = pd.read_csv(od / "metrics.csv", index_col=0)
    ref_id = json.loads((od / "measures" / "meta.json").read_text())["references"][0]
    cats = {}
    for c in ["cell_type", "sample", "study", "tissue", "planted_effect"]:
        col = pd.Categorical(obs[c].astype(str).iloc[sub])
        cats[c] = {"names": [str(x) for x in col.categories], "codes": col.codes.astype(int).tolist()}
    embeddings = [{
        "name": short_name(r), "model_id": r.model_id, "config_id": r.config_id, "seed": int(r.seed),
        "factors": {"method": r.method, "batch_key": r.batch_key, "n_hvg": int(r.n_hvg),
                    "hvg_batch_aware": bool(r.hvg_batch_aware), "exclude_igx": bool(r.exclude_igx)},
        "metrics": {m: round(float(met.loc[r.model_id, m]), 4) for m in ["overall", "bio", "batch", "ARI", "NMI", "graph_iLISI", "kBET_like"]},
    } for r in models.itertuples()]
    meta = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "multiverse-pipeline toy dataset (synthetic, planted ground truth)",
        "n": n, "n_total_cells": int(adata.n_obs), "k_max": k_max,
        "cell_ids": obs.index[sub].tolist(),
        "label_key": "cell_type",
        "categories": cats,
        "embeddings": embeddings,
        "reference_model": ref_id,
        "presets": presets(models, ref_id),
        "default_preset": "methods_x_batchkey",
        "neutral": [
            {"key": "truth", "name": "True biology (simulation)", "file": "neutral_truth.bin", "dim": int(truth.shape[1]),
             "severity_floor": floors["truth"],
             "description": "Noise-free, batch-free cell profile (cell type + IFN state; no batch, IGX or QC effects). Only exists in simulation."},
            {"key": "expression", "name": "Observed expression (uncorrected)", "file": "neutral_expression.bin", "dim": int(expr.shape[1]),
             "severity_floor": floors["expression"],
             "description": "PCA of observed expression — available on real data, but it contains batch effects, so correcting batches looks 'severe'."},
        ],
        "files": files,
    }
    (exd / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    total = sum(f.stat().st_size for f in exd.iterdir())
    log.info(f"exported {len(models)} embeddings × {n} cells to {exd} ({total / 1e6:.1f} MB)")
    append_runtime(cfg, "export_ec", time.perf_counter() - t0)
    return meta


if __name__ == "__main__":
    main()
