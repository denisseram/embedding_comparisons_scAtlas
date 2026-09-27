"""Paper-style benchmark metrics (after scAtlasTb, Mueller et al., 'Integration benchmark metrics').

All metrics are computed on each embedding's own kNN graph or latent space (never on 2D).
Every metric is oriented so that higher = better, min-max scaled across models, then
  bio = mean(scaled bio metrics), batch = mean(scaled batch metrics),
  overall = w_batch * batch + w_bio * bio          (weights in config.yaml)

Simplifications vs the reference implementations (logged in README):
  * kBET-like: per-cell chi-square test of the eval-batch composition of the k=50 neighbourhood
    against the composition expected for that cell's label; score = acceptance rate.
  * graph iLISI: inverse Simpson index of the eval batch in each cell's k=50 kNN neighbourhood,
    rescaled to [0,1] by (iLISI-1)/(n_batches-1), averaged over cells.
  * PCR comparison: (pcr_before - pcr_after)/pcr_before clipped at 0, where "before" is the
    uncorrected `pca` model of the same features and seed.
  * Leiden: on the symmetrised, unweighted k=15 graph; ARI/NMI are the best over resolutions.
"""
from __future__ import annotations

import json
import time
import warnings

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from pipeline.common import append_runtime, get_logger, load_config, log_decision, out_dir, p
from pipeline.integrate import config_id
from pipeline.knn import load_knn
from pipeline.preprocess import load_lognorm

log = get_logger("metrics")
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------- graph helpers
def knn_graph(nbrs: np.ndarray) -> ig.Graph:
    n, k = nbrs.shape
    edges = np.column_stack([np.repeat(np.arange(n), k), nbrs.ravel()])
    g = ig.Graph(n=n, edges=edges, directed=False)
    return g.simplify()


def leiden(g: ig.Graph, resolution: float, seed: int) -> np.ndarray:
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution, seed=seed)
    return np.asarray(part.membership, dtype=np.int32)


def morans_i(x: np.ndarray, nbrs: np.ndarray) -> float:
    """Moran's I with binary directed kNN weights."""
    z = x - x.mean()
    denom = (z ** 2).sum()
    if denom == 0:
        return 0.0
    n, k = nbrs.shape
    return float((n / (n * k)) * (z[:, None] * z[nbrs]).sum() / denom)


# ---------------------------------------------------------------------------- batch metrics
def kbet_like(nbrs: np.ndarray, batch: np.ndarray, label: np.ndarray, alpha: float) -> float:
    B = batch.max() + 1
    k = nbrs.shape[1]
    counts = np.zeros((nbrs.shape[0], B))
    for b in range(B):
        counts[:, b] = (batch[nbrs] == b).sum(1)
    accept = np.zeros(nbrs.shape[0], bool)
    for lab in np.unique(label):
        m = label == lab
        freq = np.bincount(batch[m], minlength=B) / m.sum()
        present = freq > 0
        if present.sum() < 2:          # label in one batch only: nothing to mix, accept
            accept[m] = True
            continue
        exp = k * freq[present]
        chi = ((counts[m][:, present] - exp) ** 2 / exp).sum(1)
        pval = stats.chi2.sf(chi, df=present.sum() - 1)
        accept[m] = pval > alpha
    return float(accept.mean())


def graph_ilisi(nbrs: np.ndarray, batch: np.ndarray) -> float:
    B = batch.max() + 1
    k = nbrs.shape[1]
    simpson = np.zeros(nbrs.shape[0])
    for b in range(B):
        simpson += ((batch[nbrs] == b).sum(1) / k) ** 2
    ilisi = 1.0 / simpson
    return float(np.mean((ilisi - 1) / (B - 1)))


def pcr(X: np.ndarray, batch: np.ndarray) -> float:
    """Variance-weighted R^2 of the principal components of X regressed on batch one-hot."""
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pcs = U * S
    var = S ** 2
    onehot = np.eye(batch.max() + 1)[batch]
    beta, *_ = np.linalg.lstsq(onehot, pcs, rcond=None)
    resid = pcs - onehot @ beta
    r2 = 1 - (resid ** 2).sum(0) / ((pcs - pcs.mean(0)) ** 2).sum(0)
    return float((r2 * var).sum() / var.sum())


# ---------------------------------------------------------------------------- paper-style local tools
def graph_dissimilarity(lat_a: np.ndarray, lat_b: np.ndarray, nbrs_a: np.ndarray,
                        dist_a: np.ndarray, dist_b: np.ndarray, q: float = 0.9) -> np.ndarray:
    """Asymmetric per-cell change of graph-a neighbourhoods measured in embedding b.

    d(c) = mean_j ||b_c - b_j|| / q_b  -  mean_j ||a_c - a_j|| / q_a,   j in N_a(c),
    where q_x is the q-quantile of all kNN distances of graph x.
    """
    qa, qb = np.quantile(dist_a, q), np.quantile(dist_b, q)
    in_a = dist_a.mean(1) / qa
    diff = lat_b[:, None, :] - lat_b[nbrs_a]
    in_b = np.sqrt((diff ** 2).sum(-1)).mean(1) / qb
    return (in_b - in_a).astype(np.float32)


def majority_vote_disagreement(clusters: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """clusters: [M, N] Leiden memberships. Each cluster takes its majority label; per cell,
    disagreement = 1 - fraction of models agreeing with the most common assigned label."""
    M, N = clusters.shape
    L = labels.max() + 1
    assigned = np.empty((M, N), np.int32)
    for m in range(M):
        c = clusters[m]
        tab = np.zeros((c.max() + 1, L), np.int64)
        np.add.at(tab, (c, labels), 1)
        assigned[m] = tab.argmax(1)[c]
    votes = np.zeros((N, L), np.int32)
    for m in range(M):
        votes[np.arange(N), assigned[m]] += 1
    return (1 - votes.max(1) / M).astype(np.float32)


_CTX: dict = {}


def _init_worker(ctx: dict) -> None:
    _CTX.clear()
    _CTX.update(ctx)
    _CTX["knn"] = load_knn(ctx["cfg"])
    _CTX["pcr_before"] = {}


def _score_model(m: int) -> tuple[dict, np.ndarray]:
    """Score one model (runs in a worker process). PCR 'before' = uncorrected pca model
    with identical features and seed."""
    c = _CTX
    cfg, r = c["cfg"], c["models"].iloc[m]
    mc, gs, k_small = cfg["metrics"], cfg["global_seed"], cfg["knn"]["k_small"]
    labels, batch = c["labels"], c["batch"]
    lat_dir = out_dir(cfg, "latent")
    nb50 = np.asarray(c["knn"][m])
    nb15 = nb50[:, :k_small]
    g = knn_graph(nb15)
    best, best_cl = {"ARI": -1.0, "NMI": -1.0}, None
    for res in mc["leiden_resolutions"]:
        cl = leiden(g, res, gs)
        ari = adjusted_rand_score(labels, cl)
        if ari > best["ARI"]:
            best_cl = cl
        best["ARI"] = max(best["ARI"], ari)
        best["NMI"] = max(best["NMI"], normalized_mutual_info_score(labels, cl))
    row = {"model_id": r.model_id, **best}
    for j, name in enumerate(c["score_names"]):
        row[f"moransI_{name[3:]}"] = morans_i(c["scores"][:, j], nb15)
    for j, name in enumerate(mc["covariates"]):
        row[f"moransI_{name}"] = morans_i(c["cov"][:, j], nb15)
    row["kBET_like"] = kbet_like(nb50, batch, labels, mc["kbet_alpha"])
    row["graph_iLISI"] = graph_ilisi(nb50, batch)
    lat = np.load(lat_dir / f"{r.model_id}.npy")
    ref = config_id(dict(method="pca", n_hvg=r.n_hvg, hvg_batch_aware=r.hvg_batch_aware,
                         exclude_igx=r.exclude_igx, batch_key=r.batch_key)) + f".s{r.seed}"
    if ref not in c["pcr_before"]:
        c["pcr_before"][ref] = pcr(np.load(lat_dir / f"{ref}.npy"), batch)
    before = c["pcr_before"][ref]
    after = pcr(lat, batch)
    row["PCR_comparison"] = float(max(0.0, (before - after) / before)) if before > 0 else 0.0
    if c["use_bras"]:  # pragma: no cover
        from scib_metrics import bras
        row["BRAS"] = float(bras(lat, labels, batch))
    return row, best_cl


# ---------------------------------------------------------------------------- main
def gene_set_scores(cfg: dict) -> pd.DataFrame:
    adata = load_lognorm(cfg)
    sets = json.loads(p(cfg, "gene_sets").read_text())
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    out = {}
    names = adata.var_names
    for name, genes in sets.items():
        idx = names.get_indexer(genes)
        out[f"gs_{name}"] = ((X[:, idx] - mu[idx]) / sd[idx]).mean(1)
    return pd.DataFrame(out, index=adata.obs_names)


def metric_info(cfg: dict, gs_cols: list[str]) -> list[dict]:
    info = [{"name": "ARI", "group": "bio"}, {"name": "NMI", "group": "bio"}]
    info += [{"name": f"moransI_{c[3:]}", "group": "bio"} for c in gs_cols]
    info += [{"name": f"moransI_{c}", "group": "bio"} for c in cfg["metrics"]["covariates"]]
    info += [{"name": "kBET_like", "group": "batch"}, {"name": "graph_iLISI", "group": "batch"},
             {"name": "PCR_comparison", "group": "batch"}]
    for d in info:
        d["direction"] = "higher_is_better"
    return info


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    mc, gs = cfg["metrics"], cfg["global_seed"]
    import anndata as ad
    obs = ad.read_h5ad(p(cfg, "data"), backed="r").obs.copy()
    models = pd.read_csv(out_dir(cfg) / "models.csv")
    scores = gene_set_scores(cfg)
    cov = obs[mc["covariates"]].to_numpy(np.float64)
    labels = pd.Categorical(obs[mc["label_key"]]).codes.astype(np.int64)
    batch = pd.Categorical(obs[mc["eval_batch_key"]]).codes.astype(np.int64)
    info = metric_info(cfg, list(scores.columns))

    use_bras = False
    if mc.get("use_scib_metrics"):
        try:  # pragma: no cover - optional
            from scib_metrics import bras  # noqa: F401
            use_bras = True
            info.append({"name": "BRAS", "group": "batch", "direction": "higher_is_better"})
        except Exception as e:
            log_decision(cfg, f"scib-metrics requested but not importable ({e!r}); BRAS skipped.")

    ctx = dict(cfg=cfg, models=models, scores=scores.to_numpy(np.float64), score_names=list(scores.columns),
               cov=cov, labels=labels, batch=batch, use_bras=use_bras)
    n_jobs = int(cfg.get("n_jobs", 1))
    rows, clusters = [], np.zeros((len(models), len(obs)), np.int32)
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=(ctx,)) as ex:
            results = ex.map(_score_model, range(len(models)), chunksize=4)
            for m, (row, cl) in enumerate(results):
                rows.append(row)
                clusters[m] = cl
                if (m + 1) % 24 == 0:
                    log.info(f"{m + 1}/{len(models)} models scored")
    else:
        _init_worker(ctx)
        for m in range(len(models)):
            row, clusters[m] = _score_model(m)
            rows.append(row)

    raw = pd.DataFrame(rows).set_index("model_id")
    names = [d["name"] for d in info]
    scaled = (raw[names] - raw[names].min()) / (raw[names].max() - raw[names].min()).replace(0, 1)
    bio_cols = [d["name"] for d in info if d["group"] == "bio"]
    batch_cols = [d["name"] for d in info if d["group"] == "batch"]
    w = mc["weights"]
    out = raw.copy()
    for c in names:
        out[f"{c}_scaled"] = scaled[c]
    out["bio"] = scaled[bio_cols].mean(axis=1)
    out["batch"] = scaled[batch_cols].mean(axis=1)
    out["overall"] = w["batch"] * out["batch"] + w["bio"] * out["bio"]
    out.to_csv(out_dir(cfg) / "metrics.csv")

    # seed-replicate sd per metric (raw, scaled and aggregates), attached to each model
    cid = models.set_index("model_id").loc[out.index, "config_id"]
    sd = out.groupby(cid.values).std(ddof=1)
    sd_per_model = sd.loc[cid.values].set_index(out.index)
    sd_per_model.to_csv(out_dir(cfg) / "metrics_seed_sd.csv")

    np.save(out_dir(cfg) / "leiden_best.npy", clusters)
    mv = majority_vote_disagreement(clusters, labels)
    np.save(out_dir(cfg) / "cell_majority_disagreement.npy", mv)
    (out_dir(cfg) / "metric_info.json").write_text(json.dumps(info, indent=1))
    append_runtime(cfg, "metrics", time.perf_counter() - t0)
    log.info(f"top-5 overall:\n{out['overall'].sort_values(ascending=False).head(5).round(3).to_string()}")
    return out


if __name__ == "__main__":
    main()
