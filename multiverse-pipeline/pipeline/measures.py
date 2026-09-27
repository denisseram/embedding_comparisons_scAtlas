"""Multiverse measures (the project's own contribution). Everything is computed on the
embeddings' own kNN graphs (idea I1); no 2D coordinate is used.

Definitions (k = measures.k, default 15):
  Δ(c,a,b)   = 1 - Jaccard(N_a(c), N_b(c))
  seed noise : per configuration C, Δ between its seed replicates (3 pairs for 3 seeds).
               For a model pair (a,b):  μ = (m_A + m_B)/2,  σ² = (v_A + v_B)/2
               (m, v = per-cell mean and ddof=1 variance of the config's seed Δs), then both are
               smoothed over the consensus kNN graph (mean over the cell and its k neighbours).
  z(c,a,b)   = (Δ - μ̃) / max(σ̃, ε)
  rank-z     = Φ⁻¹(percentile of Δ among the pooled seed Δs of A and B over the cell's
               consensus neighbourhood)
  Δ̄(c,A,B)   = mean over seed pairings i≠j of Δ(c, A_i, B_j). Seed index i uses the same PCA
               subsample in every configuration, so equal-index pairings would share part of
               their noise; only independent reruns (i≠j) are averaged.
  P_F        = configuration pairs differing only in factor F
  E(c,F)     = mean_{P_F} z,   H(c,F) = sd_{P_F} z
  seed pseudo-factor (null): for each configuration and each seed pair, the pair's Δ is scored
               against a leave-one-out baseline built from the configuration's other seed pairs.
  A(a,b)     = mean over a stratified cell sample of Δ(c,a,b)
  stability  = per cell, fraction of its k consensus slots filled by neighbours present in
               ≥ threshold of the models' kNN sets (all models, and seed replicates only).
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from pipeline.common import append_runtime, get_logger, load_config, out_dir, p
from pipeline.integrate import FACTORS
from pipeline.knn import load_knn

log = get_logger("measures")


# ----------------------------------------------------------------------------- primitives
def delta(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-row 1 - Jaccard of two neighbour-index arrays of shape [n, k] (no duplicates per row)."""
    k = A.shape[1]
    s = np.sort(np.concatenate([A, B], axis=1), axis=1)
    inter = (s[:, 1:] == s[:, :-1]).sum(1)
    return (1.0 - inter / (2 * k - inter)).astype(np.float32)


def delta_composition(Ca: np.ndarray, Cb: np.ndarray) -> np.ndarray:
    """Exploratory variant: 1 - overlap of neighbour micro-cluster composition.
    Ca, Cb: [n, n_blocks] neighbour counts per block (rows sum to k)."""
    k = Ca[0].sum()
    return (1.0 - np.minimum(Ca, Cb).sum(1) / k).astype(np.float32)


def micro_clusters(cons: np.ndarray, target: int, seed: int) -> tuple[np.ndarray, float]:
    """Label-free partition of the consensus graph with the number of blocks closest to target."""
    from pipeline.metrics import knn_graph, leiden
    g = knn_graph(cons)
    best = None
    for res in [0.5, 1, 2, 3, 5, 8, 12, 20]:
        lab = leiden(g, res, seed)
        n = lab.max() + 1
        if best is None or abs(n - target) < abs(best[0].max() + 1 - target):
            best = (lab, res)
    return best


def composition_rep(nbrs: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    M, N, k = nbrs.shape
    nb = blocks.max() + 1
    rep = np.zeros((M, N, nb), np.uint8)
    rows = np.repeat(np.arange(N), k)
    for m in range(M):
        np.add.at(rep[m], (rows, blocks[nbrs[m]].ravel()), 1)
    return rep


def neighbour_counts(nbr_sets: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """nbr_sets: [n, L] pooled neighbour ids per cell. Returns flat (row, id, count) of unique ids."""
    n, L = nbr_sets.shape
    s = np.sort(nbr_sets, axis=1).astype(np.int64) + np.arange(n)[:, None] * n_cells
    flat = s.ravel()
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    counts = np.diff(np.r_[starts, flat.size])
    u = flat[starts]
    return u // n_cells, u % n_cells, counts


def consensus(nbrs: np.ndarray, k: int, threshold: float, chunk: int = 1000):
    """nbrs: [M, N, k]. Returns consensus kNN [N, k] (most frequent neighbours) and stability [N]."""
    M, N, _ = nbrs.shape
    cons = np.empty((N, k), np.int32)
    stab = np.empty(N, np.float32)
    need = int(np.ceil(threshold * M - 1e-9))
    for s0 in range(0, N, chunk):
        s1 = min(N, s0 + chunk)
        pooled = np.transpose(nbrs[:, s0:s1, :], (1, 0, 2)).reshape(s1 - s0, M * k)
        row, ids, cnt = neighbour_counts(pooled, N)
        stab[s0:s1] = np.bincount(row, weights=(cnt >= need), minlength=s1 - s0) / k
        order = np.lexsort((ids, -cnt, row))
        row, ids = row[order], ids[order]
        first = np.searchsorted(row, np.arange(s1 - s0))
        take = first[:, None] + np.arange(k)[None, :]
        cons[s0:s1] = ids[take]
    return cons, stab


def consensus_share(nbrs: np.ndarray, threshold: float, chunk: int = 1000) -> np.ndarray:
    """Per model: mean fraction of its k neighbours that are consensus neighbours
    (present in >= threshold of all models' kNN sets). Model-level summary of stability."""
    M, N, k = nbrs.shape
    need = int(np.ceil(threshold * M - 1e-9))
    share = np.zeros(M)
    for s0 in range(0, N, chunk):
        s1 = min(N, s0 + chunk)
        pooled = np.transpose(nbrs[:, s0:s1, :], (1, 0, 2)).reshape(s1 - s0, M * k)
        row, ids, cnt = neighbour_counts(pooled, N)
        keys = np.sort(row[cnt >= need] * N + ids[cnt >= need])
        base = np.arange(s1 - s0, dtype=np.int64)[:, None] * N
        for m in range(M):
            share[m] += np.isin(base + nbrs[m, s0:s1], keys, assume_unique=False).sum()
    return (share / (N * k)).astype(np.float32)


def smooth(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Mean of x over each cell and its reference-graph neighbours. x: [..., N]."""
    return ((x + x[..., ref].sum(-1)) / (ref.shape[1] + 1)).astype(np.float32)


def rank_z(value: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """value [N], pool [N, P]: mid-rank percentile of value within pool, mapped through Φ⁻¹."""
    P = pool.shape[1]
    r = ((pool < value[:, None]).sum(1) + 0.5 * (pool == value[:, None]).sum(1)) / P
    r = np.clip(r, 1 / (P + 1), P / (P + 1))
    return norm.ppf(r).astype(np.float32)


def matched_pairs(configs: pd.DataFrame, factors: list[str] | None = None) -> pd.DataFrame:
    """All configuration pairs that differ in exactly one factor."""
    factors = factors or FACTORS
    rows = []
    cfg_rows = configs.to_dict("records")
    for F in factors:
        others = [f for f in factors if f != F]
        groups: dict[tuple, list] = {}
        for c in cfg_rows:
            groups.setdefault(tuple(c[f] for f in others), []).append(c)
        for grp in groups.values():
            grp = sorted(grp, key=lambda c: str(c[F]))
            for a, b in itertools.combinations(grp, 2):
                rows.append({"factor": F, "config_a": a["config_id"], "config_b": b["config_id"],
                             "level_a": str(a[F]), "level_b": str(b[F])})
    out = pd.DataFrame(rows)
    out.insert(0, "pair_id", np.arange(len(out)))
    return out


def stratified_sample(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = labels.size
    if n >= N:
        return np.arange(N)
    idx = []
    for lab in np.unique(labels):
        m = np.flatnonzero(labels == lab)
        take = max(1, int(round(n * m.size / N)))
        idx.append(rng.choice(m, min(take, m.size), replace=False))
    return np.sort(np.concatenate(idx))


# ----------------------------------------------------------------------------- main
def main(cfg: dict | None = None, mode: str | None = None) -> dict:
    """mode: 'jaccard' (pre-registered primary measure) or 'composition' (exploratory)."""
    cfg = cfg or load_config()
    modes = cfg["measures"].get("modes", ["jaccard"])
    mode = mode or modes[0]
    primary = mode == modes[0]
    sfx = "" if primary else f"_{mode}"
    t0 = time.perf_counter()
    mc, gs = cfg["measures"], cfg["global_seed"]
    k, eps = mc["k"], mc["sigma_eps"]
    od = out_dir(cfg, "measures")
    models = pd.read_csv(out_dir(cfg) / "models.csv")
    M = len(models)
    nbrs = np.ascontiguousarray(load_knn(cfg)[:, :, :k])
    N = nbrs.shape[1]
    mid_to_idx = {m: i for i, m in enumerate(models.model_id)}
    configs = models.drop_duplicates("config_id")[["config_id", *FACTORS]].reset_index(drop=True)
    seeds = sorted(models.seed.unique())
    by_cfg = {c: [mid_to_idx[f"{c}.s{s}"] for s in seeds] for c in configs.config_id}
    S = len(seeds)
    seed_pairs = list(itertools.combinations(range(S), 2))

    # --- consensus graph and stability ----------------------------------------------------
    cons, stab_all = consensus(nbrs, k, mc["consensus_threshold"])
    stab_seed = np.zeros(N, np.float32)
    for c, idx in by_cfg.items():
        stab_seed += consensus(nbrs[idx], k, mc["consensus_threshold"])[1]
    stab_seed /= len(by_cfg)
    log.info(f"stability: all-models median {np.median(stab_all):.2f}, seed-only median {np.median(stab_seed):.2f}")

    if mode == "jaccard":
        rep, dfun = nbrs, delta
    elif mode == "composition":
        blocks, bres = micro_clusters(cons, max(2, round(N / (5 * k))), gs)
        log.info(f"composition mode: {blocks.max() + 1} consensus micro-clusters (resolution {bres})")
        np.save(od / "micro_clusters.npy", blocks)
        rep, dfun = composition_rep(nbrs, blocks), delta_composition
    else:
        raise ValueError(f"unknown measures mode {mode!r}")

    # --- seed noise per configuration ----------------------------------------------------
    seed_delta = np.zeros((len(configs), len(seed_pairs), N), np.float32)
    for ci, c in enumerate(configs.config_id):
        for pi, (i, j) in enumerate(seed_pairs):
            seed_delta[ci, pi] = dfun(rep[by_cfg[c][i]], rep[by_cfg[c][j]])
    cfg_index = {c: i for i, c in enumerate(configs.config_id)}
    m_cfg = seed_delta.mean(1)
    v_cfg = seed_delta.var(1, ddof=1)

    def noise(ca: str, cb: str) -> tuple[np.ndarray, np.ndarray]:
        ia, ib = cfg_index[ca], cfg_index[cb]
        mu = smooth((m_cfg[ia] + m_cfg[ib]) / 2, cons)
        sd = np.sqrt(smooth((v_cfg[ia] + v_cfg[ib]) / 2, cons))
        return mu, np.maximum(sd, eps)

    def pool(ca: str, cb: str) -> np.ndarray:
        """Pooled seed Δs of both configurations over each cell's consensus neighbourhood."""
        sd_ = np.concatenate([seed_delta[cfg_index[ca]], seed_delta[cfg_index[cb]]], 0)  # [2S', N]
        hood = np.concatenate([np.arange(N)[:, None], cons], 1)                        # [N, k+1]
        return sd_[:, hood].transpose(1, 0, 2).reshape(N, -1)

    # --- matched one-factor contrasts ----------------------------------------------------
    pairs = matched_pairs(configs)
    P = len(pairs)
    pair_delta = np.zeros((N, P), np.float32)
    pair_z = np.zeros((P, N), np.float32)
    pair_rz = np.zeros((P, N), np.float32)
    off_diag = [(i, j) for i in range(S) for j in range(S) if i != j]
    for r in pairs.itertuples():
        A, B = by_cfg[r.config_a], by_cfg[r.config_b]
        d = np.mean([dfun(rep[A[i]], rep[B[j]]) for i, j in off_diag], axis=0)
        mu, sd = noise(r.config_a, r.config_b)
        pair_delta[:, r.pair_id] = d
        pair_z[r.pair_id] = (d - mu) / sd
        pair_rz[r.pair_id] = rank_z(d, pool(r.config_a, r.config_b))

    active = [F for F in FACTORS if (pairs.factor == F).any()]  # factors with >1 level in the design
    factors = active + ["seed"]
    E = np.zeros((N, len(factors)), np.float32)
    H = np.zeros_like(E)
    E_rank = np.zeros_like(E)
    for fi, F in enumerate(active):
        sel = pairs.factor.values == F
        E[:, fi] = pair_z[sel].mean(0)
        H[:, fi] = pair_z[sel].std(0, ddof=1)
        E_rank[:, fi] = pair_rz[sel].mean(0)

    # seed pseudo-factor: leave-one-out null
    hood = np.concatenate([np.arange(N)[:, None], cons], 1)
    z_seed, rz_seed = [], []
    for ci in range(len(configs)):
        for pi in range(len(seed_pairs)):
            rest = np.delete(seed_delta[ci], pi, axis=0)
            mu = smooth(rest.mean(0), cons)
            sd = np.maximum(np.sqrt(smooth(rest.var(0, ddof=1), cons)), eps)
            z_seed.append((seed_delta[ci, pi] - mu) / sd)
            rz_seed.append(rank_z(seed_delta[ci, pi], rest[:, hood].transpose(1, 0, 2).reshape(N, -1)))
    z_seed = np.asarray(z_seed)
    E[:, -1] = z_seed.mean(0)
    H[:, -1] = z_seed.std(0, ddof=1)
    E_rank[:, -1] = np.asarray(rz_seed).mean(0)
    log.info("median |E| per factor: " + ", ".join(f"{F}={np.median(np.abs(E[:, i])):.2f}"
                                                    for i, F in enumerate(factors)))

    # --- model agreement --------------------------------------------------------------------
    import anndata as ad
    obs = ad.read_h5ad(p(cfg, "data"), backed="r").obs
    labels = pd.Categorical(obs[cfg["metrics"]["label_key"]]).codes
    sample_cells = stratified_sample(labels, mc["agreement_n_cells"], gs)
    sub = rep[:, sample_cells, :]
    A_mat = np.zeros((M, M), np.float32)
    for a in range(M):
        for b in range(a + 1, M):
            A_mat[a, b] = A_mat[b, a] = dfun(sub[a], sub[b]).mean()

    # --- calibrated change versus reference models -------------------------------------------
    met = pd.read_csv(out_dir(cfg) / "metrics.csv", index_col=0).loc[models.model_id]
    refs = []
    for col in ["overall", "bio", "batch"]:
        best = met[col].idxmax()
        if best not in refs:
            refs.append(best)
    refs = refs[: cfg["export"]["n_reference_models"]]
    z_ref = np.zeros((len(refs), N, M), np.float32)
    for ri, ref in enumerate(refs):
        a = mid_to_idx[ref]
        ca = models.config_id[a]
        for b in range(M):
            mu, sd = noise(ca, models.config_id[b])
            z_ref[ri, :, b] = (dfun(rep[a], rep[b]) - mu) / sd

    # --- paper-style asymmetric graph dissimilarity over matched pairs (baseline) ----------
    from pipeline.metrics import graph_dissimilarity
    gd = None
    dist = np.load(out_dir(cfg) / "knn50_dist.npy", mmap_mode="r")
    lat_dir = out_dir(cfg, "latent")
    q = cfg["metrics"]["graph_dissim_quantile"]
    for r in (pairs.itertuples() if primary else []):
        if gd is None:
            gd = np.zeros(N, np.float64)
        a, b = by_cfg[r.config_a][0], by_cfg[r.config_b][1 % S]
        la = np.load(lat_dir / f"{models.model_id[a]}.npy")
        lb = np.load(lat_dir / f"{models.model_id[b]}.npy")
        da, db = np.asarray(dist[a, :, :k]), np.asarray(dist[b, :, :k])
        gd += 0.5 * (np.abs(graph_dissimilarity(la, lb, nbrs[a], da, db, q))
                     + np.abs(graph_dissimilarity(lb, la, nbrs[b], db, da, q)))
    if gd is not None:
        gd /= P

    # --- save ---------------------------------------------------------------------------------
    if primary:
        np.save(od / "consensus_knn.npy", cons)
        np.save(od / "stability_all.npy", stab_all)
        np.save(od / "stability_seed.npy", stab_seed)
        np.save(od / "model_consensus_share.npy", consensus_share(nbrs, mc["consensus_threshold"]))
        pairs.to_csv(od / "pairs.csv", index=False)
        np.save(od / "agreement_cells.npy", sample_cells)
        np.save(od / "paper_graph_dissim.npy", gd.astype(np.float32))
        (od / "meta.json").write_text(json.dumps({"factors": factors, "k": k, "references": refs, "modes": modes,
                                                  "configs": configs.config_id.tolist()}, indent=1))
    for name, arr in [("seed_delta", seed_delta), ("pair_delta", pair_delta), ("pair_z", pair_z), ("E", E),
                      ("H", H), ("E_rank", E_rank), ("agreement", A_mat), ("z_vs_ref", z_ref)]:
        np.save(od / f"{name}{sfx}.npy", arr)
    append_runtime(cfg, f"measures_{mode}", time.perf_counter() - t0)
    log.info(f"measures done in {time.perf_counter() - t0:.1f}s; references: {refs}")
    return {"E": E, "H": H, "factors": factors}


if __name__ == "__main__":
    import sys
    main(mode=sys.argv[1] if len(sys.argv) > 1 else None)
