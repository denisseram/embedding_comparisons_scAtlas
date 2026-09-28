"""Comparative fuzzy UpSet: which labels mix in each embedding's kNN graph, and how that differs across models.

Per model m and one label column (never mixing columns):
  A  = binarised kNN adjacency with self-loops, row-normalised (directed: every row has k+1 entries)
  L  = one-hot labels [n_cells, n_labels]; unlabelled cells (code -1) get no label but still count as neighbours
  P  = A @ L, P[i, l] = fraction of cell i's neighbourhood (incl. itself) with label l
  score = P, or P / global frequency of l when normalize=True (enrichment; decides which labels pass tau only)
  signature(i) = sorted labels with score >= tau; > max_size labels -> "diffuse", 0 labels -> "none"
  strength(i)  = min over the signature of the RAW membership P[i, l] (always in [0, 1])
Per intersection: n_cells, fuzzy_size = sum of strengths, mean_strength.

Everything is sparse or row-chunked; no n_cells x n_cells dense matrix is ever built. The browser re-implements
the threshold/aggregation step (integration/.../compute/fuzzyUpset.ts) on exported membership counts; this module
is the reference implementation and the offline / large-scale path.

CLI:  python -m pipeline.fuzzy_upset --label cell_type [--tau 0.1 --max-size 3 --normalize --models seed0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pipeline.common import get_logger, load_config, out_dir, p

log = get_logger("fuzzy_upset")
NONE, DIFFUSE = -1, -2  # special signature ids


# ---- labels -------------------------------------------------------------------------------------

def label_codes(values) -> tuple[np.ndarray, list[str]]:
    """Dictionary-encode a label column; NaN / None / '' -> -1 (unlabelled)."""
    s = pd.Series(values)
    s = s.where(~s.isna() & (s.astype(str) != ""), None)
    c = pd.Categorical(s.dropna().astype(str))
    codes = np.full(len(s), -1, np.int32)
    codes[s.notna().to_numpy()] = c.codes
    return codes, [str(x) for x in c.categories]


def label_frequency(codes: np.ndarray, n_labels: int) -> np.ndarray:
    """Global frequency of each label among labelled cells (used for the enrichment normalisation)."""
    ok = codes >= 0
    cnt = np.bincount(codes[ok], minlength=n_labels).astype(np.float64)
    return cnt / max(ok.sum(), 1)


def label_warnings(codes: np.ndarray, levels: list[str]) -> list[str]:
    out = []
    n_unl = int((codes < 0).sum())
    if n_unl:
        out.append(f"{n_unl} unlabelled cells: they have no label of their own but still count as neighbours "
                   f"(so neighbourhood fractions of their neighbours sum to < 1).")
    cnt = np.bincount(codes[codes >= 0], minlength=len(levels))
    for l, c in zip(levels, cnt):
        if c == 0:
            out.append(f"label '{l}' has no cells.")
        elif c == 1:
            out.append(f"label '{l}' has a single cell; its membership can never exceed 1/(k+1) in any neighbour.")
    return out


# ---- memberships --------------------------------------------------------------------------------

def membership_counts(knn: np.ndarray, codes: np.ndarray, n_labels: int, chunk: int = 262_144) -> np.ndarray:
    """Neighbour label counts (self included) from a directed kNN index array [n, k] (self excluded).

    Returns uint8 (uint16 if k+1 > 255) [n, n_labels]; P = counts / (k+1). Row-chunked; memory O(chunk * k).
    """
    n, k = knn.shape
    out = np.zeros((n, n_labels), np.uint8 if k + 1 <= 255 else np.uint16)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        nb = np.concatenate([np.arange(s, e)[:, None], np.asarray(knn[s:e], np.int64)], axis=1)
        c = codes[nb].ravel()
        rows = np.repeat(np.arange(e - s), k + 1)
        ok = c >= 0
        out[s:e] = np.bincount(rows[ok] * n_labels + c[ok], minlength=(e - s) * n_labels).reshape(e - s, n_labels)
    return out


def memberships(knn: np.ndarray, codes: np.ndarray, n_labels: int) -> sp.csr_matrix:
    """P = A @ L for a directed kNN index array (fast path; equals memberships_from_adjacency)."""
    k = knn.shape[1]
    return (sp.csr_matrix(membership_counts(knn, codes, n_labels), dtype=np.float32) / np.float32(k + 1)).tocsr()


def memberships_from_adjacency(A: sp.spmatrix, codes: np.ndarray, n_labels: int) -> sp.csr_matrix:
    """P = A @ L for any sparse adjacency: binarised, self-loops set, row-normalised."""
    A = sp.csr_matrix(A, dtype=np.float32, copy=True)
    A.data[:] = 1.0
    A = (A + sp.identity(A.shape[0], np.float32, format="csr")).tocsr()
    A.data[:] = 1.0  # a self-loop already present must not count twice
    deg = np.asarray(A.sum(1)).ravel()
    A = sp.diags(1.0 / deg).astype(np.float32) @ A
    n = len(codes)
    ok = codes >= 0
    L = sp.csr_matrix((np.ones(ok.sum(), np.float32), (np.flatnonzero(ok), codes[ok])), shape=(n, n_labels))
    return (A @ L).tocsr()


# ---- signatures ---------------------------------------------------------------------------------

@dataclass
class Signatures:
    sig: np.ndarray                  # int32 [n]: intersection id, NONE (-1) or DIFFUSE (-2)
    strength: np.ndarray             # float32 [n]: min raw membership over the signature (0 for NONE/DIFFUSE)
    intersections: list[tuple[int, ...]]
    n_none: int
    n_diffuse: int


def signatures(P: sp.csr_matrix, tau: float = 0.1, max_size: int = 3, normalize: bool = False,
               freq: np.ndarray | None = None) -> Signatures:
    """Per-cell signature (labels with score >= tau), fuzzy strength and the diffuse / none buckets. Vectorised."""
    P = sp.csr_matrix(P)
    P.sort_indices()
    n = P.shape[0]
    rows = np.repeat(np.arange(n), np.diff(P.indptr))
    lab, raw = P.indices, P.data
    score = raw
    if normalize:
        if freq is None:
            raise ValueError("normalize=True needs the global label frequencies (label_frequency).")
        f = np.asarray(freq, np.float64)[lab]
        score = np.where(f > 0, raw / np.where(f > 0, f, 1), 0.0)
    keep = (score >= tau) & (raw > 0)
    rows, lab, raw = rows[keep], lab[keep], raw[keep]
    size = np.bincount(rows, minlength=n)
    sig = np.full(n, NONE, np.int32)
    sig[size > max_size] = DIFFUSE
    strength = np.zeros(n, np.float32)
    good = (size >= 1) & (size <= max_size)
    if good.any():
        sel = good[rows]
        r, l, v = rows[sel], lab[sel], raw[sel]
        smin = np.full(n, np.inf, np.float32)
        np.minimum.at(smin, r, v)
        strength[good] = smin[good]
        # padded sorted label tuple per cell; entries are already sorted by label within a row (sort_indices)
        start = np.searchsorted(r, np.arange(n))  # first kept entry of each row
        pos = np.arange(len(r)) - start[r]
        pad = np.full((n, max_size), -1, np.int64)
        pad[r, pos] = l
        uniq, inv = np.unique(pad[good], axis=0, return_inverse=True)
        sig[good] = inv.ravel()
        inters = [tuple(int(x) for x in u if x >= 0) for u in uniq]
    else:
        inters = []
    return Signatures(sig, strength, inters, int((sig == NONE).sum()), int((sig == DIFFUSE).sum()))


def aggregate(s: Signatures) -> pd.DataFrame:
    """One row per intersection: labels, degree, n_cells, fuzzy_size, mean_strength."""
    ok = s.sig >= 0
    K = len(s.intersections)
    n = np.bincount(s.sig[ok], minlength=K)
    fz = np.bincount(s.sig[ok], weights=s.strength[ok], minlength=K)
    return pd.DataFrame({"labels": s.intersections, "degree": [len(t) for t in s.intersections],
                         "n_cells": n, "fuzzy_size": fz, "mean_strength": np.where(n > 0, fz / np.maximum(n, 1), 0.0)})


def cells_for(s: Signatures, labels: tuple[int, ...]) -> np.ndarray:
    """Cell indices whose signature is exactly `labels` (sorted label ids)."""
    t = tuple(sorted(labels))
    if t not in s.intersections:
        return np.zeros(0, np.int64)
    return np.flatnonzero(s.sig == s.intersections.index(t))


# ---- cross-model comparison ---------------------------------------------------------------------

@dataclass
class Comparison:
    levels: list[str]
    models: list[str]                                   # models with a graph, in input order
    fuzzy: pd.DataFrame                                 # intersection (label tuple) x model -> fuzzy_size (0 if absent)
    n_cells: pd.DataFrame                               # same shape, cell counts
    per_model: dict[str, Signatures]
    buckets: pd.DataFrame                               # model x {n_none, n_diffuse}
    warnings: list[str] = field(default_factory=list)


def compare(P_by_model: dict[str, sp.csr_matrix | None], levels: list[str], tau: float = 0.1, max_size: int = 3,
            normalize: bool = False, freq: np.ndarray | None = None) -> Comparison:
    """Signatures for every model with a graph and the intersection x model tables. Models whose P is None
    (no graph) are skipped with a warning."""
    warn, per = [], {}
    for mid, P in P_by_model.items():
        if P is None:
            warn.append(f"model '{mid}' has no kNN graph; it is left out of the comparison.")
            continue
        per[mid] = signatures(P, tau, max_size, normalize, freq)
    return _tables(per, levels, warn)


def _tables(per: dict[str, Signatures], levels: list[str], warn: list[str]) -> Comparison:
    if not per:
        raise ValueError("no model has a kNN graph; nothing to compare. " + " ".join(warn))
    tabs = {m: aggregate(s).set_index("labels") for m, s in per.items()}
    keys = sorted({t for d in tabs.values() for t in d.index}, key=lambda t: (len(t), t))
    idx = pd.Index(keys, tupleize_cols=False, name="labels")
    fz = pd.DataFrame({m: d.fuzzy_size.reindex(idx, fill_value=0.0) for m, d in tabs.items()}, index=idx)
    nc = pd.DataFrame({m: d.n_cells.reindex(idx, fill_value=0) for m, d in tabs.items()}, index=idx)
    buckets = pd.DataFrame({m: {"n_none": s.n_none, "n_diffuse": s.n_diffuse} for m, s in per.items()}).T
    return Comparison(levels, list(per), fz, nc, per, buckets, warn)


def difference(fuzzy: pd.DataFrame, how: str = "range", models: list[str] | None = None,
               pair: tuple[str, str] | None = None, pseudocount: float = 1.0) -> pd.Series:
    """Difference score per intersection across models.

    range: max - min; var: population variance; log_ratio: log2((a + pc) / (b + pc)) for pair=(a, b) (signed;
    sort by its absolute value). pseudocount is in fuzzy-size units (cells)."""
    if how == "log_ratio":
        if not pair or len(pair) != 2:
            raise ValueError("log_ratio needs exactly two models: pair=(a, b).")
        a, b = pair
        return np.log2((fuzzy[a] + pseudocount) / (fuzzy[b] + pseudocount)).rename("log2_ratio")
    X = fuzzy[models] if models else fuzzy
    if how == "range":
        return (X.max(axis=1) - X.min(axis=1)).rename("range")
    if how == "var":
        return X.var(axis=1, ddof=0).rename("variance")
    raise ValueError(f"unknown difference '{how}' (range | var | log_ratio)")


def ranked(cmp: Comparison, how: str = "range", hide_pure: bool = True, min_size: float = 0.0, **kw) -> pd.DataFrame:
    """Intersections sorted by difference (descending |score|), pure (single-label) ones hidden by default."""
    d = difference(cmp.fuzzy, how, **kw)
    df = pd.DataFrame({"labels": list(cmp.fuzzy.index), "names": [" & ".join(cmp.levels[i] for i in t) for t in cmp.fuzzy.index],
                       "degree": [len(t) for t in cmp.fuzzy.index], "score": d.to_numpy(),
                       "max_fuzzy_size": cmp.fuzzy.max(axis=1).to_numpy()})
    if hide_pure:
        df = df[df.degree > 1]
    df = df[df.max_fuzzy_size >= min_size]
    return df.iloc[np.argsort(-np.abs(df.score.to_numpy()), kind="stable")].reset_index(drop=True)


# ---- per-intersection attributes ----------------------------------------------------------------

def attributes(s: Signatures, second: np.ndarray | None = None, second_levels: list[str] | None = None,
               qc: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """For one model: composition of each intersection by a second categorical column (fractions) and mean QC."""
    ok = s.sig >= 0
    K = len(s.intersections)
    n = np.maximum(np.bincount(s.sig[ok], minlength=K), 1)
    out = pd.DataFrame(index=pd.Index(s.intersections, tupleize_cols=False, name="labels"))
    if second is not None:
        g = ok & (second >= 0)
        G = len(second_levels)
        comp = np.bincount(s.sig[g] * G + second[g], minlength=K * G).reshape(K, G) / n[:, None]
        for j, lv in enumerate(second_levels):
            out[f"comp:{lv}"] = comp[:, j]
    for name, v in (qc or {}).items():
        v = np.asarray(v, np.float64)
        f = ok & np.isfinite(v)
        out[f"mean:{name}"] = np.bincount(s.sig[f], weights=v[f], minlength=K) / np.maximum(np.bincount(s.sig[f], minlength=K), 1)
    return out


# ---- pipeline integration: cached per model --------------------------------------------------------

def select_models(models: pd.DataFrame, spec) -> list[str]:
    """'seed0' -> seed-0 replicate of every configuration; 'all'; or an explicit list of model ids."""
    if spec in (None, "seed0"):
        return models.model_id[models.seed == models.seed.min()].tolist()
    if spec == "all":
        return models.model_id.tolist()
    missing = [m for m in spec if m not in set(models.model_id)]
    if missing:
        raise ValueError(f"fuzzy_upset.models lists unknown model ids: {missing}")
    return list(spec)


def run(cfg: dict, label: str, tau: float = 0.1, max_size: int = 3, normalize: bool = False,
        models_spec=None, k: int | None = None) -> Comparison:
    """Load obs + kNN from the pipeline outputs and compare; per-model signatures are cached on disk keyed by
    (model, label column, k, tau, max_size, normalize)."""
    import anndata as ad
    fc = cfg.get("fuzzy_upset", {})
    k = k or fc.get("k", cfg["measures"]["k"])
    obs = ad.read_h5ad(p(cfg, "data"), backed="r").obs
    if label not in obs.columns:
        raise KeyError(f"label column '{label}' is not in obs (available: {list(obs.columns)}).")
    codes, levels = label_codes(obs[label].to_numpy(object))
    freq = label_frequency(codes, len(levels))
    models = pd.read_csv(out_dir(cfg) / "models.csv")
    ids = select_models(models, models_spec if models_spec is not None else fc.get("models", "seed0"))
    knn_path = out_dir(cfg) / "knn50.npy"
    knn = np.load(knn_path, mmap_mode="r") if knn_path.exists() else None
    order = {m: i for i, m in enumerate(models.model_id)}
    cache = out_dir(cfg, "fuzzy_upset", "cache")
    lab_hash = hashlib.sha1(codes.tobytes()).hexdigest()[:10]
    knn_stamp = knn_path.stat().st_mtime_ns if knn is not None else 0  # invalidates the cache when graphs are recomputed
    per, warn = {}, label_warnings(codes, levels)
    for mid in ids:
        if knn is None or order[mid] >= knn.shape[0]:
            warn.append(f"model '{mid}' has no kNN graph; it is left out of the comparison.")
            continue
        key = f"{mid}|{label}|{lab_hash}|{knn_stamp}|k{k}|t{tau}|s{max_size}|n{int(normalize)}"
        f = cache / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".npz")
        if f.exists():
            z = np.load(f, allow_pickle=False)
            per[mid] = Signatures(z["sig"], z["strength"], [tuple(int(x) for x in r if x >= 0) for r in z["inters"]],
                                  int(z["n_none"]), int(z["n_diffuse"]))
            continue
        P = memberships(np.asarray(knn[order[mid], :, :k]), codes, len(levels))
        s = signatures(P, tau, max_size, normalize, freq)
        pad = np.full((len(s.intersections), max_size), -1, np.int32)
        for i, t in enumerate(s.intersections):
            pad[i, :len(t)] = t
        np.savez(f, sig=s.sig, strength=s.strength, inters=pad, n_none=s.n_none, n_diffuse=s.n_diffuse)
        per[mid] = s
    for w in warn:
        log.warning(w)
    return _tables(per, levels, warn)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--label", required=True, help="obs column whose labels are intersected")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--normalize", action="store_true", help="threshold enrichment P/freq instead of P")
    ap.add_argument("--models", default=None, help="'seed0', 'all' or comma-separated model ids")
    ap.add_argument("--how", default="range", choices=["range", "var", "log_ratio"])
    ap.add_argument("--pair", default=None, help="two comma-separated model ids for --how log_ratio")
    ap.add_argument("--show-pure", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    spec = args.models if args.models in (None, "seed0", "all") else args.models.split(",")
    cmp = run(cfg, args.label, args.tau, args.max_size, args.normalize, spec)
    kw = {"pair": tuple(args.pair.split(","))} if args.how == "log_ratio" else {}
    r = ranked(cmp, args.how, hide_pure=not args.show_pure, **kw)
    d = out_dir(cfg, "fuzzy_upset")
    stem = f"{args.label}_tau{args.tau}_max{args.max_size}_{'enrich' if args.normalize else 'raw'}"
    tbl = cmp.fuzzy.copy()
    tbl.index = [" & ".join(cmp.levels[i] for i in t) for t in tbl.index]
    tbl.to_csv(d / f"{stem}_fuzzy_size.csv")
    r.assign(labels=r.labels.map(list)).to_csv(d / f"{stem}_ranked.csv", index=False)
    (d / f"{stem}_meta.json").write_text(json.dumps({"warnings": cmp.warnings, "buckets": cmp.buckets.to_dict("index")}, indent=1))
    log.info(f"{len(cmp.fuzzy)} intersections over {len(cmp.models)} models → {d}/{stem}_*")
    print(r.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
