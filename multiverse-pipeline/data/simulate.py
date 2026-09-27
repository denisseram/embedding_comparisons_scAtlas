"""Synthetic scRNA-seq count data with planted ground truth.

Planted effects (see README / config.yaml `simulate`):
  P1  close subtypes T-A / T-B separated by ~15 weak markers
  P2  tissue-confounded state: 'Ciliated' epithelium, only in tissue 2 (samples s5, s6 of study B)
  P3  low-quality cells (~3%, 80% from one sample): low counts, high mito, damped markers
  P4  doublets (~4%): sums of two cells of different types, with a noisy doublet score
  P5  donor-variable IGX_ gene family inside B cells (spurious sub-clusters)
  + gene-wise multiplicative study and sample batch effects

Usage:  python data/simulate.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import get_logger, load_config, p  # noqa: E402

log = get_logger("simulate")

CELL_TYPES = ["T-A", "T-B", "B", "Myeloid", "NK", "Epithelial", "Ciliated"]
LINEAGE = {"T-A": "T", "T-B": "T", "B": "B", "Myeloid": "Myeloid", "NK": "NK",
           "Epithelial": "Epithelial", "Ciliated": "Epithelial"}
LINEAGES = ["T", "B", "Myeloid", "NK", "Epithelial"]
# composition per tissue (tissue 2 has Ciliated instead of Epithelial)
COMP_T1 = {"T-A": .17, "T-B": .15, "B": .15, "Myeloid": .20, "NK": .10, "Epithelial": .23}
COMP_T2 = {"T-A": .17, "T-B": .15, "B": .15, "Myeloid": .20, "NK": .10, "Ciliated": .23}


def build_genes(s: dict, rng: np.random.Generator):
    """Return gene names, gene-group dict (name -> index array) and baseline log-means."""
    names, groups = [], {}

    def add(group, prefix, n):
        start = len(names)
        names.extend(f"{prefix}{i + 1}" for i in range(n))
        groups[group] = np.arange(start, start + n)

    add("mito", "MT-", s["n_mito_genes"])
    for lin in LINEAGES:
        add(f"mk_{lin}", f"MK{lin[:3].upper()}_", s["n_markers_per_type"])
    add("p1_TA", "TSUBA_", (s["p1_n_weak_markers"] + 1) // 2)
    add("p1_TB", "TSUBB_", s["p1_n_weak_markers"] // 2)
    add("p2_state", "CIL_", s["p2_n_state_genes"])
    add("secondary", "SEC_", s["n_secondary_markers"])
    add("igx", "IGX_", s["p5_n_igx"])
    add("ifn", "ISG_", s["n_ifn_genes"])
    n_bg = s["n_genes"] - len(names)
    if n_bg < 100:
        raise ValueError("n_genes too small for the configured gene blocks")
    add("background", "GENE_", n_bg)

    base = rng.normal(0.0, 1.0, len(names))            # log baseline expression
    base[groups["mito"]] = rng.normal(2.2, 0.3, len(groups["mito"]))
    # weak P1 markers are moderately expressed so a 1.8x change is detectable in counts
    base[groups["p1_TA"]] = rng.normal(1.2, 0.2, len(groups["p1_TA"]))
    base[groups["p1_TB"]] = rng.normal(1.2, 0.2, len(groups["p1_TB"]))
    base[groups["igx"]] = rng.normal(-1.5, 0.3, len(groups["igx"]))
    return np.array(names), groups, base


def type_logprofiles(s, groups, base, rng):
    """Log-mean profile (unnormalised) for each cell type."""
    n_genes = base.size
    sec = groups["secondary"]
    sec_lineage = rng.integers(0, len(LINEAGES), sec.size)
    sec_sign = rng.choice([-1.0, 1.0], sec.size, p=[0.3, 0.7])
    prof = {}
    for ct in CELL_TYPES:
        lp = base.copy()
        lin = LINEAGE[ct]
        lp[groups[f"mk_{lin}"]] += s["marker_log_fc"]
        li = LINEAGES.index(lin)
        m = sec_lineage == li
        lp[sec[m]] += s["secondary_log_fc"] * sec_sign[m]
        if ct == "T-A":
            lp[groups["p1_TA"]] += s["p1_log_fc"]
        if ct == "T-B":
            lp[groups["p1_TB"]] += s["p1_log_fc"]
        if ct == "Ciliated":
            lp[groups["p2_state"]] += s["p2_log_fc"] + 1.0
        assert lp.size == n_genes
        prof[ct] = lp
    return prof


def nb_sample(mu: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson-gamma (negative binomial) with var = mu + phi*mu^2."""
    lam = rng.gamma(shape=1.0 / phi, scale=mu * phi)
    return rng.poisson(lam).astype(np.int32)


def simulate(cfg: dict) -> tuple[ad.AnnData, pd.DataFrame, dict]:
    s = cfg["simulate"]
    rng = np.random.default_rng(cfg["global_seed"])
    names, groups, base = build_genes(s, rng)
    G = names.size
    prof = type_logprofiles(s, groups, base, rng)

    # --- cells: sample, study, tissue, cell type ---------------------------------------
    samples = [(smp, st) for st, smps in s["studies"].items() for smp in smps]
    N = s["n_cells"]
    sample_sizes = rng.multinomial(N, rng.dirichlet(np.full(len(samples), 30.0)))
    obs_rows = []
    for (smp, st), n in zip(samples, sample_sizes):
        tissue = "tissue2" if smp in s["tissue2_samples"] else "tissue1"
        comp = COMP_T2 if tissue == "tissue2" else COMP_T1
        cts = rng.choice(list(comp), size=n, p=np.array(list(comp.values())))
        obs_rows += [(smp, st, tissue, ct) for ct in cts]
    obs = pd.DataFrame(obs_rows, columns=["sample", "study", "tissue", "cell_type"])
    obs.index = [f"cell{i:06d}" for i in range(N)]

    # --- planted effect assignment -------------------------------------------------------
    n_low = int(round(s["p3_fraction"] * N))
    main = np.flatnonzero(obs["sample"].values == s["p3_main_sample"])
    rest = np.flatnonzero(obs["sample"].values != s["p3_main_sample"])
    n_main = int(round(s["p3_main_sample_share"] * n_low))
    low_idx = np.concatenate([rng.choice(main, n_main, replace=False),
                              rng.choice(rest, n_low - n_main, replace=False)])
    is_low = np.zeros(N, bool)
    is_low[low_idx] = True
    n_dbl = int(round(s["p4_fraction"] * N))
    dbl_idx = rng.choice(np.flatnonzero(~is_low), n_dbl, replace=False)
    is_dbl = np.zeros(N, bool)
    is_dbl[dbl_idx] = True

    # --- batch effects -------------------------------------------------------------------
    bs = s["batch_strength"]
    study_eff = {st: rng.normal(0, bs["study_log_sd"], G) for st in s["studies"]}
    sample_eff = {smp: rng.normal(0, bs["sample_log_sd"], G) for smp, _ in samples}
    # P5: IGX genes vary strongly by sample (on top of generic batch effects)
    igx = groups["igx"]
    for smp in sample_eff:
        sample_eff[smp][igx] += rng.normal(0, 0.8, igx.size)

    # IFN-like program: active in a subset of cells, more often in s1
    p_ifn = np.where(obs["sample"].values == "s1", 0.30, 0.10)
    ifn_on = rng.random(N) < p_ifn

    # IGX clonotype-like modules: each B cell expresses one random third of the family
    n_mod = 3
    igx_modules = np.array_split(rng.permutation(igx), n_mod)
    is_B = obs["cell_type"].values == s["p5_cell_type"]
    igx_module_of_cell = np.where(is_B, rng.integers(0, n_mod, N), -1)

    lib = rng.lognormal(np.log(s["mean_library_size"]), 0.35, N)
    lib[is_low] *= 0.2

    # partner cell type for doublets (different type, same sample)
    partner = np.array([None] * N, dtype=object)
    for i in dbl_idx:
        other = [c for c in CELL_TYPES if c != obs["cell_type"].iat[i]
                 and (c != "Ciliated" or obs["tissue"].iat[i] == "tissue2")
                 and (c != "Epithelial" or obs["tissue"].iat[i] == "tissue1")]
        partner[i] = rng.choice(other)

    def cell_logmean(i: int, ct: str) -> np.ndarray:
        lp = prof[ct].copy()
        if ifn_on[i]:
            lp[groups["ifn"]] += 1.5
        if igx_module_of_cell[i] >= 0 and ct == s["p5_cell_type"]:
            lp[igx] += 2.0
            lp[igx_modules[igx_module_of_cell[i]]] += 3.0
        if is_low[i]:  # damped markers + high mito
            lp = base + 0.3 * (lp - base)
            lp[groups["mito"]] += 1.8
        smp = obs["sample"].iat[i]
        return lp + study_eff[obs["study"].iat[i]] + sample_eff[smp]

    def to_mu(lp: np.ndarray, libsize: float) -> np.ndarray:
        w = np.exp(lp - lp.max())
        return libsize * w / w.sum()

    phi = s["nb_dispersion"]
    chunks, chunk = [], 5000
    for start in range(0, N, chunk):
        stop = min(N, start + chunk)
        mu = np.empty((stop - start, G))
        for r, i in enumerate(range(start, stop)):
            mu[r] = to_mu(cell_logmean(i, obs["cell_type"].iat[i]), lib[i])
        X = nb_sample(mu, phi, rng)
        for r, i in enumerate(range(start, stop)):
            if is_dbl[i]:  # add a second cell of a different type from the same sample
                mu2 = to_mu(cell_logmean(i, partner[i]), rng.lognormal(np.log(s["mean_library_size"]), 0.35))
                X[r] += nb_sample(mu2[None, :], phi, rng)[0]
        chunks.append(sp.csr_matrix(X))
    X = sp.vstack(chunks).tocsr()

    total = np.asarray(X.sum(1)).ravel()
    mito = np.asarray(X[:, groups["mito"]].sum(1)).ravel()
    obs["total_counts"] = total.astype(np.float32)
    obs["log_total_counts"] = np.log1p(total).astype(np.float32)
    obs["pct_mito"] = (100 * mito / np.maximum(total, 1)).astype(np.float32)
    ds = np.where(is_dbl, rng.beta(6, 3, N), rng.beta(2, 8, N))
    obs["doublet_score"] = ds.astype(np.float32)

    # ground truth: boolean membership per effect + a primary label (priority order)
    gt = pd.DataFrame(index=obs.index)
    gt["P1_close_subtypes"] = obs["cell_type"].isin(["T-A", "T-B"]).values & ~is_dbl & ~is_low
    gt["P2_tissue_state"] = (obs["cell_type"].values == "Ciliated") & ~is_dbl & ~is_low
    gt["P3_low_quality"] = is_low
    gt["P4_doublet"] = is_dbl
    gt["P5_igx"] = is_B & ~is_dbl & ~is_low
    primary = np.full(N, "none", dtype=object)
    for col in ["P5_igx", "P1_close_subtypes", "P2_tissue_state", "P3_low_quality", "P4_doublet"]:
        primary[gt[col].values] = col
    gt["planted_effect"] = primary
    obs["planted_effect"] = pd.Categorical(primary)
    obs["doublet_partner"] = [x if x is not None else "" for x in partner]
    for c in ["sample", "study", "tissue", "cell_type"]:
        obs[c] = pd.Categorical(obs[c])

    var = pd.DataFrame(index=names)
    var["mito"] = np.isin(np.arange(G), groups["mito"])
    var["igx"] = np.isin(np.arange(G), igx)
    group_of = np.empty(G, dtype=object)
    for gname, idx in groups.items():
        group_of[idx] = gname
    var["gene_group"] = pd.Categorical(group_of)

    adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    adata.layers["counts"] = X
    adata.uns["planted"] = {"igx_modules": [names[m].tolist() for m in igx_modules]}

    gene_sets = {
        "markers_T-A": names[np.r_[groups["mk_T"], groups["p1_TA"]]].tolist(),
        "markers_T-B": names[np.r_[groups["mk_T"], groups["p1_TB"]]].tolist(),
        "markers_B": names[groups["mk_B"]].tolist(),
        "markers_Myeloid": names[groups["mk_Myeloid"]].tolist(),
        "markers_NK": names[groups["mk_NK"]].tolist(),
        "markers_Epithelial": names[groups["mk_Epithelial"]].tolist(),
        "markers_Ciliated": names[np.r_[groups["mk_Epithelial"], groups["p2_state"]]].tolist(),
        "interferon": names[groups["ifn"]].tolist(),
        "IGX": names[igx].tolist(),
    }
    return adata, gt, gene_sets


def summary(adata: ad.AnnData, gt: pd.DataFrame) -> str:
    o = adata.obs
    lines = [f"cells x genes: {adata.n_obs} x {adata.n_vars}",
             f"mean total counts: {o.total_counts.mean():.0f}  "
             f"median pct_mito: {o.pct_mito.median():.2f}"]
    lines.append("\ncell_type x sample:\n" + pd.crosstab(o.cell_type, o["sample"], margins=True).to_string())
    lines.append("\nsample -> study / tissue:\n" + o.groupby("sample", observed=True)[["study", "tissue"]].first().to_string())
    eff = gt.drop(columns="planted_effect").sum().rename("n_cells").to_frame()
    eff["pct"] = (100 * eff.n_cells / len(gt)).round(2)
    lines.append("\nplanted effects (membership, may overlap):\n" + eff.to_string())
    low = gt.P3_low_quality.values
    lines.append(f"\nP3 share from main sample: {o['sample'][low].value_counts(normalize=True).iloc[0]:.2f}; "
                 f"total_counts low vs rest: {o.total_counts[low].median():.0f} vs {o.total_counts[~low].median():.0f}; "
                 f"pct_mito: {o.pct_mito[low].median():.1f} vs {o.pct_mito[~low].median():.1f}")
    d = gt.P4_doublet.values
    lines.append(f"P4 doublet_score mean: doublets {o.doublet_score[d].mean():.2f} vs singlets {o.doublet_score[~d].mean():.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    adata, gt, gene_sets = simulate(cfg)
    out = p(cfg, "data")
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out, compression="gzip")
    gt.to_csv(p(cfg, "ground_truth"), index_label="cell_id")
    Path(p(cfg, "gene_sets")).write_text(json.dumps(gene_sets, indent=1))
    print(summary(adata, gt))
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
