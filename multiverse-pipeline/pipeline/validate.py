"""Validation checks against the planted ground truth (brief section 6).

Operationalisation (fixed before any check was computed; thresholds are the brief's):
  effect region  = the region with the highest fraction of its cells belonging to the effect
  attribution    = argmax over real factors (seed excluded) of the mean E over the effect's cells
  V1 Null        : seed pseudo-factor median |E| < 0.5 AND fraction(|E| > 2) < 0.05
  V2 P2          : attribution == batch_key AND P2 region ranks <= 3 among regions by mean E[batch_key]
  V3 P5          : attribution == exclude_igx
  V4 P1          : attribution == n_hvg
  V5 P3 / P4     : effect region ranks <= 3 by qc_deviation (P3) / mean doublet_score (P4)
                   AND <= 3 lowest by all-model consensus stability
  V6 baselines   : precision@k (k = number of effect cells) and AUROC for (a) paper-style graph
                   dissimilarity, (b) majority-vote disagreement, (c) max_F |E(c,F)|. Report only.
  V7 leaderboard : tested in tests/test_export.py on the exported models.json.
Diagnostics (reported, not gating): specificity AUROC of E[expected factor] for effect cells
vs unaffected ('none') cells.
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.common import ROOT, load_config, out_dir, p

EXPECTED = {"P1_close_subtypes": "n_hvg", "P2_tissue_state": "batch_key", "P5_igx": "exclude_igx"}
EFFECTS = ["P1_close_subtypes", "P2_tissue_state", "P3_low_quality", "P4_doublet", "P5_igx"]


def load(cfg: dict, mode: str = "") -> dict:
    od = out_dir(cfg, "measures")
    sfx = f"_{mode}" if mode else ""
    meta = json.loads((od / "meta.json").read_text())
    reg = json.loads((od / f"regions{sfx}.json").read_text())["regions"] if (od / f"regions{sfx}.json").exists() \
        else json.loads((od / "regions.json").read_text())["regions"]
    return dict(
        factors=meta["factors"], E=np.load(od / f"E{sfx}.npy"),
        regions=np.load(od / f"regions{sfx}.npy") if (od / f"regions{sfx}.npy").exists() else np.load(od / "regions.npy"),
        region_table=reg,
        stab_all=np.load(od / "stability_all.npy"),
        gt=pd.read_csv(p(cfg, "ground_truth"), index_col=0),
        obs=ad.read_h5ad(p(cfg, "data"), backed="r").obs,
        gd=np.load(od / "paper_graph_dissim.npy"),
        mv=np.load(out_dir(cfg) / "cell_majority_disagreement.npy"),
    )


def effect_region(d: dict, effect: str) -> int:
    member = d["gt"][effect].to_numpy(bool)
    frac = [member[d["regions"] == r].mean() for r in range(d["regions"].max() + 1)]
    return int(np.argmax(frac))


def rank_of(values: list[float], idx: int, descending: bool = True) -> int:
    order = np.argsort(-np.asarray(values) if descending else np.asarray(values), kind="stable")
    return int(np.flatnonzero(order == idx)[0]) + 1


def attribution(d: dict, effect: str) -> tuple[str, dict]:
    member = d["gt"][effect].to_numpy(bool)
    means = {F: float(d["E"][member, i].mean()) for i, F in enumerate(d["factors"]) if F != "seed"}
    return max(means, key=means.get), means


def run_checks(cfg: dict, mode: str = "") -> dict:
    d = load(cfg, mode)
    F = d["factors"]
    res: dict = {"mode": mode or "jaccard", "checks": {}, "diagnostics": {}}
    si = F.index("seed")
    e_seed = np.abs(d["E"][:, si])
    med, frac = float(np.median(e_seed)), float((e_seed > 2).mean())
    res["checks"]["V1_null"] = {"pass": med < 0.5 and frac < 0.05,
                                "detail": f"seed median |E| = {med:.3f} (<0.5), fraction |E|>2 = {frac:.4f} (<0.05)"}

    tbl = d["region_table"]
    bk = F.index("batch_key")
    top, means = attribution(d, "P2_tissue_state")
    r2 = effect_region(d, "P2_tissue_state")
    rank = rank_of([t["E_mean"]["batch_key"] for t in tbl], r2)
    res["checks"]["V2_P2"] = {"pass": top == "batch_key" and rank <= 3,
                              "detail": f"attributed to {top} (means {fmt(means)}); P2 region R{r2} ranks {rank} by E[batch_key]"}
    for key, eff in [("V3_P5", "P5_igx"), ("V4_P1", "P1_close_subtypes")]:
        top, means = attribution(d, eff)
        res["checks"][key] = {"pass": top == EXPECTED[eff],
                              "detail": f"attributed to {top}; expected {EXPECTED[eff]} (means {fmt(means)})"}
    parts, ok = [], True
    for eff, col in [("P3_low_quality", "qc_deviation"), ("P4_doublet", "doublet_score")]:
        r = effect_region(d, eff)
        vals = [t["qc_deviation"] if col == "qc_deviation" else t["qc"]["doublet_score"]["mean"] for t in tbl]
        rq = rank_of(vals, r)
        rs = rank_of([t["stability_all"] for t in tbl], r, descending=False)
        frac_eff = d["gt"][eff].to_numpy(bool)[d["regions"] == r].mean()
        ok &= rq <= 3 and rs <= 3
        parts.append(f"{eff}: region R{r} ({100 * frac_eff:.0f}% effect cells) ranks {rq} by {col}, "
                     f"{rs} lowest by stability ({tbl[r]['stability_all']:.3f})")
    res["checks"]["V5_P3_P4"] = {"pass": bool(ok), "detail": "; ".join(parts)}

    # V6 baselines
    ours = np.abs(d["E"][:, [i for i, f in enumerate(F) if f != "seed"]]).max(1)
    scores = {"paper_graph_dissimilarity": d["gd"], "majority_vote_disagreement": d["mv"], "ours_max_abs_E": ours}
    rows = []
    for eff in EFFECTS:
        y = d["gt"][eff].to_numpy(bool)
        k = int(y.sum())
        for name, s in scores.items():
            top_k = np.argsort(-s, kind="stable")[:k]
            rows.append({"effect": eff, "score": name, "k": k, "precision_at_k": float(y[top_k].mean()),
                         "auroc": float(roc_auc_score(y, s)), "base_rate": float(y.mean())})
    res["baselines"] = rows

    # diagnostics: specificity of the expected factor
    none = d["gt"]["planted_effect"].to_numpy(str) == "none"
    for eff, fac in EXPECTED.items():
        y = d["gt"][eff].to_numpy(bool)
        s = d["E"][:, F.index(fac)]
        m = y | none
        res["diagnostics"][eff] = {"factor": fac, "auroc_vs_unaffected": float(roc_auc_score(y[m], s[m])),
                                   "mean_E_effect": float(s[y].mean()), "mean_E_unaffected": float(s[none].mean())}
    return res


def fmt(means: dict) -> str:
    return ", ".join(f"{k}={v:.2f}" for k, v in means.items())


def to_markdown(results: list[dict]) -> str:
    out = []
    for res in results:
        out.append(f"### Measure: `{res['mode']}`\n")
        out.append("| check | result | detail |\n|---|---|---|")
        for k, v in res["checks"].items():
            out.append(f"| {k} | {'PASS' if v['pass'] else '**FAIL**'} | {v['detail']} |")
        out.append("\n**V6 baseline comparison** (precision@k, k = number of effect cells; AUROC)\n")
        df = pd.DataFrame(res["baselines"])
        pk = df.pivot(index="effect", columns="score", values="precision_at_k")
        au = df.pivot(index="effect", columns="score", values="auroc")
        base = df.groupby("effect").base_rate.first()
        out.append("| effect | base rate | " + " | ".join(f"P@k {c}" for c in pk.columns) + " | "
                   + " | ".join(f"AUROC {c}" for c in au.columns) + " |")
        out.append("|---" * (2 + 2 * len(pk.columns)) + "|")
        for eff in pk.index:
            out.append(f"| {eff} | {base[eff]:.3f} | " + " | ".join(f"{x:.3f}" for x in pk.loc[eff]) + " | "
                       + " | ".join(f"{x:.3f}" for x in au.loc[eff]) + " |")
        out.append("\n**Diagnostic: specificity of the expected factor** (not a pass/fail check)\n")
        out.append("| effect | factor | mean E (effect cells) | mean E (unaffected) | AUROC effect vs unaffected |\n|---|---|---|---|---|")
        for eff, v in res["diagnostics"].items():
            out.append(f"| {eff} | {v['factor']} | {v['mean_E_effect']:.2f} | {v['mean_E_unaffected']:.2f} | {v['auroc_vs_unaffected']:.3f} |")
        out.append("")
    return "\n".join(out)


def main(cfg: dict | None = None, modes: tuple[str, ...] = ("",)) -> list[dict]:
    cfg = cfg or load_config()
    results = [run_checks(cfg, m) for m in modes]
    (out_dir(cfg) / "validation.json").write_text(json.dumps(results, indent=1))
    (out_dir(cfg) / "validation_results.md").write_text(to_markdown(results))
    print(to_markdown(results))
    return results


if __name__ == "__main__":
    import sys
    main(modes=tuple(sys.argv[1:]) or ("",))
