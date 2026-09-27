# Validation report

Default toy dataset (5,000 cells × 2,000 genes, 144 embeddings), `config.yaml`, global seed 20260927.
Every number below was produced by `make validate` (`pipeline/validate.py`) or `make test`;
the raw output is in `outputs/toy/validation_results.md` and `outputs/toy/validation.json`.
Re-running the full pipeline from scratch reproduced these numbers exactly.

## Summary

| check | neighbour identity (pre-registered, primary) | neighbour composition (exploratory, post hoc) |
|---|---|---|
| V1 null (seed pseudo-factor) | **PASS** | **PASS** |
| V2 P2 → `batch_key`, region top-3 | **FAIL** (attributed to `n_hvg`; region rank 1) | **FAIL** (attributed to `batch_key`; region rank 11) |
| V3 P5 → `exclude_igx` | **PASS** (specific) | **PASS** (specific) |
| V4 P1 → `n_hvg` | **PASS**, but *not specific* | **PASS**, but *not specific* |
| V5 P3/P4 high QC / doublet score and low stability | **FAIL** (QC part passes, stability part fails) | **FAIL** (same) |
| V6 baseline comparison | reported below: baselines win for P1, P2, P4 | reported below |
| V7 leaderboard sanity | **PASS** (`tests/test_export.py`) | – |

In `tests/test_validation.py` the failing checks are marked `xfail(strict=True)`. They still show
up as failures (XFAIL) in every test run, and the suite errors if one of them starts passing,
so this record cannot go stale silently.

**Bottom line.** The system correctly separates decision effects from rerun noise (V1). It also
localises and attributes one planted effect cleanly: the donor-variable IGX family (P5). It does
**not** localise the close-subtype (P1) or tissue-confounded (P2) effects. The reason is not a
bug. Neighbour-identity change saturates everywhere, described in the diagnosis below.
Majority-vote label disagreement, a simple paper-style baseline, ranks P1, P2 and P4 cells better
than our |E|.

## How the checks were defined, and in what order things happened

To be transparent about what was and was not fixed in advance:

1. **Simulator calibration (before any measure existed).** The weak P1 markers were first too weak:
   T-A/T-B were not separable even at 1,500 HVGs (kNN label purity 0.49). I swept only
   `p1_log_fc` against two *design-level* criteria (few markers in the top 500 HVGs; T-A/T-B
   separable at 1,500) and chose 1.2: 3/15 markers at 500 HVGs (purity 0.51), 15/15 at 1,500
   (purity 0.80). No multiverse measure or validation check was consulted.
2. **Validation operationalisation.** The thresholds are the brief's. The operational details
   (effect region = the region with the highest fraction of effect cells; attribution = argmax of
   the mean E over effect cells; "top 3" / "3 lowest" for V5) were written in `pipeline/validate.py`
   *after* I had seen one summary table of mean E per planted effect, but *before* any check was
   computed. They were not changed afterwards.
3. **Exploratory variant.** The neighbour-composition measure was added *after* the primary
   measure failed V2/V5, so it is post hoc. Its one free parameter (micro-cluster size ≈ 5·k
   cells, i.e. 66 blocks) was fixed by rule before it was run. It was run once and not iterated.
   Its results are reported as they came out and do not replace the primary results.

## Detailed results — neighbour identity (primary)

| check | result | detail |
|---|---|---|
| V1_null | PASS | seed median \|E\| = 0.252 (<0.5), fraction \|E\|>2 = 0.0012 (<0.05) |
| V2_P2 | **FAIL** | attributed to n_hvg (means n_hvg=3.18, hvg_batch_aware=2.57, exclude_igx=2.10, batch_key=2.11, method=1.11); P2 region R11 ranks 1 by E[batch_key] |
| V3_P5 | PASS | attributed to exclude_igx (means n_hvg=3.35, hvg_batch_aware=2.57, exclude_igx=4.01, batch_key=2.56, method=1.88) |
| V4_P1 | PASS | attributed to n_hvg (means n_hvg=3.27, hvg_batch_aware=2.65, exclude_igx=2.17, batch_key=2.15, method=1.47) |
| V5_P3_P4 | **FAIL** | P3: region R9 (100% effect cells) ranks 1 by QC deviation but 6th lowest by stability (0.011); P4: region R12 (97% effect cells) ranks 1 by doublet score but 13th (= highest) by stability (0.293) |

**Specificity diagnostic** (not a pass/fail check): AUROC of E[expected factor] for effect cells vs unaffected cells.

| effect | factor | mean E (effect) | mean E (unaffected) | AUROC |
|---|---|---|---|---|
| P1 close subtypes | n_hvg | 3.27 | 3.26 | 0.520 |
| P2 tissue state | batch_key | 2.11 | 2.20 | 0.409 |
| P5 IGX | exclude_igx | 4.01 | 2.09 | 1.000 |

V4 "passes" only because `n_hvg` is the largest factor for *every* cell. The effect on T cells is
indistinguishable from the effect on unaffected cells (AUROC 0.52).

## Detailed results — neighbour composition (exploratory)

| check | result | detail |
|---|---|---|
| V1_null | PASS | seed median \|E\| = 0.177, fraction \|E\|>2 = 0.0000 |
| V2_P2 | **FAIL** | attributed to batch_key (means n_hvg=2.24, hvg_batch_aware=0.88, exclude_igx=0.59, batch_key=3.89, method=2.35); P2 region R15 ranks 11 by E[batch_key] |
| V3_P5 | PASS | exclude_igx = 6.03 vs next 4.10 |
| V4_P1 | PASS | n_hvg = 3.35, tied with batch_key = 3.35 |
| V5_P3_P4 | **FAIL** | P3 region ranks 1 by QC deviation, 8th lowest stability; P4 region (23% doublets) ranks 1 by doublet score, highest stability |

Specificity AUROC: P1/n_hvg 0.538, P2/batch_key 0.365, P5/exclude_igx 0.984. The attribution for P2
is now correct, but `batch_key` changes neighbourhood composition *everywhere* (mean E 4.13 on
unaffected cells vs 3.89 on P2 cells). So the P2 cells do not stand out.

## V6 — baseline comparison

Precision@k (k = number of effect cells) and AUROC for ranking cells by each score. Base rate =
precision of a random ranking. Primary measure shown; composition values in `outputs/toy/validation_results.md`.

| effect | base rate | P@k majority vote | P@k ours max\|E\| | P@k graph dissim. | AUROC majority vote | AUROC ours | AUROC graph dissim. |
|---|---|---|---|---|---|---|---|
| P1 close subtypes | 0.292 | **0.513** | 0.190 | 0.274 | **0.723** | 0.438 | 0.525 |
| P2 tissue state | 0.080 | **0.104** | 0.005 | 0.005 | **0.827** | 0.369 | 0.223 |
| P3 low quality | 0.030 | 0.000 | 0.000 | **0.033** | **0.654** | 0.404 | 0.333 |
| P4 doublet | 0.040 | **0.415** | 0.180 | 0.000 | **0.812** | 0.493 | 0.070 |
| P5 IGX | 0.145 | 0.000 | **0.639** | 0.401 | 0.282 | **0.907** | 0.774 |

The baselines win for P1, P2 and P4; majority vote is the strongest of the three. Our measure wins
clearly for P5. For P1, P2 and P3 our global score ranks planted cells *below* random (AUROC < 0.5).
Majority vote uses the `cell_type` labels, which our measures do not, so the comparison is not
label-free on both sides. It is still the honest comparison the brief asked for.

## Diagnosis: why neighbour identity fails for P1 and P2

The planted structure does change as designed. The measure cannot see it:

| quantity (from `knn50.npy`, k=15) | T-A | B | Ciliated | Epithelial | Myeloid |
|---|---|---|---|---|---|
| Δ between two seeds (harmony, study) | 0.61 | 0.38 | 0.61 | 0.60 | 0.59 |
| Δ harmony study ↔ sample (batch_key) | 0.84 | 0.84 | 0.84 | 0.83 | 0.83 |
| Δ pca 500 ↔ 1500 HVGs (n_hvg) | 0.90 | 0.67 | 0.87 | 0.86 | 0.87 |
| Δ pca exclude_igx no ↔ yes | 0.76 | 0.83 | 0.75 | 0.76 | 0.72 |

At the same time, same-label kNN purity changes exactly where it should:

- T-A: 0.55 (500 HVGs) → 0.81 (1,500 HVGs).
- Ciliated: 0.98 (harmony, study) → 0.70 (harmony, sample).
- Ciliated under ComBat: 0.98 → 0.96.

Inside a homogeneous population of hundreds of cells, *any* change in features or correction
reshuffles which of the equally close same-type cells become the 15 nearest neighbours. So Δ
saturates around 0.8–0.9 for every cell type. Neighbour identity therefore cannot distinguish
"reshuffled within the same population" from "moved into a different population". The seed noise
baseline (Δ ≈ 0.6) is much smaller than this churn, so every factor gets a large E everywhere,
roughly 2–3 seed-noise units. Only effects that are larger than the churn (P5) stand out.

The composition variant removes the within-population reshuffling. It then picks up a different
global signal: sample/batch structure inside cell types changes with `batch_key` everywhere.

## Diagnosis: V5 stability

- Consensus stability over **all** 144 models is almost zero everywhere (median 0.00). At the
  brief's threshold, almost no neighbour is shared by ≥80% of models, because of the same churn.
  Over seed replicates only, the median is 0.35.
- Doublets are *more* stable than other cells (all-model median 0.20, seed-only 0.71, vs about 0.0
  and 0.34 elsewhere). My hypothesis, which I have not tested: summed profiles form a small,
  distinct group whose members are each other's nearest neighbours in every embedding.
  The planted "doublets are unstable" expectation does not hold in this simulator.
- The QC half of V5 works: the P3 and P4 regions rank 1st by QC deviation and doublet score in V7.

## Possible next steps (not implemented, not validated)

- Judge a factor's effect against the factor's *typical* effect, not only against seed noise.
  The dashboard offers "E relative to each factor's median" in V4 as a clearly labelled display
  transform; it has not been validated.
- Measure change at the population level, for example neighbour composition over a partition at
  the right granularity, or distances as in the paper's graph dissimilarity, combined with
  seed calibration.
- Use a lower consensus threshold, or a rank-based stability, so that V2 is not uniformly zero.

## Browser checks

Automated with Playwright (`check.mjs`, 36 checks). It covers: no console errors, every view renders, the
dropdown recolours V0a and updates V0b, lasso / click linking across V0a, V0b, V1, V3, V4, V5, V6
and V7, measure-mode switch, reset, and V8 JSON/Markdown downloads. Results are in the README.
