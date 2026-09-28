# Embedding Multiverse Explorer — pipeline and dashboard

A working toy prototype. It trains every path through a small decision space of single-cell
integrations, measures **which decisions reshape which cells** against a seed-replicate noise
baseline, and ships the evidence to a static, in-browser dashboard at `/multiverse` in the
Astro site (`../integration`).

> **Read [VALIDATION.md](VALIDATION.md) before trusting any view.** On the planted ground truth
> the pre-registered measure recovers the null and the IGX effect, but not the close-subtype or
> tissue-confounded effects. A simple baseline beats it on three of five effects.

## Quick start

```bash
conda create -n multiverse -c conda-forge python=3.12 pip
conda run -n multiverse pip install numpy scipy pandas scikit-learn anndata scanpy umap-learn \
    pynndescent leidenalg igraph harmonypy pyyaml pytest h5py
cd multiverse-pipeline
make data       # simulate data/toy.h5ad (+ ground_truth.csv, gene_sets.json)      ~5 s
make run        # preprocess → integrate → kNN → metrics → measures → regions → variants → export  ~6 min
make test       # unit + export-schema + validation tests
make validate   # print the validation table (both measure modes)
make smoke      # 10-second plumbing check on Iris
make scale      # 50,000 cells × 48 models scale test (writes only under outputs/scale_test)
```

`harmonypy` is not on conda-forge, so the Python packages are installed with pip inside the conda
env. The Makefile finds the `multiverse` env automatically; override it with `make PY=/path/to/python`.
All decisions, seeds, k values, weights and paths are in [`config.yaml`](config.yaml). The global
seed is `20260927`.

## What the pipeline does

| step | module | output (under `outputs/toy/`) |
|---|---|---|
| simulate | `data/simulate.py` | `data/toy.h5ad`: 5,000 cells × 2,000 genes, 2 studies × 3 samples, 2 tissues, 7 cell types, planted P1–P5 + batch effects |
| HVG sets | `pipeline/preprocess.py` | `hvg_sets.json` (12 distinct feature sets) |
| integrate | `pipeline/integrate.py` | `latent/<model_id>.npy` (30-d), `models.csv`: 48 configurations × 3 seeds = 144 embeddings |
| kNN | `pipeline/knn.py` | `knn50.npy` [144, N, 50] (pynndescent in latent space; k=15 = first 15 columns) |
| per-model UMAPs | `pipeline/umaps.py` | `umap_per_model.npy` [144, N, 2]: standard scanpy UMAP of each embedding, for visual QC only |
| metrics | `pipeline/metrics.py` | `metrics.csv` (raw + min–max scaled, bio / batch / overall), `metrics_seed_sd.csv` |
| measures | `pipeline/measures.py` | Δ, seed noise, z, E/H per factor, agreement A(a,b), consensus stability, z vs references |
| regions | `pipeline/regions.py` | Leiden on the E-profile with a consensus-kNN spatial constraint; per-region compositions, QC, Wilcoxon top genes |
| variants | `pipeline/variants.py` | per region × model: typical / merged / split / joined |
| fuzzy UpSet | `pipeline/fuzzy_upset.py` | neighbour-label memberships per model (exported by the export step); CLI tables under `fuzzy_upset/` |
| export | `pipeline/export.py` | `../integration/public/multiverse-data/` (see its `README.md`) |
| validate | `pipeline/validate.py` | `validation_results.md`, `validation.json` |

## Runtime and memory (Apple M1, 8 GB RAM)

**Default toy run** (`make data run`, measured 2026-09-27; timings from `outputs/toy/run_all_timings.json`):

| step | seconds |
|---|---|
| simulate | 4 |
| preprocess | 4.4 |
| integrate (144 embeddings) | 25.9 |
| kNN (pynndescent, k=50) | 120.0 |
| per-model UMAPs (132 unique, 4 workers; added 2026-09-28) | 298.5 |
| metrics (4 worker processes) | 89.4 |
| measures (both modes) | 16.1 |
| regions (both modes) | 3.8 |
| variants (both modes) | 81.3 |
| export | 26.6 |
| **total** | **≈ 670 s (11 min)** including per-model UMAPs, under the 15-minute target |

Peak RSS: 0.85 GB for the main process (`ru_maxrss`). During the metrics step, the sampled RSS of the whole process tree (main + 4 workers, `pipeline/memwatch.py`, 0.5 s sampling) peaked at 0.71 GB. Sampling can miss short spikes, but everything stays far below the 8 GB of the machine.

**Scale test** (`make scale`: 50,000 cells, 48 models = 16 configurations × 3 seeds;
`hvg_batch_aware` fixed to no, methods pca + harmony, one measure mode):

| step | first full run (s) | rerun after the region-fallback fix (s) |
|---|---|---|
| simulate | 23 | – |
| preprocess | 7.8 | – |
| integrate (48 embeddings) | 104.9 | – |
| kNN | 264.4 | – |
| metrics (3 workers) | 597.8 | 529.6 |
| measures | 21.3 | 22.4 |
| regions | 58.5 (fell through to 70 regions: bug) | 18.0 (11 regions) |
| variants | 183.0 | 194.5 |
| export | 106.3 | 106.9 |
| **total** | **≈ 1,372 s (23 min)** incl. simulate | |

Memory: the main process peaked at 1.6 GB RSS (`ru_maxrss`), and the sampled whole-process-tree RSS
during the rerun peaked at 1.4 GB. The exported web data is 28 MB, above the toy budget, as expected.
Leiden in the metrics step dominates; it scales with cells × models × 5 resolutions.

The scale test's web export is not size-budgeted and is never written into the site. In-browser numbers
for it are under "Browser checks".

## Web export and dashboard

`pipeline/export.py` writes compact files to `integration/public/multiverse-data/`: 14 files,
**23.8 MB** in total (budget 25 MB; the fuzzy UpSet file adds 0.83 MB gzipped, 2026-09-28). Dtypes, shapes and orders are documented in the generated
[`README.md` there](../integration/public/multiverse-data/README.md). The browser loader
(`integration/src/components/multiverse/data/loader.ts`) validates every shape and byte count
against `manifest.json` and shows a readable error on mismatch.

The dashboard is a React island (`client:only="react"`) on `src/pages/multiverse.astro`. It uses
the site's existing `Layout.astro`, D3 v7 for all charts (canvas for the cell maps, SVG for the
rest), and React context for the shared selection. Loading is staged:

- **Entry views** (V0a, V0b, V3) load `manifest.json`, `models.json` and the agreement matrices
  (≈ 0.6 MB).
- **Cell-level files** load lazily when the *Cells & regions* tab opens (≈ 7.9 MB).
- **`z_vs_ref`** for the current measure mode (8.6 MB primary, 2.9 MB composition) loads only when a view needs calibrated change.

Everything that depends on a selection is computed in the browser:

- fraction of cells with |z| > 2 per model,
- lattice edge colours,
- region effects,
- the "relative to factor median" transform,
- neighbour provenance.

| view | what it shows |
|---|---|
| V0a model map | UMAP (precomputed A, n_neighbors=10), t-SNE (precomputed A, perplexity 15) or classical MDS of all models. The colour dropdown offers multiverse measures (consensus share, mean Δ, fraction \|z\|>2 vs the reference), decisions (every factor + seed) and every benchmark metric (overall / bio / batch, each metric raw and scaled). Seed replicates are joined by thin outlines. Hover, click, shift-click and lasso. |
| V0b leaderboard | Top 10 for the benchmark metric chosen in its own **Rank by** dropdown (aggregates, raw or scaled metrics): sortable table with seed sd and an "≈1" marker when within noise of rank 1, or a D3 funky heatmap (setup / overall / batch / bio; bars and circles) |
| Label mixing tab | comparative fuzzy UpSet: which labels mix in each model's kNN graph, and how that differs across models. See [Label mixing (fuzzy UpSet)](#label-mixing-fuzzy-upset). |
| Embeddings tab | the standard UMAP of **each** embedding (scanpy `pp.neighbors` on that model's latent + `tl.umap`), up to 4 side by side, coloured by sample / study / cell type / QC / any measure, with each model's iLISI, kBET, PCR, ARI and NMI. Defaults to the reference model and its counterparts with the other methods (same features and seed). Lasso selects cells everywhere. For checking integrations visually. |
| V1 fixed cell map | UMAP of the consensus kNN graph on canvas (hex-binned density from 50,000 cells), coloured by any obs, QC, measure, region or z; quadtree hover, lasso, pan/zoom, click-to-highlight legend |
| V2 stability map | all-model vs seed-only consensus stability side by side |
| V3 agreement matrix | A(a,b) in factor order with factor strips; click selects both models |
| V4 fingerprint | regions × factors (+ seed null): mean E as colour, mean H as ring size; optional "relative to factor median" display transform |
| V5 variant strip | regions × models (V3 order) coloured by local variant, filtered by the model selection |
| V6 design lattice | 48 configurations on a factor grid; one-factor edges coloured by mean Δ̄ over the current cell selection |
| V7 region inspector | composition vs rest, QC and doublet-score densities, top genes, neighbour provenance for two exported models |
| V8 decision record | chosen model + metrics with ranks and seed sd + evidence snapshot, downloaded as JSON / Markdown via Blob |

**Linking.** One shared state holds `selectedModels`, `selectedCells`, `selectedRegion`, the
colour keys, `layout`, `referenceModel` and the measure mode.

- Model selections from V0a, V0b, V3, V6 or V5 filter V5 and restrict V6 edges.
- Cell or region selections from V1, V4 or V5 highlight cells on V1 and drive V6, V7 and V0a's
  "fraction changed".
- "Reset selection" clears everything.

## Embedding-comparison framework

The integration embeddings can also be analysed with the embedding-stability framework in
`../embedding-comparison` (a separate git repository; changes are on its branch `integration-dataset`).
That framework compares a small ensemble of embeddings through:
- a **meta-map** of embedding × embedding dissimilarity;
- **consensus neighbours** (proposed by a quorum of embeddings);
- per-cell **magnitude** (1 − mean Jaccard to the consensus);
- **severity** (how far disputed neighbours sit, compared with consensus ones, in a neutral space no embedding sees);
- **agreement patterns**.

```bash
make export-ec                                   # ~10 s → ../embedding-comparison/public/data/integration (15 MB, gitignored there)
cd ../embedding-comparison && npm run dev         # open /?dataset=integration
npx tsx scripts/analyze-integration.ts 12 0.5 truth   # Markdown report; saved runs in reports/
```

Choices made for the export (`pipeline/export_ec.py`):
- **Cells:** a stratified subsample of 1,499 cells, because the app draws one SVG mark per cell.
- **Neighbours:** kNN in each model's **latent** space, recomputed within the subsample. Each view draws that model's own UMAP.
- **Ensembles:** four presets.
  - 3 methods × 2 batch keys
  - the reference setup × 3 seeds
  - Harmony × all feature choices
  - all 48 configurations
- **Severity spaces:**
  - **True biology**: the simulator's noise-free, batch-free profile, added to `toy.h5ad` as `obsm["X_truth"]` without changing any counts. It is simulation-only and has exact ties, so severity uses additive smoothing (dd + f)/(cc + f), with f = 5% of the median distance.
  - **Observed expression**: uncorrected. It is available on real data, but it rewards leaving batch effects in place.

## Label mixing (fuzzy UpSet)

Added 2026-09-28. The **Label mixing** tab shows which labels of one column (`cell_type`, `sample`,
`study` or `tissue`, one UpSet per column, never mixed) share neighbourhoods in each embedding, and
how that differs across embeddings.

**What is computed** (`pipeline/fuzzy_upset.py`; the browser re-implements the last steps in
`compute/fuzzyUpset.ts`, and a pytest parity test runs both on the same inputs):

1. **Memberships.** A = binarised, directed k = 15 kNN graph of the model's own latent space with
   self-loops, row-normalised. L = one-hot labels. P = A·L, so P[i, l] is the fraction of cell i's
   16-cell neighbourhood with label l. Unlabelled cells have no label but still count as neighbours.
2. **Signatures.** A cell belongs to every label with P ≥ τ. Its signature is that set of labels. More
   than *max size* labels puts it in the **diffuse** bucket; no label ≥ τ puts it in **none**. Both
   counts are shown.
3. **Fuzzy size.** A cell's strength is the smallest of its memberships over the signature. Per
   intersection: cells, fuzzy size (sum of strengths) and mean strength.
4. **Comparison.** Intersection × model table of fuzzy sizes (0 if absent). The difference score
   is max − min or variance across the compared models, or |log2 ratio| (pseudocount 1) for
   exactly two.
5. **Attributes** for the focused model: composition by a second column and mean `total_counts`,
   `pct_mito`, `doublet_score` (`n_genes` does not exist in this dataset).

**Reading the view.** Columns are intersections. The bar is the focused model's fuzzy size. The
strip below has one row per compared model (focused model first), and the dot matrix names the
labels. Sorting is by difference across models (default), fuzzy size or cells. Pure (single-label)
intersections are hidden by default. Clicking a column selects that intersection's cells in the
focused model. They are highlighted on the focused model's UMAP next to the chart and in the
Cells / Embeddings tabs. Clicking a strip cell focuses that model. A table view is under the chart.

**Parameters.**

| parameter | default | where |
|---|---|---|
| τ (threshold) | 0.1 raw, 1.0 in enrichment mode | browser slider |
| max labels per intersection | 3 | browser |
| enrichment normalisation | off | browser |
| compared / focused models | reference setup (seed 0) × 3 methods × 2 batch keys | browser |
| k | 15 | `fuzzy_upset.k` in config.yaml (needs re-export) |
| exported models | seed 0 of each configuration (48) | `fuzzy_upset.models` (`seed0` / `all` / list) |

In **enrichment mode**, τ applies to P ÷ global label frequency, so τ = 1 means "as common as in
the whole dataset". Rare labels can then enter a signature. Strengths and fuzzy sizes always use the
raw fractions (0–1).

**Interpretation.**
- For **cell-type labels**, mixing usually means lost biology: two populations the embedding no
  longer separates. On the toy data, T-A & T-B (planted P1) and Ciliated & Epithelial (P2) rank first.
- For **batch labels** (sample, study), mixing is usually what integration is supposed to achieve.
  A large {s1, s2, s3} intersection means those samples share neighbourhoods. If it is absent in
  one model, that model left a batch effect.
- **Compare across models, not in absolute terms.** The difference score is the point of the view.

**Known limitations.**
- **Threshold sensitivity.** With k = 15, memberships move in steps of 1/16 = 0.0625, so τ = 0.1
  means "at least 2 of 16". Small τ changes do nothing, and crossing a step can move many cells at
  once.
- **Label imbalance.** Large labels dominate fuzzy sizes. Enrichment mode helps rare labels pass τ
  but does not rescale sizes.
- **Natural mixing at boundaries.** Related cell states (subtypes, differentiation trajectories)
  mix at their borders in any good embedding. Treat a mixed intersection as a pointer to inspect,
  not a verdict.
- **Directed graph.** Asymmetric neighbourhoods are kept as is. A cell can list neighbours that
  do not list it back.
- **Not exported: 96 models.** Only the seed-0 replicates are in the browser, to stay within the
  budget. The Python CLI covers all models.
- The browser file stores exact neighbour counts, so changing k needs a re-export.

**Offline / large scale.**

```bash
python -m pipeline.fuzzy_upset --label cell_type [--tau 0.1 --max-size 3 --normalize --models all \
       --how log_ratio --pair A,B --show-pure]      # tables in outputs/<name>/fuzzy_upset/
make export-fuzzy                                   # refresh only the browser file + manifest
python -m pipeline.fuzzy_upset_scale                # 1M cells × 20 models scale check
```

Per-model signatures are cached on disk, keyed by model, label column, k, τ, max size,
normalisation and the kNN file's timestamp. The browser memoises the same key. Scale check
(Apple M1, 2026-09-28): 1,000,000 cells × 20 models, k = 15, 20 labels, 1% unlabelled.

| step | time |
|---|---|
| memberships | 10.5 s |
| signatures | 29.9 s |
| tables and ranking | 0.6 s |

Peak RSS was 0.71 GB. Graphs are processed one model at a time; no dense cell × cell matrix is
built. On the toy data, the CLI takes 2.7 s for 48 models with a peak RSS of 0.2 GB.

## Deploy

The site is fully static: no adapter, no API routes. The dashboard fetches only
`<base>/multiverse-data/*` via `import.meta.env.BASE_URL`.

**Data files are committed.** `integration/public/multiverse-data/` (20 MB) is in git, so the site
builds anywhere without Python. The heavy intermediates (`multiverse-pipeline/outputs/`,
`data/*.h5ad`) are gitignored.

To regenerate the data and rebuild:

```bash
cd multiverse-pipeline && make data run test     # rewrites ../integration/public/multiverse-data/
cd ../integration && npm ci && npm run build     # static site in integration/dist/
npm run preview                                  # check locally
```

The deploy target is not decided yet. For a subpath host such as GitHub Pages
(`https://<user>.github.io/embedding_comparison_scAtlas/`), build with:

```bash
ASTRO_SITE=https://<user>.github.io ASTRO_BASE=/embedding_comparison_scAtlas npm run build
```

`astro.config.mjs` reads both variables; unset means root hosting. There is no CI yet. When a
host is chosen, a workflow only needs `npm ci && npm run build` in `integration/` (with the two
variables for a subpath) and should upload `integration/dist/`. The pipeline does not need to
run in CI because the data is committed.

## Browser checks

`browser-checks/check.mjs` (Playwright, 51 checks since 2026-09-28) covers:

- no console errors,
- every view renders,
- the dropdown recolours V0a and updates V0b, including the lazy frac-|z| colour,
- funky heatmap,
- linking: leaderboard click, V0a lasso, V3 click, V1 hover and lasso → V6/V7, V4 region click,
  V6 node → V5 filter, fuzzy UpSet column click → cell selection,
- fuzzy UpSet: renders, ranks a planted mixing first, τ change, label-column switch,
- measure-mode switch, reset,
- V8 JSON and Markdown downloads,
- interaction timings.

Timings are measured end to end from Playwright, including its overhead.
Results from 2026-09-27 are in `browser-checks/results-2026-09-27.txt`:

| target | browser | result | recolour | model lasso | cell recolour | cell lasso + linked recompute |
|---|---|---|---|---|---|---|
| `astro build` + `astro preview`, base `/` | Chrome (installed) | 36/36 | 32 ms | 28 ms | 30 ms | 39 ms |
| same | Firefox (Playwright) | 36/36 | 47 ms | 40 ms | 28 ms | 41 ms |
| same | WebKit (Playwright, Safari engine) | 36/36 | 22 ms | 40 ms | 17 ms | 34 ms |
| same, dark mode | Chrome | 36/36 | 16 ms | 22 ms | 18 ms | 30 ms |
| same, dark mode | Firefox | 36/36 | 58 ms | 42 ms | 27 ms | 42 ms |
| build with `ASTRO_BASE=/embedding_comparison_scAtlas` + preview | Chrome | 36/36 | – | – | – | – |
| `astro dev` | Chrome | 36/36 | – | – | – | – |

All interactions are under the 200 ms target. Entry data loads in about 20 ms and cell data in
about 60 ms from localhost; network time on a real host will dominate. Under the subpath build,
`/multiverse-data/manifest.json` at the root returns 404, which confirms that every fetch goes
through `BASE_URL`. The existing `/` page still builds and serves (HTTP 200).

Bugs found and fixed by these checks:

- Hover tooltips re-rendered and rebuilt the V4 and V5 SVGs between mouse-down and mouse-up,
  swallowing clicks. Real users would hit this too; Chrome only passed because automated clicks
  are fast.
- The lasso behaviour is now bound once and reads positions through refs.

Scale test in the browser (Chrome, the built site with the 50k-cell export swapped in, served locally, no console errors):

| step | time |
|---|---|
| navigation → V0a drawn | 919 ms |
| model recolour + leaderboard | 27 ms |
| cells tab: fetch the 19.8 MB cell bundle, parse, first render of V1–V7 | 899 ms |
| cell map recolour (hex-binned density, used from 50,000 cells) | 29 ms |
| cell lasso → selection + linked recomputes (14,172 cells) | 93 ms |

Command: `node browser-checks/scale_check.mjs <url>`.

## Decisions and assumptions

Modelling choices made while implementing the brief. Runtime fallbacks are also appended to
`outputs/<name>/decisions_log.txt`.

**Simulator**

1. Poisson-gamma (negative binomial) counts, dispersion 0.3, log-normal library sizes (mean 3,000).
2. There are 7 labels: T-A, T-B, B, Myeloid, NK, Epithelial and Ciliated.
   - T-A and T-B share the T marker block and differ by 15 weak markers (P1).
   - Ciliated is the epithelium of tissue 2 (samples s5 and s6 of study B only): the epithelial
     block plus 30 state genes (P2).
   - 360 secondary markers act at lineage level, so the weak P1 markers are not top HVGs.
3. **P1 strength was calibrated** before any measure existed: `p1_log_fc` 0.6 → 1.2. The design
   criterion was markers mostly outside the top 500 HVGs but T-A/T-B separable at 1,500
   (kNN purity 0.51 vs 0.80).
4. P2 is built so that there is no within-sample contrast in tissue-2 samples. Every epithelial
   cell in s5/s6 is Ciliated, while study B also contains tissue-1 epithelium (s4). Correcting by
   sample removes the state (Ciliated kNN purity 0.98 → 0.70 under harmony); correcting by study
   keeps it.
5. P3 is 150 cells, 80% from s2: library × 0.2, marker log-fold changes × 0.3, mito × e^1.8.
6. P4 is 200 cells formed by adding a second cell of a different type from the same sample.
   `doublet_score` ~ Beta(6,3) for doublets and Beta(2,8) for singlets.
7. P5 is 30 `IGX_` genes. Each B cell expresses one of three random modules, with an extra
   per-sample log-normal effect (sd 0.8).
8. The interferon-like program is on in 30% of s1 cells and 10% of other cells.
9. `ground_truth.csv` stores boolean membership per effect (effects can overlap) plus one primary
   label, with priority doublet > low-QC > P2 > P1 > P5.

**Integration and kNN**

10. **Seed variation.** For near-deterministic methods, PCA is *fit* on a seed-specific random
    90% subsample and then applied to all cells. Harmony also gets `random_state=seed`. ComBat is
    deterministic, so its variation comes from the PCA subsample.
11. The subsample depends only on the seed index, so seed *i* uses the same cells in every
    configuration. Matched contrasts therefore average only over **independent** reruns
    (pairings i ≠ j, 6 per pair).
12. With batch-unaware HVGs, `batch_key` has no effect on `pca`. Those embeddings are identical,
    and these real zero-effect contrasts stay in the design. Their kNN graphs are computed once
    (detected by hash).
13. HVGs use scanpy's `seurat` flavour on log-normalised data. Batch-aware selection ranks by the
    number of batches in which a gene is highly variable, then by mean normalised dispersion.
    `exclude_igx` drops IGX genes before selection.
14. `sc.pp.scale(max_value=10)`. `combat_pca` runs ComBat on log-normalised HVGs, then scales,
    then fits PCA with 30 components.
15. kNN: pynndescent (k=50, fixed `random_state`, recall@15 vs exact = 1.0 on a spot check). The
    k=15 graph is the first 15 columns of that list.
16. `scvi` and `scib-metrics` are behind config flags, off by default, and fall back with a
    logged message. They were **not** installed or exercised in this run.

**Benchmark metrics**

17. Batch metrics are always evaluated against **`sample`**, the finest batch variable, so they
    are comparable across models. This favours `batch_key=sample` models: all top-10 overall
    models use it.
18. The kBET-like score is a per-cell chi-square test of the k=50 batch composition against the
    composition expected for that cell's label; the score is the acceptance rate at α = 0.05.
19. Graph iLISI is simplified: inverse Simpson index on the plain k=50 neighbourhood, not
    shortest-path LISI.
20. PCR comparison is measured against the uncorrected `pca` model with the same features and
    seed, clipped at 0.
21. Leiden runs on the unweighted, symmetrised k=15 graph. ARI and NMI are each the best over
    resolutions 0.2–1.0.
22. Moran's I uses binary directed kNN weights. Gene-set scores are the mean of z-scored log
    expression.
23. Per the brief, the Moran's I of *all* gene-set scores (including `IGX`) and of covariates
    counts as bio conservation. So a model with spurious IGX structure gains bio score, which
    reproduces the paper's concern.
24. Paper-style graph dissimilarity: per matched pair, using seeds (0, 1), the per-cell mean of
    |d(a→b)| and |d(b→a)|, averaged over pairs.
25. Majority vote uses each model's best-ARI Leiden clustering.

**Multiverse measures**

26. μ and σ² for a model pair are the means of the two configurations' per-cell seed-Δ means and
    variances. They are smoothed over the consensus kNN graph (each cell plus its 15 neighbours),
    with ε = 0.05.
27. z uses the σ of a single rerun, while Δ̄ averages 6 pairings. So E is conservative (it lies in
    units of single-rerun noise).
28. Rank-based z is Φ⁻¹ of the mid-rank percentile of Δ̄ among the pooled seed Δs of both
    configurations over the cell's consensus neighbourhood.
29. The seed pseudo-factor is a **leave-one-out** null. Each seed pair of a configuration is scored
    against the configuration's other seed pairs. Including the pair itself would make the null
    trivially 0.
30. The consensus kNN graph takes the k most frequent neighbours across all models (ties by id).
    Consensus stability uses the brief's threshold, ≥ 80% of models.
31. Agreement A(a,b) is computed on 2,000 cells, stratified by cell type.
32. Regions: Leiden on the consensus graph with edge weights exp(−‖Δprofile‖² / median) on the
    standardised E profile (seed excluded). The first resolution giving 8–20 regions is used, and
    regions under 25 cells are merged into their best-connected neighbour. If no resolution gives
    8–20, the resolution closest to that range is used. This fallback was fixed after the scale
    test exposed a bug; the toy run always landed in range.
33. Variants use Leiden (resolution 0.5) on the region's induced subgraph in each model, counting
    sub-clusters ≥ 10% of the region. Rules: `joined` if ≥ 50% of neighbours lie outside the
    region, else `split` / `merged` relative to the modal count across models, else `typical`.
34. The **neighbour-composition** mode is an exploratory post-hoc variant (see VALIDATION.md).
    Neighbours are mapped to about N/(5k) label-free Leiden micro-clusters of the consensus graph,
    and Δ = 1 − overlap of the two compositions. Everything else is identical.
35. Reference models are the top overall, best bio and best batch models. `z_vs_ref` is exported
    for all three in the primary mode and only for the top-overall reference in composition mode,
    to stay within the size budget.

**Layouts and dashboard**

36. The model map uses UMAP (precomputed A, n_neighbors = 10, random init, fixed seed), t-SNE
    (precomputed A, perplexity 15, random init, fixed seed) or classical MDS. The cell map is a UMAP of the binary symmetrised consensus graph via `scanpy.tl.umap`.
    Both are used only for location; no measure reads 2D coordinates.
37. Colours use the validated reference palette (light and dark steps).
    - Factors have ≤ 3 levels, within the 3-slot all-pairs-safe set.
    - Cell types (7) and samples (6) exceed it on the scatter. The mitigation is a
      click-to-highlight legend plus hover tooltips.
    - Regions (13–16) are **not** colour-coded categorically: they are coloured by total |E|
      and labelled directly.
38. V4's "relative to factor median" is a display transform only and has not been validated.
39. Per-embedding UMAPs (added 2026-09-28 at the user's request, for checking each integration visually):
    scanpy `pp.neighbors(use_rep=latent, n_neighbors=15)` + `tl.umap`, fixed seed. Identical embeddings are
    computed once. They are exported as uint16 (per-model min/max in the manifest, precision ≈ range/65535)
    to stay under the size budget. Disabled in the scale config (about 1 min per model at 50k cells).

## Known limitations

- See VALIDATION.md. Neighbour-identity change saturates inside homogeneous populations, so P1
  and P2 are not localised. Consensus stability at the 80% threshold is about 0 everywhere.
- The paper's BRAS metric and scVI were not run (optional dependencies not installed).
- `pynndescent` dominates runtime. An exact kNN (`knn.backend: exact`) would be faster at 5k cells
  but deviates from the brief.
