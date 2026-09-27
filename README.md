# embedding_comparison_scAtlas — Embedding Multiverse Explorer (toy prototype)

This repository asks **which analysis decisions reshape which cells**. It treats the set of
single-cell integrations produced by a factorial decision space as a *multiverse*, and runs
end to end on a synthetic dataset with planted ground truth.

| folder | what | runs |
|---|---|---|
| [`multiverse-pipeline/`](multiverse-pipeline/) | Python pipeline: simulator, 144 integrations, kNN, benchmark metrics, multiverse measures, regions, export, tests | offline (`make all`) |
| [`integration/`](integration/) | Astro site; the dashboard is the React/D3 island at `/multiverse` | `npm run dev` / `npm run build` (static) |

- Start here: [multiverse-pipeline/README.md](multiverse-pipeline/README.md), which covers setup,
  the pipeline, the dashboard, deployment, and the decisions and assumptions made.
- Honest results: [multiverse-pipeline/VALIDATION.md](multiverse-pipeline/VALIDATION.md). Some
  planted effects are **not** recovered, and that report explains why.
- Data format: [integration/public/multiverse-data/README.md](integration/public/multiverse-data/README.md).
