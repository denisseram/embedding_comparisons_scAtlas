// React island root: loads entry data, owns the shared selection state, lays out the views.
import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { loadEntry, type EntryData } from './data/loader';
import { DataProvider, useData } from './data/DataContext';
import { SelectionProvider, useSelection, type SelectionState } from './state';
import { colorOptions } from './views/modelColor';
import ModelMap from './views/ModelMap';
import Leaderboard from './views/Leaderboard';
import './multiverse.css';

const CellViews = lazy(() => import('./views/CellViews'));
const AgreementMatrix = lazy(() => import('./views/AgreementMatrix'));
const DecisionRecord = lazy(() => import('./views/DecisionRecord'));

export default function MultiverseApp() {
  const [entry, setEntry] = useState<EntryData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const t0 = performance.now();
    loadEntry()
      .then((d) => {
        setEntry(d);
        console.info(`[multiverse] entry data loaded in ${(performance.now() - t0).toFixed(0)} ms`);
      })
      .catch((e: Error) => setError(e.message));
  }, []);
  if (error)
    return (
      <div className="mv-root">
        <div className="mv-error" role="alert">
          <strong>Could not load the multiverse data.</strong> {error}
        </div>
      </div>
    );
  if (!entry)
    return (
      <div className="mv-root">
        <p className="mv-loading" role="status">Loading model overview…</p>
      </div>
    );
  const m = entry.manifest;
  const initial: SelectionState = {
    selectedModels: new Set(),
    selectedCells: null,
    selectedRegion: null,
    colorBy: 'factor:method',
    cellColorBy: 'cat:planted_effect',
    layout: 'umap',
    referenceModel: m.references[0],
    mode: m.measure.primary_mode,
  };
  return (
    <SelectionProvider initial={initial}>
      <DataProvider entry={entry}>
        <Shell />
      </DataProvider>
    </SelectionProvider>
  );
}

type Tab = 'models' | 'cells' | 'record';

function Shell() {
  const { entry, fracZNote, zError, requestZ, zStatus } = useData();
  const { manifest, models } = entry;
  const { state, dispatch } = useSelection();
  const [tab, setTab] = useState<Tab>('models');
  const opts = useMemo(() => colorOptions(manifest), [manifest]);
  const groups = useMemo(() => Array.from(new Set(opts.map((o) => o.group))), [opts]);

  // the "fraction changed" colour needs z_vs_ref (lazy, ~9 MB)
  useEffect(() => {
    if (state.colorBy === 'mv:frac_z' && !zStatus[state.mode]) requestZ(state.mode);
  }, [state.colorBy, state.mode, zStatus, requestZ]);

  const nSel = state.selectedModels.size;
  const nCells = state.selectedCells?.length ?? 0;

  return (
    <div className="mv-root">
      <header className="mv-header">
        <div>
          <h1>Embedding Multiverse Explorer</h1>
          <p className="mv-muted">
            Which decisions reshape which cells? {manifest.dataset.n_models} embeddings ({manifest.dataset.n_configs} configurations ×{' '}
            {manifest.dataset.seeds.length} seeds) of {manifest.dataset.n_cells.toLocaleString()} synthetic cells with planted ground truth. All
            measures use each embedding's own kNN graph; 2D maps show location only.
          </p>
        </div>
      </header>

      <div className="mv-controls" role="toolbar" aria-label="Global controls">
        <label>
          Measure
          <select value={state.mode} onChange={(e) => dispatch({ type: 'set', key: 'mode', value: e.target.value })}>
            {manifest.measure.modes.map((mo) => (
              <option key={mo} value={mo}>
                {manifest.measure.mode_labels[mo] ?? mo}
              </option>
            ))}
          </select>
        </label>
        <label>
          Colour models by
          <select value={state.colorBy} onChange={(e) => dispatch({ type: 'set', key: 'colorBy', value: e.target.value })}>
            {groups.map((g) => (
              <optgroup key={g} label={g}>
                {opts
                  .filter((o) => o.group === g)
                  .map((o) => (
                    <option key={o.key} value={o.key}>
                      {o.label}
                    </option>
                  ))}
              </optgroup>
            ))}
          </select>
        </label>
        <label>
          Layout
          <select value={state.layout} onChange={(e) => dispatch({ type: 'set', key: 'layout', value: e.target.value })}>
            <option value="umap">UMAP (precomputed A, n_neighbors=10)</option>
            <option value="tsne">t-SNE (precomputed A, perplexity 15)</option>
            <option value="mds">Classical MDS</option>
          </select>
        </label>
        <label>
          Reference model
          <select value={state.referenceModel} onChange={(e) => dispatch({ type: 'set', key: 'referenceModel', value: e.target.value })}>
            {manifest.references.map((r, i) => (
              <option key={r} value={r}>
                {r} {i === 0 ? '(top overall)' : i === 1 ? '(best bio)' : i === 2 ? '(best batch)' : ''}
              </option>
            ))}
          </select>
        </label>
        <div className="mv-selection-summary" aria-live="polite">
          {nSel} model{nSel === 1 ? '' : 's'} · {nCells ? `${nCells.toLocaleString()} cells` : 'all cells'}
          {state.selectedRegion !== null ? ` · region R${state.selectedRegion}` : ''}
        </div>
        <button type="button" className="mv-button" onClick={() => dispatch({ type: 'reset' })}>
          Reset selection
        </button>
      </div>
      {(fracZNote || zError) && state.colorBy === 'mv:frac_z' && <p className="mv-warning">{zError ?? fracZNote}</p>}
      {state.mode !== manifest.measure.primary_mode && (
        <p className="mv-warning">
          Exploratory measure: neighbour composition was added after the pre-registered neighbour-identity measure failed validation checks. See
          VALIDATION.md.
        </p>
      )}
      <details className="mv-explainer">
        <summary>What does “Measure” mean?</summary>
        <p>
          Every multiverse view compares two models cell by cell: for a cell, take its 15 nearest neighbours in each model's own latent space and ask how
          different they are. <strong>Neighbour identity</strong> (the pre-registered measure) uses Δ = 1 − Jaccard of the two neighbour sets: 0 when the
          cell has exactly the same neighbours, 1 when it shares none. <strong>Neighbour composition</strong> (exploratory) first maps every neighbour to
          one of ~66 small clusters of the consensus graph and compares <em>which clusters</em> the neighbours come from, so reshuffling among
          equivalent cells of the same population no longer counts as change. The choice changes Δ, and with it the agreement matrix, the model
          map layout, E/H, regions, variants and the lattice.
        </p>
      </details>

      <nav className="mv-tabs" role="tablist" aria-label="Views">
        {(
          [
            ['models', 'Models (V0a, V0b, V3)'],
            ['cells', 'Cells & regions (V1–V7)'],
            ['record', 'Decision record (V8)'],
          ] as [Tab, string][]
        ).map(([t, label]) => (
          <button key={t} type="button" role="tab" aria-selected={tab === t} className={tab === t ? 'is-active' : ''} onClick={() => setTab(t)}>
            {label}
          </button>
        ))}
      </nav>

      <Suspense fallback={<p className="mv-loading">Loading view…</p>}>
        {tab === 'models' && (
          <div className="mv-grid">
            <section className="mv-card mv-span-2" aria-labelledby="v0a">
              <h2 id="v0a">V0a · Model map</h2>
              <ModelMap />
            </section>
            <section className="mv-card mv-span-2" aria-labelledby="v0b">
              <h2 id="v0b">V0b · Leaderboard</h2>
              <Leaderboard />
            </section>
            <section className="mv-card mv-span-2" aria-labelledby="v3">
              <h2 id="v3">V3 · Model agreement matrix</h2>
              <AgreementMatrix />
            </section>
          </div>
        )}
        {tab === 'cells' && <CellViews />}
        {tab === 'record' && <DecisionRecord />}
      </Suspense>
      <footer className="mv-muted mv-small">
        Data schema {manifest.schema_version}, generated {manifest.generated_at}. {models.length} models loaded.
      </footer>
    </div>
  );
}
