// Embedding viewer: the standard UMAP of each embedding (scanpy neighbors on that model's latent
// + tl.umap), side by side, for visually checking each integration (batch mixing, cell-type
// separation). Visual QC only — no multiverse measure is computed from these coordinates.
import { useEffect, useMemo, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { fmt } from '../d3/colors';
import { loadModelUmaps, modelUmap, type ModelUmaps } from '../data/loader';
import { cellColorOptions } from './cellColor';
import CellMap from './CellMap';

const N_SLOTS = 4;
const METRICS: [string, string][] = [
  ['graph_iLISI', 'iLISI'],
  ['kBET_like', 'kBET'],
  ['PCR_comparison', 'PCR'],
  ['ARI', 'ARI'],
  ['NMI', 'NMI'],
];
let cache: Promise<ModelUmaps | null> | null = null;

export default function EmbeddingViewer() {
  const { entry, cells, cellsStatus, requestCells } = useData();
  const { manifest, models } = entry;
  const { state } = useSelection();
  const [umaps, setUmaps] = useState<ModelUmaps | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [colorKey, setColorKey] = useState('cat:sample');

  useEffect(() => {
    if (cellsStatus === 'idle') requestCells();
    cache ??= loadModelUmaps(manifest);
    cache.then(setUmaps).catch((e: Error) => {
      cache = null;
      setError(e.message);
    });
  }, [cellsStatus, requestCells, manifest]);

  // default panels: the reference model and its counterparts with the other methods (same features, seed)
  const defaults = useMemo(() => {
    const ref = state.referenceModel;
    const rest = ref.slice(ref.indexOf('.'));
    const methods = manifest.factors.find((f) => f.name === 'method')!.levels.map(String);
    const ids = [ref, ...methods.map((m) => `${m}${rest}`).filter((id) => id !== ref)];
    return ids.filter((id) => models.some((m) => m.model_id === id)).slice(0, N_SLOTS);
  }, [state.referenceModel, manifest, models]);
  const [slots, setSlots] = useState<string[]>(defaults);
  useEffect(() => setSlots(defaults), [defaults]);

  if (error) return <div className="mv-error" role="alert">Could not load per-model UMAPs: {error}</div>;
  if (umaps === null)
    return <p className="mv-muted">This export has no per-model UMAPs (set <code>per_model_umap.enabled: true</code> in config.yaml and re-run).</p>;
  if (!umaps || !cells) return <p className="mv-loading" role="status">Loading per-embedding UMAPs (≈ 3 MB) and cell annotations…</p>;

  const opts = cellColorOptions(manifest, cells, state.mode);
  const groups = Array.from(new Set(opts.map((o) => o.group)));
  const selected = [...state.selectedModels];

  return (
    <div className="mv-card">
      <h2>Embeddings · UMAP of each integration</h2>
      <p className="mv-small mv-muted">
        Standard per-embedding UMAP ({umaps.recipe}). Use it to check each integration visually: colour by <em>sample</em> or <em>study</em> to see batch
        mixing, by <em>cell_type</em> to see whether populations stay separated. These maps are for inspection only; no measure is computed from them.
        Lasso on any panel selects cells everywhere.
      </p>
      <div className="mv-inline" style={{ marginBottom: 10 }}>
        <label>
          Colour cells by
          <select value={colorKey} onChange={(e) => setColorKey(e.target.value)}>
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
        <button type="button" className="mv-button" disabled={!selected.length} onClick={() => setSlots(selected.slice(0, N_SLOTS))}>
          Show selected models{selected.length ? ` (${Math.min(selected.length, N_SLOTS)})` : ''}
        </button>
        <button type="button" className="mv-button" onClick={() => setSlots(defaults)}>
          Reference + other methods
        </button>
      </div>
      <div className="mv-grid">
        {Array.from({ length: N_SLOTS }, (_, i) => (
          <EmbeddingPanel
            key={i}
            slot={i}
            modelId={slots[i] ?? ''}
            onChange={(id) =>
              setSlots((s) => {
                const next = Array.from({ length: N_SLOTS }, (_, j) => s[j] ?? '');
                next[i] = id;
                return next;
              })
            }
            colorKey={colorKey}
            umaps={umaps}
          />
        ))}
      </div>
    </div>
  );
}

function EmbeddingPanel({
  slot,
  modelId,
  onChange,
  colorKey,
  umaps,
}: {
  slot: number;
  modelId: string;
  onChange: (id: string) => void;
  colorKey: string;
  umaps: ModelUmaps;
}) {
  const { entry } = useData();
  const { manifest, models } = entry;
  const N = manifest.dataset.n_cells;
  const mi = models.findIndex((m) => m.model_id === modelId);
  const coords = useMemo(() => (mi >= 0 ? modelUmap(umaps, mi, N) : null), [umaps, mi, N]);
  const m = mi >= 0 ? models[mi] : null;
  return (
    <section className="mv-embed-panel" aria-label={`Embedding panel ${slot + 1}`}>
      <label className="mv-small">
        Model {slot + 1}
        <select value={modelId} onChange={(e) => onChange(e.target.value)} aria-label={`Model for panel ${slot + 1}`}>
          <option value="">(none)</option>
          {models.map((x) => (
            <option key={x.model_id} value={x.model_id}>
              {x.model_id}
            </option>
          ))}
        </select>
      </label>
      {m && coords ? (
        <>
          <p className="mv-small mv-muted" style={{ margin: '4px 0' }}>
            {Object.entries(m.factors)
              .map(([k, v]) => `${k}=${v}`)
              .join(' · ')}{' '}
            · seed {m.seed}
          </p>
          <CellMap colorKey={colorKey} height={340} coords={coords} ariaLabel={`UMAP of embedding ${m.model_id}`} />
          <p className="mv-small" style={{ margin: '4px 0 0' }}>
            {METRICS.map(([k, label]) => (
              <span key={k} style={{ marginRight: 10 }}>
                {label} <strong>{fmt(m.raw[k], 2)}</strong>
              </span>
            ))}
            <span className="mv-muted">batch metrics vs sample</span>
          </p>
        </>
      ) : (
        <p className="mv-muted mv-small">Pick a model.</p>
      )}
    </section>
  );
}
