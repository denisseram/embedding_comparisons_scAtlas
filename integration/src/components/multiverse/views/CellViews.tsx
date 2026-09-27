// Cell-level tab: lazily loads per-cell arrays, then shows V1, V2, V4, V5, V6, V7.
import { useEffect, useMemo, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { regionEffects } from '../compute/selection';
import { cellColorOptions } from './cellColor';
import CellMap from './CellMap';
import Fingerprint from './Fingerprint';
import VariantStrip from './VariantStrip';
import DesignLattice from './DesignLattice';
import RegionInspector from './RegionInspector';

export default function CellViews() {
  const { entry, cells, cellsStatus, cellsError, requestCells, z, zStatus, requestZ } = useData();
  const { manifest, models } = entry;
  const { state, dispatch } = useSelection();
  const [relative, setRelative] = useState(false);

  useEffect(() => {
    if (cellsStatus === 'idle') requestCells();
  }, [cellsStatus, requestCells]);
  useEffect(() => {
    if (state.cellColorBy === 'zsel' && !zStatus[state.mode]) requestZ(state.mode);
  }, [state.cellColorBy, state.mode, zStatus, requestZ]);

  const effects = useMemo(() => (cells ? regionEffects(cells, manifest, state.mode, relative) : []), [cells, manifest, state.mode, relative]);
  const effectsAbs = useMemo(() => (cells ? regionEffects(cells, manifest, state.mode, false) : []), [cells, manifest, state.mode]);

  // z(c, reference, selected model) when exactly one model is selected
  const { zSel, zLabel } = useMemo(() => {
    const d = z[state.mode];
    if (!d || state.selectedModels.size !== 1) return { zSel: null, zLabel: undefined };
    const r = d.refs.indexOf(state.referenceModel);
    const id = [...state.selectedModels][0];
    const m = models.findIndex((x) => x.model_id === id);
    if (r < 0 || m < 0) return { zSel: null, zLabel: undefined };
    const N = manifest.dataset.n_cells;
    const M = manifest.dataset.n_models;
    const out = new Float32Array(N);
    for (let c = 0; c < N; c++) out[c] = d.z[(r * N + c) * M + m];
    return { zSel: out, zLabel: `z(c, ${state.referenceModel} → ${id})` };
  }, [z, state.mode, state.selectedModels, state.referenceModel, models, manifest]);

  if (cellsStatus === 'error')
    return (
      <div className="mv-error" role="alert">
        <strong>Could not load cell-level data.</strong> {cellsError}{' '}
        <button type="button" className="mv-button" onClick={requestCells}>
          Retry
        </button>
      </div>
    );
  if (!cells) return <p className="mv-loading" role="status">Loading cell-level data (≈ 5 MB)…</p>;

  const opts = cellColorOptions(manifest, cells, state.mode);
  const groups = Array.from(new Set(opts.map((o) => o.group)));
  return (
    <div className="mv-grid">
      <section className="mv-card mv-span-2" aria-labelledby="v1">
        <h2 id="v1">V1 · Fixed cell map</h2>
        <div className="mv-inline">
          <label>
            Colour cells by
            <select value={state.cellColorBy} onChange={(e) => dispatch({ type: 'set', key: 'cellColorBy', value: e.target.value })}>
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
          {state.selectedCells && (
            <button type="button" className="mv-button" onClick={() => dispatch({ type: 'selectCells', cells: null })}>
              Clear cell selection
            </button>
          )}
        </div>
        <CellMap
          colorKey={state.cellColorBy}
          regionEffects={effectsAbs}
          zSel={zSel}
          zLabel={zLabel}
          title="UMAP of the consensus kNN graph — a fixed map for location only, never used as evidence."
        />
      </section>
      <section className="mv-card mv-span-2" aria-labelledby="v2">
        <h2 id="v2">V2 · Stability map</h2>
        <p className="mv-small mv-muted">
          Share of each cell's consensus neighbour slots filled by neighbours present in ≥80% of the kNN sets — across all models (left) and across seed
          replicates only (right). The gap between the two is change caused by decisions rather than reruns.
        </p>
        <div className="mv-grid">
          <CellMap colorKey="measure:stability_all" height={320} compact />
          <CellMap colorKey="measure:stability_seed" height={320} compact />
        </div>
      </section>
      <section className="mv-card" aria-labelledby="v4">
        <h2 id="v4">V4 · Regions × decisions fingerprint</h2>
        <Fingerprint effects={effects} relative={relative} setRelative={setRelative} />
      </section>
      <section className="mv-card" aria-labelledby="v7">
        <h2 id="v7">V7 · Region inspector</h2>
        <RegionInspector />
      </section>
      <section className="mv-card mv-span-2" aria-labelledby="v5">
        <h2 id="v5">V5 · Local variant strip</h2>
        <VariantStrip effects={effectsAbs} />
      </section>
      <section className="mv-card mv-span-2" aria-labelledby="v6">
        <h2 id="v6">V6 · Design lattice</h2>
        <DesignLattice />
      </section>
    </div>
  );
}
