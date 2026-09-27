// Holds loaded data; entry files are loaded up front, per-cell files lazily on request.
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react';
import { type CellData, type EntryData, loadCells, loadZ } from './loader';
import { useSelection } from '../state';
import { allCells, fracChanged } from '../compute/selection';

type Status = 'idle' | 'loading' | 'ready' | 'error';

interface DataCtx {
  entry: EntryData;
  cells: CellData | null;
  cellsStatus: Status;
  cellsError: string | null;
  requestCells: () => void;
  z: Record<string, { z: Float32Array; refs: string[] }>;
  zStatus: Record<string, Status>;
  zError: string | null;
  requestZ: (mode: string) => void;
  /** fraction |z|>2 vs the reference, per model, over the selected (or all) cells; null if unavailable */
  fracZ: Float32Array | null;
  fracZNote: string | null;
  activeCells: Uint32Array;
}

const Ctx = createContext<DataCtx | null>(null);

export function DataProvider({ entry, children }: { entry: EntryData; children: ReactNode }) {
  const { state } = useSelection();
  const [cells, setCells] = useState<CellData | null>(null);
  const [cellsStatus, setCellsStatus] = useState<Status>('idle');
  const [cellsError, setCellsError] = useState<string | null>(null);
  const [z, setZ] = useState<Record<string, { z: Float32Array; refs: string[] }>>({});
  const [zStatus, setZStatus] = useState<Record<string, Status>>({});
  const [zError, setZError] = useState<string | null>(null);
  const inflight = useRef(new Set<string>());
  const { manifest } = entry;
  const N = manifest.dataset.n_cells;
  const M = manifest.dataset.n_models;

  const requestCells = useCallback(() => {
    if (inflight.current.has('cells')) return;
    inflight.current.add('cells');
    setCellsStatus('loading');
    const t0 = performance.now();
    loadCells(manifest)
      .then((cd) => {
        setCells(cd);
        setCellsStatus('ready');
        console.info(`[multiverse] cell-level data loaded in ${(performance.now() - t0).toFixed(0)} ms`);
      })
      .catch((e: Error) => {
        inflight.current.delete('cells');
        setCellsError(e.message);
        setCellsStatus('error');
      });
  }, [manifest]);

  const requestZ = useCallback(
    (mode: string) => {
      const key = `z:${mode}`;
      if (inflight.current.has(key)) return;
      inflight.current.add(key);
      setZStatus((s) => ({ ...s, [mode]: 'loading' }));
      loadZ(manifest, mode)
        .then((d) => {
          setZ((s) => ({ ...s, [mode]: d }));
          setZStatus((s) => ({ ...s, [mode]: 'ready' }));
        })
        .catch((e: Error) => {
          inflight.current.delete(key);
          setZError(e.message);
          setZStatus((s) => ({ ...s, [mode]: 'error' }));
        });
    },
    [manifest],
  );

  const all = useMemo(() => allCells(N), [N]);
  const activeCells = state.selectedCells ?? all;

  const { fracZ, fracZNote } = useMemo(() => {
    const d = z[state.mode];
    if (!d) return { fracZ: null, fracZNote: null };
    const r = d.refs.indexOf(state.referenceModel);
    if (r < 0) return { fracZ: null, fracZNote: `z vs ${state.referenceModel} was not exported for this measure mode (available: ${d.refs.join(', ')}).` };
    return { fracZ: fracChanged(d.z, r, N, M, activeCells), fracZNote: null };
  }, [z, state.mode, state.referenceModel, activeCells, N, M]);

  const value: DataCtx = { entry, cells, cellsStatus, cellsError, requestCells, z, zStatus, zError, requestZ, fracZ, fracZNote, activeCells };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useData() {
  const v = useContext(Ctx);
  if (!v) throw new Error('useData outside DataProvider');
  return v;
}
