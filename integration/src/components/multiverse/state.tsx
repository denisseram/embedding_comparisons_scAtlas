// Shared selection state for all linked views (React context + reducer; the site has no store library).
import { createContext, useContext, useReducer, type Dispatch, type ReactNode } from 'react';

export type LayoutKind = 'umap' | 'tsne' | 'mds';

export interface SelectionState {
  selectedModels: Set<string>;
  selectedCells: Uint32Array | null; // cell indices; null = no cell selection
  selectedRegion: number | null;
  colorBy: string; // model-map colour key, e.g. 'agg:overall', 'raw:ARI', 'factor:method', 'mv:frac_z'
  cellColorBy: string; // cell-map colour key, e.g. 'cat:cell_type', 'measure:stability_all'
  layout: LayoutKind;
  referenceModel: string;
  mode: string; // measure mode (jaccard | composition)
}

export type Action =
  | { type: 'selectModels'; ids: string[]; additive?: boolean }
  | { type: 'toggleModel'; id: string }
  | { type: 'selectCells'; cells: Uint32Array | null }
  | { type: 'selectRegion'; region: number | null; cells?: Uint32Array | null }
  | { type: 'set'; key: 'colorBy' | 'cellColorBy' | 'layout' | 'referenceModel' | 'mode'; value: string }
  | { type: 'reset' };

function reducer(s: SelectionState, a: Action): SelectionState {
  switch (a.type) {
    case 'selectModels': {
      const next = a.additive ? new Set(s.selectedModels) : new Set<string>();
      a.ids.forEach((id) => next.add(id));
      return { ...s, selectedModels: next };
    }
    case 'toggleModel': {
      const next = new Set(s.selectedModels);
      if (next.has(a.id)) next.delete(a.id);
      else next.add(a.id);
      return { ...s, selectedModels: next };
    }
    case 'selectCells':
      return { ...s, selectedCells: a.cells && a.cells.length ? a.cells : null, selectedRegion: null };
    case 'selectRegion':
      return { ...s, selectedRegion: a.region, selectedCells: a.cells ?? null };
    case 'set':
      if (a.key === 'mode') return { ...s, mode: a.value, selectedRegion: null, selectedCells: null };
      return { ...s, [a.key]: a.value } as SelectionState;
    case 'reset':
      return { ...s, selectedModels: new Set(), selectedCells: null, selectedRegion: null };
  }
}

const Ctx = createContext<{ state: SelectionState; dispatch: Dispatch<Action> } | null>(null);

export function SelectionProvider({ initial, children }: { initial: SelectionState; children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initial);
  return <Ctx.Provider value={{ state, dispatch }}>{children}</Ctx.Provider>;
}

export function useSelection() {
  const v = useContext(Ctx);
  if (!v) throw new Error('useSelection outside SelectionProvider');
  return v;
}
