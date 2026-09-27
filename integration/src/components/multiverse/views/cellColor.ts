// Resolves a cell-map colour key into per-cell colours and legend information.
import type { CellData, Manifest } from '../data/loader';
import { cellArray } from '../data/loader';
import { INK, categorical, diverging, sequential, type Theme } from '../d3/colors';

export interface CellColorOption {
  key: string;
  label: string;
  group: string;
}

export function cellColorOptions(manifest: Manifest, cd: CellData, mode: string): CellColorOption[] {
  const out: CellColorOption[] = [];
  for (const c of Object.keys(cd.cells.categorical)) out.push({ key: `cat:${c}`, label: c, group: 'Observations' });
  for (const c of Object.keys(cd.cells.numeric)) out.push({ key: `num:${c}`, label: c, group: 'QC' });
  out.push({ key: 'measure:stability_all', label: 'consensus stability (all models)', group: 'Stability' });
  out.push({ key: 'measure:stability_seed', label: 'consensus stability (seed replicates only)', group: 'Stability' });
  for (const f of manifest.measure_factors) out.push({ key: `measure:E.${mode}.${f}`, label: `E(c, ${f})`, group: 'Effect E' });
  for (const f of manifest.measure_factors) out.push({ key: `measure:H.${mode}.${f}`, label: `H(c, ${f})`, group: 'Interaction H' });
  for (const f of manifest.measure_factors) out.push({ key: `measure:Erank.${mode}.${f}`, label: `rank-based E(c, ${f})`, group: 'Effect E (rank-based)' });
  out.push({ key: 'region', label: 'region (coloured by total |E|, labelled)', group: 'Regions' });
  out.push({ key: 'zsel', label: 'z vs reference for the selected model', group: 'Calibrated change' });
  out.push({ key: 'measure:majority_vote_disagreement', label: 'majority-vote label disagreement', group: 'Paper-style baselines' });
  out.push({ key: 'measure:paper_graph_dissimilarity', label: 'graph dissimilarity (mean |d|)', group: 'Paper-style baselines' });
  return out;
}

export type CellColor =
  | { kind: 'categorical'; label: string; levels: string[]; colorOf: (l: string) => string; code: (c: number) => number; color: (c: number) => string }
  | { kind: 'continuous'; label: string; value: (c: number) => number; color: (c: number) => string; scale: (v: number) => string; domain: [number, number]; note?: string };

function extent(a: ArrayLike<number>): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo <= hi ? [lo, hi] : [0, 1];
}

/** robust symmetric range for signed effects: 98th percentile of |v| */
function robustAbs(a: ArrayLike<number>): number {
  const s = Float64Array.from(a, Math.abs).sort();
  return s[Math.floor(0.98 * (s.length - 1))] || 1;
}

export function resolveCellColor(
  key: string,
  cd: CellData,
  manifest: Manifest,
  theme: Theme,
  opts: { mode?: string; regionTotals?: number[]; zSel?: Float32Array | null; zLabel?: string } = {},
): CellColor {
  const N = manifest.dataset.n_cells;
  const [kind, name] = key.split(/:(.*)/s);
  if (kind === 'cat') {
    const col = cd.cells.categorical[name];
    const colorOf = categorical(theme, col.levels);
    const lut = col.levels.map(colorOf);
    return { kind: 'categorical', label: name, levels: col.levels, colorOf, code: (c) => col.codes[c], color: (c) => lut[col.codes[c]] };
  }
  let values: ArrayLike<number>;
  let label = name;
  let signed = false;
  let note: string | undefined;
  if (kind === 'num') values = cd.cells.numeric[name];
  else if (kind === 'measure') {
    values = cellArray(cd, name, N);
    signed = name.startsWith('E.') || name.startsWith('Erank.');
    label = name;
  } else if (kind === 'region') {
    const reg = cellArray(cd, `region.${opts.mode ?? manifest.measure.primary_mode}`, N);
    const tot = opts.regionTotals ?? [];
    values = Float32Array.from(reg, (r) => tot[r] ?? 0);
    label = 'region total |E|';
  } else {
    values = opts.zSel ?? new Float32Array(N).fill(NaN);
    signed = true;
    label = opts.zLabel ?? 'z vs reference';
    if (!opts.zSel) note = 'Select exactly one model (and wait for z to load) to colour by its calibrated change vs the reference.';
  }
  const domain: [number, number] = signed ? [-robustAbs(values), robustAbs(values)] : extent(values);
  const scale = signed ? diverging(theme, domain[1]) : sequential(theme, domain);
  const nan = INK[theme].neutral;
  return {
    kind: 'continuous',
    label,
    value: (c) => values[c],
    color: (c) => (Number.isFinite(values[c]) ? scale(values[c]) : nan),
    scale,
    domain,
    note,
  };
}
