// In-browser recomputation over the current cell selection (everything selection-dependent
// is computed here from exported per-cell arrays; nothing is sent to a server).
import type { CellData, Manifest, Model } from '../data/loader';
import { cellArray } from '../data/loader';

export function allCells(N: number): Uint32Array {
  const out = new Uint32Array(N);
  for (let i = 0; i < N; i++) out[i] = i;
  return out;
}

/** Fraction of selected cells with |z(c, ref, model)| > thr, for every model. */
export function fracChanged(z: Float32Array, refIdx: number, N: number, M: number, cells: Uint32Array, thr = 2): Float32Array {
  const out = new Float32Array(M);
  const base = refIdx * N * M;
  for (let ci = 0; ci < cells.length; ci++) {
    const row = base + cells[ci] * M;
    for (let m = 0; m < M; m++) if (Math.abs(z[row + m]) > thr) out[m] += 1;
  }
  const n = cells.length || 1;
  for (let m = 0; m < M; m++) out[m] /= n;
  return out;
}

/** Mean seed-averaged Δ̄ per matched pair over the selected cells (lattice edge colours). */
export function pairMeans(pairDelta: Float32Array, P: number, cells: Uint32Array): Float32Array {
  const out = new Float32Array(P);
  for (let ci = 0; ci < cells.length; ci++) {
    const row = cells[ci] * P;
    for (let p = 0; p < P; p++) out[p] += pairDelta[row + p];
  }
  const n = cells.length || 1;
  for (let p = 0; p < P; p++) out[p] /= n;
  return out;
}

export function median(a: ArrayLike<number>): number {
  const s = Float64Array.from(a).sort();
  if (!s.length) return NaN;
  const h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

export interface RegionEffect {
  region: number;
  size: number;
  E: number[]; // per factor (manifest.measure_factors order)
  H: number[];
  total: number; // sum |E| over real factors
}

/** Region x factor effect means, optionally relative to each factor's median over all cells. */
export function regionEffects(cd: CellData, manifest: Manifest, mode: string, relative: boolean): RegionEffect[] {
  const N = manifest.dataset.n_cells;
  const F = manifest.measure_factors;
  const reg = cellArray(cd, `region.${mode}`, N);
  const nR = Math.max(...Array.from(reg)) + 1;
  const Es = F.map((f) => cellArray(cd, `E.${mode}.${f}`, N));
  const Hs = F.map((f) => cellArray(cd, `H.${mode}.${f}`, N));
  const offs = F.map((_, i) => (relative ? median(Es[i]) : 0));
  const out: RegionEffect[] = Array.from({ length: nR }, (_, r) => ({ region: r, size: 0, E: F.map(() => 0), H: F.map(() => 0), total: 0 }));
  for (let c = 0; c < N; c++) {
    const r = out[reg[c]];
    r.size++;
    for (let i = 0; i < F.length; i++) {
      r.E[i] += Es[i][c] - offs[i];
      r.H[i] += Hs[i][c];
    }
  }
  for (const r of out) {
    for (let i = 0; i < F.length; i++) {
      r.E[i] /= r.size || 1;
      r.H[i] /= r.size || 1;
    }
    r.total = r.E.reduce((a, v, i) => a + (F[i] === 'seed' ? 0 : Math.abs(v)), 0);
  }
  return out;
}

export function regionCells(cd: CellData, manifest: Manifest, mode: string, region: number): Uint32Array {
  const N = manifest.dataset.n_cells;
  const reg = cellArray(cd, `region.${mode}`, N);
  const idx: number[] = [];
  for (let c = 0; c < N; c++) if (reg[c] === region) idx.push(c);
  return Uint32Array.from(idx);
}

/** Lexicographic model order by factors then seed (used by V3 and V5). */
export function lexOrder(manifest: Manifest, models: Model[]): number[] {
  const keys = ['method', 'batch_key', 'n_hvg', 'hvg_batch_aware', 'exclude_igx'];
  const levelIdx = (f: string, v: unknown) => manifest.factors.find((x) => x.name === f)!.levels.map(String).indexOf(String(v));
  return models
    .map((m, i) => ({ i, k: [...keys.map((f) => levelIdx(f, m.factors[f])), m.seed] }))
    .sort((a, b) => {
      for (let j = 0; j < a.k.length; j++) if (a.k[j] !== b.k[j]) return a.k[j] - b.k[j];
      return 0;
    })
    .map((x) => x.i);
}

/** Rank (1 = best) of every model for a value accessor; ties share the better rank. */
export function ranks(values: number[]): number[] {
  const order = values.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]);
  const r = new Array(values.length);
  order.forEach(([v, i], pos) => {
    r[i] = pos > 0 && order[pos - 1][0] === v ? r[order[pos - 1][1]] : pos + 1;
  });
  return r;
}
