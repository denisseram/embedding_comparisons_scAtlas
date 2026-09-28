// Comparative fuzzy UpSet, browser side: threshold -> signature -> aggregation on the exported neighbour-label
// counts (fuzzy_memberships.bin.gz). Mirrors multiverse-pipeline/pipeline/fuzzy_upset.py (the reference
// implementation; tests/test_fuzzy_upset.py checks this file against it). No imports, so Node can run it directly.
//
// P[c, l] = count / (k+1); a label is in the cell's signature when score >= tau, where score = P (or P / global
// label frequency when normalize is on). Strength = min raw P over the signature. > maxSize labels -> diffuse.

export interface FuzzyColumn {
  name: string;
  levels: string[];
  offset: number;
  freq: number[];
  n_unlabelled: number;
  warnings: string[];
}

export interface FuzzyInfo {
  file: string;
  k: number;
  models: string[];
  columns: FuzzyColumn[];
  n_levels: number;
  second_columns: string[];
  second_default: string | null;
  qc_columns: string[];
  defaults: { tau: number; max_size: number; normalize: boolean; hide_pure: boolean };
}

export interface FuzzyParams {
  tau: number;
  maxSize: number;
  normalize: boolean;
}

export const NONE = -1;
export const DIFFUSE = -2;

export interface ModelSignatures {
  sig: Int32Array; // per cell: index into keys, NONE or DIFFUSE
  strength: Float32Array; // min raw membership over the signature (0 for NONE / DIFFUSE)
  keys: string[]; // intersection keys: sorted label indices joined by ','
  nNone: number;
  nDiffuse: number;
}

/** Signatures of one model (index m in FuzzyInfo.models) for one label column. */
export function modelSignatures(counts: Uint8Array, info: FuzzyInfo, N: number, m: number, col: FuzzyColumn, p: FuzzyParams): ModelSignatures {
  const L = col.levels.length;
  const denom = info.k + 1;
  // NumPy compares float32 memberships with a Python float in float32 (NEP 50); enrichment is float64 there too
  const tau = p.normalize ? p.tau : Math.fround(p.tau);
  const sig = new Int32Array(N);
  const strength = new Float32Array(N);
  const keyId = new Map<string, number>();
  const keys: string[] = [];
  const buf: number[] = [];
  let nNone = 0;
  let nDiffuse = 0;
  for (let c = 0; c < N; c++) {
    const base = (m * N + c) * info.n_levels + col.offset;
    buf.length = 0;
    let smin = Infinity;
    for (let l = 0; l < L; l++) {
      const cnt = counts[base + l];
      if (!cnt) continue;
      const raw = Math.fround(cnt / denom); // float32 like the Python reference
      const f = col.freq[l];
      const score = p.normalize ? (f > 0 ? raw / f : 0) : raw;
      if (score >= tau) {
        buf.push(l);
        if (raw < smin) smin = raw;
      }
    }
    if (buf.length === 0) {
      sig[c] = NONE;
      nNone++;
    } else if (buf.length > p.maxSize) {
      sig[c] = DIFFUSE;
      nDiffuse++;
    } else {
      const key = buf.join(',');
      let id = keyId.get(key);
      if (id === undefined) {
        id = keys.length;
        keyId.set(key, id);
        keys.push(key);
      }
      sig[c] = id;
      strength[c] = smin;
    }
  }
  return { sig, strength, keys, nNone, nDiffuse };
}

export interface Intersection {
  key: string;
  labels: number[];
  fuzzy: Float64Array; // per compared model (order of the models argument)
  nCells: Uint32Array;
}

export interface Comparison {
  intersections: Intersection[]; // sorted by (degree, labels)
  perModel: ModelSignatures[];
}

/** Intersection x model tables (fuzzy_size and n_cells; 0 where an intersection is absent in a model). */
export function compareModels(perModel: ModelSignatures[]): Comparison {
  const byKey = new Map<string, Intersection>();
  const M = perModel.length;
  perModel.forEach((ms, j) => {
    const fz = new Float64Array(ms.keys.length);
    const nc = new Uint32Array(ms.keys.length);
    for (let c = 0; c < ms.sig.length; c++) {
      const s = ms.sig[c];
      if (s < 0) continue;
      fz[s] += ms.strength[c];
      nc[s]++;
    }
    ms.keys.forEach((key, i) => {
      let it = byKey.get(key);
      if (!it) {
        it = { key, labels: key.split(',').map(Number), fuzzy: new Float64Array(M), nCells: new Uint32Array(M) };
        byKey.set(key, it);
      }
      it.fuzzy[j] = fz[i];
      it.nCells[j] = nc[i];
    });
  });
  const intersections = [...byKey.values()].sort((a, b) => {
    if (a.labels.length !== b.labels.length) return a.labels.length - b.labels.length;
    for (let i = 0; i < a.labels.length; i++) if (a.labels[i] !== b.labels[i]) return a.labels[i] - b.labels[i];
    return 0;
  });
  return { intersections, perModel };
}

export type DiffKind = 'range' | 'var' | 'log_ratio';

/** Difference across models: max - min, population variance, or log2((a+pc)/(b+pc)) for two model positions. */
export function difference(it: Intersection, how: DiffKind, pair?: [number, number], pseudocount = 1): number {
  const v = it.fuzzy;
  if (how === 'log_ratio') {
    if (!pair) throw new Error('log_ratio needs two models');
    return Math.log2((v[pair[0]] + pseudocount) / (v[pair[1]] + pseudocount));
  }
  if (!v.length) return 0;
  if (how === 'range') {
    let lo = Infinity;
    let hi = -Infinity;
    for (const x of v) {
      if (x < lo) lo = x;
      if (x > hi) hi = x;
    }
    return hi - lo;
  }
  let mean = 0;
  for (const x of v) mean += x;
  mean /= v.length;
  let s = 0;
  for (const x of v) s += (x - mean) ** 2;
  return s / v.length;
}

/** Cells whose signature is exactly `key` in one model. */
export function cellsFor(ms: ModelSignatures, key: string): Uint32Array {
  const id = ms.keys.indexOf(key);
  if (id < 0) return new Uint32Array(0);
  let n = 0;
  for (let c = 0; c < ms.sig.length; c++) if (ms.sig[c] === id) n++;
  const out = new Uint32Array(n);
  n = 0;
  for (let c = 0; c < ms.sig.length; c++) if (ms.sig[c] === id) out[n++] = c;
  return out;
}

export interface Attributes {
  composition: Map<string, Float64Array>; // key -> fraction of the intersection's cells per level of the second column
  qcMean: Map<string, number[]>; // key -> mean per QC column (NaN if no finite value)
}

/** For one model: composition by a second categorical column (codes < 0 ignored) and mean QC values. */
export function attributes(ms: ModelSignatures, second: ArrayLike<number> | null, nSecond: number, qc: ArrayLike<number>[]): Attributes {
  const K = ms.keys.length;
  const n = new Float64Array(K);
  const comp = new Float64Array(K * Math.max(nSecond, 1));
  const qs = qc.map(() => new Float64Array(K));
  const qn = qc.map(() => new Float64Array(K));
  for (let c = 0; c < ms.sig.length; c++) {
    const s = ms.sig[c];
    if (s < 0) continue;
    n[s]++;
    if (second) {
      const g = second[c];
      if (g >= 0) comp[s * nSecond + g]++;
    }
    for (let q = 0; q < qc.length; q++) {
      const v = qc[q][c];
      if (v !== null && Number.isFinite(v)) {
        qs[q][s] += v;
        qn[q][s]++;
      }
    }
  }
  const composition = new Map<string, Float64Array>();
  const qcMean = new Map<string, number[]>();
  ms.keys.forEach((key, i) => {
    const row = new Float64Array(nSecond);
    for (let g = 0; g < nSecond; g++) row[g] = comp[i * nSecond + g] / (n[i] || 1);
    composition.set(key, row);
    qcMean.set(key, qc.map((_, q) => (qn[q][i] ? qs[q][i] / qn[q][i] : NaN)));
  });
  return { composition, qcMean };
}

// ---- memoisation: signatures per (column, model, tau, maxSize, normalize) ---------------------------------
const CACHE_MAX = 256;
const cache = new Map<string, ModelSignatures>();

export function cachedSignatures(counts: Uint8Array, info: FuzzyInfo, N: number, m: number, col: FuzzyColumn, p: FuzzyParams): ModelSignatures {
  const key = `${col.name}|${m}|${p.tau}|${p.maxSize}|${p.normalize ? 1 : 0}`;
  let v = cache.get(key);
  if (v) {
    cache.delete(key); // refresh LRU position
    cache.set(key, v);
    return v;
  }
  v = modelSignatures(counts, info, N, m, col, p);
  cache.set(key, v);
  if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value!);
  return v;
}
