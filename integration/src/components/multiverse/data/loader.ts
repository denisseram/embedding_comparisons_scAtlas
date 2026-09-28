// Fetches the static files written by multiverse-pipeline/pipeline/export.py and validates
// every shape against manifest.json. Only static files under <base>/multiverse-data/ are used.

export interface FileEntry {
  dtype: string;
  shape?: number[];
  bytes: number;
  order?: string;
  description?: string;
}

export interface Pair {
  pair_id: number;
  factor: string;
  config_a: string;
  config_b: string;
  level_a: string;
  level_b: string;
}

export interface MetricInfo {
  name: string;
  group: 'bio' | 'batch' | 'aggregate';
  direction: string;
}

export interface Manifest {
  schema_version: string;
  generated_at: string;
  dataset: { name: string; n_cells: number; n_genes: number; n_models: number; n_configs: number; n_pairs: number; seeds: number[] };
  factors: { name: string; levels: (string | number | boolean)[] }[];
  measure_factors: string[];
  measure: { k: number; modes: string[]; primary_mode: string; mode_labels: Record<string, string> };
  metrics: MetricInfo[];
  aggregates: MetricInfo[];
  weights: { batch: number; bio: number };
  model_order: string[];
  configs: string[];
  references: string[];
  neighbor_models: string[];
  pairs: Pair[];
  cell_measures: { index: number; name: string; description: string }[];
  umap_models?: { min: [number, number][]; max: [number, number][]; recipe: string } | null;
  files: Record<string, FileEntry>;
}

export interface Model {
  model_id: string;
  config_id: string;
  seed: number;
  factors: Record<string, string | number | boolean>;
  runtime_s: number | null;
  raw: Record<string, number>;
  scaled: Record<string, number>;
  overall: number;
  bio: number;
  batch: number;
  seed_sd: Record<string, number>;
  layout: Record<string, { umap: [number, number]; tsne: [number, number]; mds: [number, number] }>;
  summary: Record<string, number>;
}

export interface Cells {
  id: string[];
  x: number[];
  y: number[];
  categorical: Record<string, { codes: number[]; levels: string[] }>;
  numeric: Record<string, number[]>;
  planted_membership: Record<string, number[]>;
}

export interface RegionRecord {
  region: number;
  size: number;
  label: string;
  comp_cell_type: Record<string, number>;
  comp_sample: Record<string, number>;
  comp_study: Record<string, number>;
  comp_tissue: Record<string, number>;
  comp_planted_effect: Record<string, number>;
  qc: Record<string, { mean: number; median: number; q25: number; q75: number; z_vs_all: number }>;
  qc_deviation: number;
  E_mean: Record<string, number>;
  H_mean: Record<string, number>;
  total_effect: number;
  stability_all: number;
  stability_seed: number;
  top_genes: { gene: string; logfc: number; padj: number }[];
  variants: { code: number[]; n_sub: number[]; outside_frac: number[]; joined_to: number[]; modal_n_sub: number };
}

export interface RegionsFile {
  variant_levels: string[];
  by_mode: Record<string, { resolution: number; regions: RegionRecord[] }>;
}

export interface EntryData {
  manifest: Manifest;
  models: Model[];
  agreement: Record<string, Float32Array>; // per measure mode, [M*M]
}

export interface CellData {
  cells: Cells;
  measures: Float32Array; // [n_arrays * N]
  measureIndex: Record<string, number>;
  regions: RegionsFile;
  pairDelta: Record<string, Float32Array>; // per mode, [N * P]
  neighbors: Uint32Array; // [n_neighbor_models * N * k]
}

export class DataError extends Error {}

const SCHEMA_MAJOR = '1';

export function dataUrl(name: string): string {
  const base = (import.meta.env.BASE_URL || '/').replace(/\/?$/, '/');
  return `${base}multiverse-data/${name}`;
}

async function fetchOk(name: string): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(dataUrl(name));
  } catch (e) {
    throw new DataError(`Network error while loading ${name}: ${(e as Error).message}`);
  }
  if (!res.ok) throw new DataError(`Could not load ${dataUrl(name)} (HTTP ${res.status}). Run the pipeline export first.`);
  return res;
}

async function fetchJson<T>(name: string): Promise<T> {
  const res = await fetchOk(name);
  try {
    return (await res.json()) as T;
  } catch {
    throw new DataError(`${name} is not valid JSON.`);
  }
}

const CTORS = { float32: Float32Array, uint32: Uint32Array, uint16: Uint16Array } as const;

async function fetchBin<K extends keyof typeof CTORS>(
  manifest: Manifest,
  name: string,
  dtype: K,
  expectShape: number[],
): Promise<InstanceType<(typeof CTORS)[K]>> {
  const entry = manifest.files[name];
  if (!entry) throw new DataError(`manifest.json does not list ${name}.`);
  if (entry.dtype !== dtype) throw new DataError(`${name}: manifest dtype ${entry.dtype}, expected ${dtype}.`);
  const shape = entry.shape ?? [];
  if (shape.length !== expectShape.length || shape.some((s, i) => expectShape[i] >= 0 && s !== expectShape[i])) {
    throw new DataError(`${name}: manifest shape [${shape}] does not match expected [${expectShape.map((s) => (s < 0 ? '*' : s))}].`);
  }
  const buf = await (await fetchOk(name)).arrayBuffer();
  const Ctor = CTORS[dtype];
  const n = shape.reduce((a, b) => a * b, 1);
  if (buf.byteLength !== n * Ctor.BYTES_PER_ELEMENT || buf.byteLength !== entry.bytes) {
    throw new DataError(`${name}: got ${buf.byteLength} bytes, manifest says ${entry.bytes} for shape [${shape}] ${dtype}.`);
  }
  return new Ctor(buf) as InstanceType<(typeof CTORS)[K]>;
}

export function modeSuffix(manifest: Manifest, mode: string): string {
  return mode === manifest.measure.primary_mode ? '' : `_${mode}`;
}

export async function loadEntry(): Promise<EntryData> {
  const manifest = await fetchJson<Manifest>('manifest.json');
  if (!manifest.schema_version?.startsWith(SCHEMA_MAJOR + '.')) {
    throw new DataError(`Unsupported data schema ${manifest.schema_version}; this dashboard reads ${SCHEMA_MAJOR}.x.`);
  }
  const M = manifest.dataset.n_models;
  if (manifest.model_order.length !== M) throw new DataError(`manifest.model_order has ${manifest.model_order.length} entries, expected ${M}.`);
  const models = await fetchJson<Model[]>('models.json');
  if (models.length !== M) throw new DataError(`models.json has ${models.length} models, manifest says ${M}.`);
  models.forEach((m, i) => {
    if (m.model_id !== manifest.model_order[i]) throw new DataError(`models.json order differs from manifest at index ${i}.`);
  });
  const agreement: Record<string, Float32Array> = {};
  await Promise.all(
    manifest.measure.modes.map(async (mode) => {
      agreement[mode] = await fetchBin(manifest, `agreement${modeSuffix(manifest, mode)}.bin`, 'float32', [M, M]);
    }),
  );
  return { manifest, models, agreement };
}

export async function loadCells(manifest: Manifest): Promise<CellData> {
  const N = manifest.dataset.n_cells;
  const P = manifest.dataset.n_pairs;
  const k = manifest.measure.k;
  const [cells, measures, regions, neighbors, ...pd] = await Promise.all([
    fetchJson<Cells>('cells.json'),
    fetchBin(manifest, 'cell_measures.bin', 'float32', [manifest.cell_measures.length, N]),
    fetchJson<RegionsFile>('regions.json'),
    fetchBin(manifest, 'neighbors_sample.bin', 'uint32', [manifest.neighbor_models.length, N, k]),
    ...manifest.measure.modes.map((mode) => fetchBin(manifest, `pair_delta${modeSuffix(manifest, mode)}.bin`, 'float32', [N, P])),
  ]);
  if (cells.id.length !== N || cells.x.length !== N) throw new DataError(`cells.json has ${cells.id.length} cells, manifest says ${N}.`);
  for (const [name, col] of Object.entries(cells.categorical)) {
    if (col.codes.length !== N) throw new DataError(`cells.json column ${name} has wrong length.`);
  }
  for (const mode of manifest.measure.modes) {
    if (!regions.by_mode[mode]) throw new DataError(`regions.json lacks measure mode ${mode}.`);
  }
  const measureIndex: Record<string, number> = {};
  manifest.cell_measures.forEach((c) => (measureIndex[c.name] = c.index));
  const pairDelta: Record<string, Float32Array> = {};
  manifest.measure.modes.forEach((mode, i) => (pairDelta[mode] = pd[i]));
  return { cells, measures, measureIndex, regions, pairDelta, neighbors };
}

/** z(c, ref, model) — [R, N, M] for the primary mode, [1, N, M] for other modes. */
export async function loadZ(manifest: Manifest, mode: string): Promise<{ z: Float32Array; refs: string[] }> {
  const N = manifest.dataset.n_cells;
  const M = manifest.dataset.n_models;
  const name = `z_vs_ref${modeSuffix(manifest, mode)}.bin`;
  const entry = manifest.files[name];
  if (!entry?.shape) throw new DataError(`manifest.json does not list ${name}.`);
  const R = entry.shape[0];
  const z = await fetchBin(manifest, name, 'float32', [R, N, M]);
  return { z, refs: manifest.references.slice(0, R) };
}

export interface ModelUmaps {
  q: Uint16Array; // [M, N, 2] quantised
  min: [number, number][];
  max: [number, number][];
  recipe: string;
}

/** Per-embedding UMAPs (visual QC of each integration); null if the export has none. */
export async function loadModelUmaps(manifest: Manifest): Promise<ModelUmaps | null> {
  if (!manifest.umap_models) return null;
  const M = manifest.dataset.n_models;
  const N = manifest.dataset.n_cells;
  const q = await fetchBin(manifest, 'umap_models.bin', 'uint16', [M, N, 2]);
  const { min, max, recipe } = manifest.umap_models;
  if (min.length !== M || max.length !== M) throw new DataError('manifest.umap_models min/max do not match n_models.');
  return { q, min, max, recipe };
}

/** Decode one model's UMAP coordinates. */
export function modelUmap(u: ModelUmaps, m: number, N: number): { x: Float32Array; y: Float32Array } {
  const x = new Float32Array(N);
  const y = new Float32Array(N);
  const [x0, y0] = u.min[m];
  const sx = (u.max[m][0] - x0) / 65535;
  const sy = (u.max[m][1] - y0) / 65535;
  const base = m * N * 2;
  for (let c = 0; c < N; c++) {
    x[c] = x0 + u.q[base + 2 * c] * sx;
    y[c] = y0 + u.q[base + 2 * c + 1] * sy;
  }
  return { x, y };
}

export function cellArray(cd: CellData, name: string, N: number): Float32Array {
  const i = cd.measureIndex[name];
  if (i === undefined) throw new DataError(`cell measure ${name} not in manifest.`);
  return cd.measures.subarray(i * N, (i + 1) * N);
}
