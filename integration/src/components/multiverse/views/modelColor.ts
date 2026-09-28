// Resolves the V0a colour key into per-model values, a colour function and legend info.
import type { Manifest, Model } from '../data/loader';
import { categorical, sequential, type Theme } from '../d3/colors';

export interface ColorOption {
  key: string;
  label: string;
  group: string;
}

/** Model-map (V0a) colour options: multiverse measures and decisions only. */
export function colorOptions(manifest: Manifest): ColorOption[] {
  const out: ColorOption[] = [
    { key: 'mv:consensus_share', label: 'mean consensus stability (share of consensus neighbours)', group: 'Multiverse measures' },
    { key: 'mv:mean_agreement', label: 'mean Δ to all other models', group: 'Multiverse measures' },
    { key: 'mv:frac_z', label: 'fraction of cells with |z| > 2 vs reference', group: 'Multiverse measures' },
  ];
  for (const f of manifest.factors) out.push({ key: `factor:${f.name}`, label: f.name, group: 'Decisions' });
  out.push({ key: 'factor:seed', label: 'seed', group: 'Decisions' });
  return out;
}

/** Leaderboard (V0b) ranking options: benchmark aggregates and individual metrics. */
export function rankOptions(manifest: Manifest): ColorOption[] {
  const out: ColorOption[] = [
    { key: 'agg:overall', label: 'overall', group: 'Benchmark aggregates' },
    { key: 'agg:bio', label: 'bio', group: 'Benchmark aggregates' },
    { key: 'agg:batch', label: 'batch', group: 'Benchmark aggregates' },
  ];
  for (const m of manifest.metrics) out.push({ key: `raw:${m.name}`, label: `${m.name} (raw, ${m.group})`, group: 'Metrics (raw)' });
  for (const m of manifest.metrics) out.push({ key: `scaled:${m.name}`, label: `${m.name} (scaled, ${m.group})`, group: 'Metrics (scaled)' });
  return out;
}

export type Resolved =
  | { kind: 'continuous'; label: string; value: (i: number) => number; color: (i: number) => string; scale: (v: number) => string; domain: [number, number]; sd?: (i: number) => number | undefined; higherBetter: boolean; missing?: string }
  | { kind: 'categorical'; label: string; value: (i: number) => string; color: (i: number) => string; levels: string[]; colorOf: (v: string) => string };

export function resolveColor(
  key: string,
  manifest: Manifest,
  models: Model[],
  theme: Theme,
  mode: string,
  fracZ: Float32Array | null,
): Resolved {
  const [kind, name] = key.split(/:(.*)/s);
  if (kind === 'factor') {
    const levels = name === 'seed' ? manifest.dataset.seeds.map(String) : manifest.factors.find((f) => f.name === name)!.levels.map(String);
    const colorOf = categorical(theme, levels);
    const value = (i: number) => (name === 'seed' ? String(models[i].seed) : String(models[i].factors[name]));
    return { kind: 'categorical', label: name, value, color: (i) => colorOf(value(i)), levels, colorOf };
  }
  let value: (i: number) => number;
  let sd: ((i: number) => number | undefined) | undefined;
  let label = name;
  let higherBetter = true;
  let missing: string | undefined;
  if (kind === 'agg') {
    value = (i) => (models[i] as unknown as Record<string, number>)[name];
    sd = (i) => models[i].seed_sd[name];
  } else if (kind === 'raw') {
    value = (i) => models[i].raw[name];
    sd = (i) => models[i].seed_sd[name];
    label = `${name} (raw)`;
  } else if (kind === 'scaled') {
    value = (i) => models[i].scaled[name];
    sd = (i) => models[i].seed_sd[`${name}_scaled`];
    label = `${name} (scaled)`;
  } else if (name === 'consensus_share') {
    value = (i) => models[i].summary.consensus_share;
    label = 'mean consensus stability';
  } else if (name === 'mean_agreement') {
    value = (i) => models[i].summary[`mean_agreement_delta_${mode}`];
    label = 'mean Δ to other models';
    higherBetter = false;
  } else {
    const fz = fracZ;
    value = (i) => (fz ? fz[i] : NaN);
    label = 'fraction |z| > 2 vs reference';
    higherBetter = false;
    if (!fz) missing = 'loading calibrated change…';
  }
  const vals = models.map((_, i) => value(i)).filter((v) => Number.isFinite(v));
  const domain: [number, number] = vals.length ? [Math.min(...vals), Math.max(...vals)] : [0, 1];
  const scale = sequential(theme, domain);
  return { kind: 'continuous', label, value, color: (i) => (Number.isFinite(value(i)) ? scale(value(i)) : '#999'), scale, domain, sd, higherBetter, missing };
}
