// V0b — Leaderboard: top 10 models for the metric selected in the V0a colour dropdown,
// as a sortable table or a model-level funky heatmap. Seed-replicate sd is shown throughout.
import { useMemo, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { fmt } from '../d3/colors';
import type { Model } from '../data/loader';
import FunkyHeatmap from './FunkyHeatmap';

export interface LeaderMetric {
  key: string;
  label: string;
  value: (m: Model) => number;
  sd: (m: Model) => number | undefined;
  fallback: boolean;
}

export function leaderMetric(colorBy: string): LeaderMetric {
  const [kind, name] = colorBy.split(/:(.*)/s);
  if (kind === 'agg') return { key: colorBy, label: name, value: (m) => (m as unknown as Record<string, number>)[name], sd: (m) => m.seed_sd[name], fallback: false };
  if (kind === 'raw') return { key: colorBy, label: `${name} (raw)`, value: (m) => m.raw[name], sd: (m) => m.seed_sd[name], fallback: false };
  if (kind === 'scaled') return { key: colorBy, label: `${name} (scaled)`, value: (m) => m.scaled[name], sd: (m) => m.seed_sd[`${name}_scaled`], fallback: false };
  return { key: 'agg:overall', label: 'overall', value: (m) => m.overall, sd: (m) => m.seed_sd.overall, fallback: true };
}

type SortKey = 'rank' | 'metric' | 'overall' | 'bio' | 'batch' | 'model_id';
const FACTOR_COLS = ['method', 'batch_key', 'n_hvg', 'hvg_batch_aware', 'exclude_igx'];

export default function Leaderboard() {
  const { entry } = useData();
  const { models } = entry;
  const { state, dispatch } = useSelection();
  const [view, setView] = useState<'table' | 'funky'>('table');
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'rank', dir: 1 });
  const metric = leaderMetric(state.colorBy);

  const top = useMemo(() => {
    const ranked = [...models].sort((a, b) => metric.value(b) - metric.value(a)).slice(0, 10);
    const best = ranked[0];
    return ranked.map((m, i) => ({
      m,
      rank: i + 1,
      v: metric.value(m),
      sd: metric.sd(m),
      // within noise of rank 1: gap smaller than the larger of the two seed sds
      withinNoise: i > 0 && Math.abs(metric.value(best) - metric.value(m)) <= Math.max(metric.sd(best) ?? 0, metric.sd(m) ?? 0),
    }));
  }, [models, metric.key]);

  const rows = useMemo(() => {
    const get = (r: (typeof top)[number]): number | string =>
      sort.key === 'rank' ? r.rank : sort.key === 'metric' ? r.v : sort.key === 'model_id' ? r.m.model_id : r.m[sort.key];
    return [...top].sort((a, b) => {
      const x = get(a);
      const y = get(b);
      return (x < y ? -1 : x > y ? 1 : 0) * sort.dir;
    });
  }, [top, sort]);

  const header = (key: SortKey, label: string) => (
    <th scope="col" aria-sort={sort.key === key ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}>
      <button type="button" onClick={() => setSort((s) => ({ key, dir: s.key === key ? (-s.dir as 1 | -1) : key === 'rank' || key === 'model_id' ? 1 : -1 }))}>
        {label}
        {sort.key === key ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}
      </button>
    </th>
  );

  const pick = (id: string, add: boolean) => dispatch(add ? { type: 'toggleModel', id } : { type: 'selectModels', ids: [id] });

  return (
    <div className="mv-leaderboard">
      <div className="mv-toolbar">
        <span>
          Top 10 by <strong>{metric.label}</strong>
          {metric.fallback && <span className="mv-muted"> (colour key is not a benchmark metric; ranking by overall)</span>}
        </span>
        <div className="mv-seg" role="group" aria-label="Leaderboard view">
          <button type="button" aria-pressed={view === 'table'} onClick={() => setView('table')}>Table</button>
          <button type="button" aria-pressed={view === 'funky'} onClick={() => setView('funky')}>Funky heatmap</button>
        </div>
      </div>
      {view === 'table' ? (
        <div className="mv-table-wrap">
          <table className="mv-table">
            <thead>
              <tr>
                {header('rank', '#')}
                {header('model_id', 'model')}
                {FACTOR_COLS.map((f) => (
                  <th scope="col" key={f}>{f}</th>
                ))}
                <th scope="col">seed</th>
                {header('metric', metric.label)}
                <th scope="col">seed sd</th>
                {header('overall', 'overall')}
                {header('bio', 'bio')}
                {header('batch', 'batch')}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.m.model_id}
                  tabIndex={0}
                  className={state.selectedModels.has(r.m.model_id) ? 'is-selected' : ''}
                  onClick={(e) => pick(r.m.model_id, e.shiftKey || e.metaKey)}
                  onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), pick(r.m.model_id, e.shiftKey))}
                  aria-selected={state.selectedModels.has(r.m.model_id)}
                >
                  <td className="num">
                    {r.rank}
                    {r.withinNoise && <span className="mv-badge" title="Gap to rank 1 is within one seed-replicate sd">≈1</span>}
                  </td>
                  <td className="mono">{r.m.model_id}</td>
                  {FACTOR_COLS.map((f) => (
                    <td key={f}>{String(r.m.factors[f])}</td>
                  ))}
                  <td className="num">{r.m.seed}</td>
                  <td className="num strong">{fmt(r.v)}</td>
                  <td className="num">± {fmt(r.sd)}</td>
                  <td className="num">{fmt(r.m.overall)}</td>
                  <td className="num">{fmt(r.m.bio)}</td>
                  <td className="num">{fmt(r.m.batch)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mv-muted mv-small">“≈1” marks models whose gap to rank 1 is within one seed-replicate sd: their rank is not distinguishable from rerun noise.</p>
        </div>
      ) : (
        <FunkyHeatmap rows={top.map((r) => r.m)} metric={metric} />
      )}
    </div>
  );
}
