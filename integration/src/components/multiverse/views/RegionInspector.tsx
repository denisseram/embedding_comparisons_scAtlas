// V7 — Region inspector: what is in the selected region / cell selection, compared with the
// remaining cells. Evidence only — the dashboard does not judge biology vs artifact.
import * as d3 from 'd3';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { INK, categorical, fmt, useTheme } from '../d3/colors';
import { cellArray } from '../data/loader';
import { regionCells } from '../compute/selection';

const COMP_COLS = ['cell_type', 'sample', 'study', 'tissue', 'planted_effect'];
const QC_COLS: [string, boolean][] = [
  ['total_counts', true],
  ['pct_mito', false],
  ['doublet_score', false],
];

export default function RegionInspector() {
  const { entry, cells } = useData();
  const cd = cells!;
  const { manifest } = entry;
  const { state, dispatch } = useSelection();
  const theme = useTheme();
  const N = manifest.dataset.n_cells;
  const sel = state.selectedCells;
  const mask = useMemo(() => {
    const m = new Uint8Array(N);
    sel?.forEach((c) => (m[c] = 1));
    return m;
  }, [sel, N]);
  const region = state.selectedRegion !== null ? cd.regions.by_mode[state.mode].regions[state.selectedRegion] : null;
  const [nbA, setNbA] = useState(manifest.neighbor_models[0]);
  const [nbB, setNbB] = useState(manifest.neighbor_models[1] ?? manifest.neighbor_models[0]);

  // keyboard-reachable region picker (V4/V5 rows are pointer targets)
  const picker = (
    <label className="mv-small">
      Region{' '}
      <select
        value={state.selectedRegion ?? ''}
        onChange={(e) => {
          const r = e.target.value === '' ? null : Number(e.target.value);
          dispatch(r === null ? { type: 'selectRegion', region: null } : { type: 'selectRegion', region: r, cells: regionCells(cd, manifest, state.mode, r) });
        }}
      >
        <option value="">(none)</option>
        {cd.regions.by_mode[state.mode].regions.map((r) => (
          <option key={r.region} value={r.region}>
            {r.label} · n={r.size}
          </option>
        ))}
      </select>
    </label>
  );

  if (!sel)
    return (
      <div>
        {picker}
        <p className="mv-muted">Select a region (here, V4 or V5) or lasso cells (V1) to inspect their composition, QC, genes and neighbour provenance.</p>
      </div>
    );

  return (
    <div>
      {picker}
      <p className="mv-small">
        <strong>{region ? region.label : 'Lasso selection'}</strong> · {sel.length.toLocaleString()} cells vs {(N - sel.length).toLocaleString()} others.
        {region && ` Consensus stability ${fmt(region.stability_all)} (all models) / ${fmt(region.stability_seed)} (seed only); QC deviation ${fmt(region.qc_deviation, 2)}.`}
      </p>
      <h3>Composition (selection vs rest)</h3>
      {COMP_COLS.map((c) => (
        <CompositionBars key={c} column={c} mask={mask} theme={theme} />
      ))}
      <h3>QC and doublet score (density, selection vs rest)</h3>
      <div className="mv-inline">
        {QC_COLS.map(([c, log]) => (
          <QcHist key={c} column={c} log={log} mask={mask} theme={theme} />
        ))}
      </div>
      <h3>Top genes (Wilcoxon vs rest)</h3>
      {region ? (
        <ol className="mv-genes">
          {region.top_genes.map((g) => (
            <li key={g.gene}>
              <span className="mono">{g.gene}</span> <span className="mv-muted">logFC {fmt(g.logfc, 2)}</span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="mv-muted mv-small">Top genes are precomputed per region; select a region to see them.</p>
      )}
      <h3>Neighbour provenance</h3>
      <div className="mv-inline">
        {[
          [nbA, setNbA, 'Model A'],
          [nbB, setNbB, 'Model B'],
        ].map(([v, set, label]) => (
          <label key={label as string}>
            {label as string}
            <select value={v as string} onChange={(e) => (set as (s: string) => void)(e.target.value)}>
              {manifest.neighbor_models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <Provenance a={manifest.neighbor_models.indexOf(nbA)} b={manifest.neighbor_models.indexOf(nbB)} mask={mask} theme={theme} />
    </div>
  );
}

function CompositionBars({ column, mask, theme }: { column: string; mask: Uint8Array; theme: 'light' | 'dark' }) {
  const { cells } = useData();
  const col = cells!.cells.categorical[column];
  const ref = useRef<SVGSVGElement>(null);
  const rows = useMemo(() => {
    const L = col.levels.length;
    const inS = new Array(L).fill(0);
    const out = new Array(L).fill(0);
    let nS = 0;
    for (let c = 0; c < mask.length; c++) {
      if (mask[c]) (inS[col.codes[c]]++, nS++);
      else out[col.codes[c]]++;
    }
    const nO = mask.length - nS || 1;
    return col.levels.map((l, i) => ({ l, s: inS[i] / (nS || 1), o: out[i] / nO })).filter((r) => r.s > 0 || r.o > 0.005);
  }, [col, mask]);
  const H = 12;
  const height = rows.length * (2 * H + 4) + 16;
  useEffect(() => {
    const ink = INK[theme];
    const accent = categorical(theme, ['sel'])('sel');
    const svg = d3.select(ref.current!);
    svg.selectAll('*').remove();
    const x = d3.scaleLinear().domain([0, 1]).range([120, 380]);
    svg.append('text').attr('class', 'mv-fh-group').attr('x', 0).attr('y', 11).text(column);
    rows.forEach((r, i) => {
      const y = 16 + i * (2 * H + 4);
      svg.append('text').attr('class', 'mv-fh-text').attr('x', 0).attr('y', y + H + 3).text(r.l);
      svg.append('rect').attr('x', x(0)).attr('y', y).attr('width', Math.max(1, x(r.s) - x(0))).attr('height', H - 2).attr('rx', 2).attr('fill', accent);
      svg.append('rect').attr('x', x(0)).attr('y', y + H).attr('width', Math.max(1, x(r.o) - x(0))).attr('height', H - 2).attr('rx', 2).attr('fill', ink.neutral);
      svg.append('text').attr('class', 'mv-fh-col').attr('x', x(r.s) + 4).attr('y', y + H - 3).text(`${(100 * r.s).toFixed(0)}%`);
      svg.append('text').attr('class', 'mv-fh-col').attr('x', x(r.o) + 4).attr('y', y + 2 * H - 3).text(`${(100 * r.o).toFixed(0)}% rest`);
    });
  }, [rows, theme, column]);
  return <svg ref={ref} width={440} height={height} className="mv-bars" role="img" aria-label={`${column} composition, selection versus rest`} />;
}

function QcHist({ column, log, mask, theme }: { column: string; log: boolean; mask: Uint8Array; theme: 'light' | 'dark' }) {
  const { cells } = useData();
  const vals = cells!.cells.numeric[column];
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const ink = INK[theme];
    const accent = categorical(theme, ['sel'])('sel');
    const W = 220;
    const Hh = 110;
    const tf = (v: number) => (log ? Math.log10(Math.max(v, 1)) : v);
    const all = Array.from(vals, tf);
    const x = d3.scaleLinear().domain(d3.extent(all) as [number, number]).nice().range([6, W - 6]);
    const bins = d3.bin().domain(x.domain() as [number, number]).thresholds(x.ticks(30));
    const bs = bins(all.filter((_, c) => mask[c]));
    const bo = bins(all.filter((_, c) => !mask[c]));
    const dens = (b: d3.Bin<number, number>[]) => {
      const n = d3.sum(b, (d) => d.length) || 1;
      return b.map((d) => d.length / n);
    };
    const ds = dens(bs);
    const dO = dens(bo);
    const y = d3.scaleLinear().domain([0, d3.max([...ds, ...dO]) || 1]).range([Hh - 22, 16]);
    const svg = d3.select(ref.current!);
    svg.selectAll('*').remove();
    svg.append('text').attr('class', 'mv-fh-group').attr('x', 6).attr('y', 11).text(log ? `log10 ${column}` : column);
    const line = (d: number[], b: d3.Bin<number, number>[]) =>
      d3
        .line<number>()
        .x((_, i) => x(((b[i].x0 ?? 0) + (b[i].x1 ?? 0)) / 2))
        .y((v) => y(v))
        .curve(d3.curveStepAfter)(d);
    svg.append('path').attr('d', line(dO, bo)).attr('fill', 'none').attr('stroke', ink.neutral).attr('stroke-width', 2);
    svg.append('path').attr('d', line(ds, bs)).attr('fill', 'none').attr('stroke', accent).attr('stroke-width', 2);
    svg
      .append('g')
      .attr('class', 'mv-axis')
      .attr('transform', `translate(0,${Hh - 22})`)
      .call(d3.axisBottom(x).ticks(4).tickSize(3));
  }, [vals, mask, theme, log, column]);
  return <svg ref={ref} width={220} height={110} role="img" aria-label={`${column} distribution, selection (blue) versus rest (grey)`} />;
}

function Provenance({ a, b, mask, theme }: { a: number; b: number; mask: Uint8Array; theme: 'light' | 'dark' }) {
  const { entry, cells } = useData();
  const { state } = useSelection();
  const { manifest } = entry;
  const cd = cells!;
  const N = manifest.dataset.n_cells;
  const k = manifest.measure.k;
  const reg = cellArray(cd, `region.${state.mode}`, N);
  const stats = useMemo(() => {
    const calc = (m: number) => {
      const counts = new Map<number, number>();
      let inside = 0;
      let tot = 0;
      for (let c = 0; c < N; c++) {
        if (!mask[c]) continue;
        const base = (m * N + c) * k;
        for (let j = 0; j < k; j++) {
          const nb = cd.neighbors[base + j];
          counts.set(reg[nb], (counts.get(reg[nb]) ?? 0) + 1);
          if (mask[nb]) inside++;
          tot++;
        }
      }
      return { counts, inside: inside / (tot || 1), tot };
    };
    let jac = 0;
    let n = 0;
    for (let c = 0; c < N; c++) {
      if (!mask[c]) continue;
      const A = new Set(Array.from(cd.neighbors.subarray((a * N + c) * k, (a * N + c + 1) * k)));
      let inter = 0;
      for (let j = 0; j < k; j++) if (A.has(cd.neighbors[(b * N + c) * k + j])) inter++;
      jac += inter / (2 * k - inter);
      n++;
    }
    return { A: calc(a), B: calc(b), jaccard: jac / (n || 1) };
  }, [a, b, mask, cd, N, k, reg]);

  const top = [...stats.A.counts.entries()].sort((x, y) => y[1] - x[1]).slice(0, 5).map(([r]) => r);
  const col = categorical(theme, top.map(String));
  const seg = (s: typeof stats.A) => {
    const parts = top.map((r) => ({ key: `R${r}`, v: (s.counts.get(r) ?? 0) / (s.tot || 1), color: col(String(r)) }));
    parts.push({ key: 'other', v: 1 - parts.reduce((acc, p) => acc + p.v, 0), color: INK[theme].neutral });
    return parts;
  };
  const Bar = ({ label, s }: { label: string; s: typeof stats.A }) => (
    <div className="mv-small" style={{ margin: '4px 0' }}>
      <div>
        {label}: <strong>{(100 * s.inside).toFixed(0)}%</strong> of neighbours inside the selection
      </div>
      <div style={{ display: 'flex', height: 12, borderRadius: 3, overflow: 'hidden', maxWidth: 420, gap: 2 }} role="img" aria-label={`${label} neighbour regions`}>
        {seg(s).map((p) => (
          <div key={p.key} title={`${p.key}: ${(100 * p.v).toFixed(1)}%`} style={{ width: `${100 * p.v}%`, background: p.color }} />
        ))}
      </div>
    </div>
  );
  return (
    <div>
      <Bar label="Model A" s={stats.A} />
      <Bar label="Model B" s={stats.B} />
      <p className="mv-small">
        Regions of neighbours: {top.map((r) => `R${r}`).join(', ')}, other. Mean neighbour-set Jaccard A vs B over these cells:{' '}
        <strong>{fmt(stats.jaccard)}</strong> (k={k}, own latent kNN graphs).
      </p>
    </div>
  );
}
