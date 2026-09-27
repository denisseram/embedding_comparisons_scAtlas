// V6 — Design lattice: configurations on a factor grid; edges join one-factor pairs and are
// coloured by the mean seed-averaged Δ̄ over the current cell selection (recomputed in the browser).
import * as d3 from 'd3';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { INK, categorical, fmt, sequential, useTheme } from '../d3/colors';
import { Colorbar, Swatches } from '../d3/Legend';
import { Tooltip, type Tip } from '../d3/Tooltip';
import { useWidth } from '../d3/useSize';
import { pairMeans } from '../compute/selection';

const COL_F = ['n_hvg', 'hvg_batch_aware', 'exclude_igx'];
const ROW_F = ['method', 'batch_key'];

export default function DesignLattice() {
  const { entry, cells, activeCells } = useData();
  const { manifest, models } = entry;
  const { state, dispatch } = useSelection();
  const theme = useTheme();
  const wrap = useRef<HTMLDivElement>(null);
  const ref = useRef<SVGSVGElement>(null);
  const width = useWidth(wrap);
  const [tip, setTip] = useState<Tip | null>(null);
  const P = manifest.dataset.n_pairs;

  const levels = (f: string) => manifest.factors.find((x) => x.name === f)!.levels.map(String);
  const combos = (fs: string[]) => fs.reduce<string[][]>((acc, f) => acc.flatMap((a) => levels(f).map((l) => [...a, l])), [[]]);
  const cols = combos(COL_F);
  const rowsL = combos(ROW_F);
  const height = 120 + rowsL.length * 70;
  const x0 = 150;
  const cw = (width - x0 - 30) / cols.length;

  const cfg = useMemo(() => {
    const out = new Map<string, { x: number; y: number; models: string[]; f: Record<string, string> }>();
    for (const m of models) {
      const f = Object.fromEntries(Object.entries(m.factors).map(([k, v]) => [k, String(v)]));
      const c = cols.findIndex((cc) => cc.every((v, i) => f[COL_F[i]] === v));
      const r = rowsL.findIndex((rr) => rr.every((v, i) => f[ROW_F[i]] === v));
      const e = out.get(m.config_id) ?? { x: x0 + (c + 0.5) * cw, y: 100 + r * 70, models: [], f };
      e.models.push(m.model_id);
      out.set(m.config_id, e);
    }
    return out;
  }, [models, cw, width]);

  const { means, computeMs } = useMemo(() => {
    if (!cells) return { means: null, computeMs: 0 };
    const t0 = performance.now();
    const m = pairMeans(cells.pairDelta[state.mode], P, activeCells);
    return { means: m, computeMs: performance.now() - t0 };
  }, [cells, state.mode, activeCells, P]);
  const domain = useMemo(() => (means ? (d3.extent(means) as [number, number]) : ([0, 1] as [number, number])), [means]);
  const color = useMemo(() => sequential(theme, domain), [theme, domain]);
  const factorOrder = manifest.factors.map((f) => f.name);

  useEffect(() => {
    if (!means) return;
    const ink = INK[theme];
    const svg = d3.select(ref.current!);
    svg.selectAll('*').remove();
    // column headers: one short code per column (n_hvg · hvg_batch_aware · exclude_igx)
    const short = (f: string, v: string) => (f === 'n_hvg' ? v : `${f === 'hvg_batch_aware' ? 'BA' : 'exIGX'}:${v === 'true' ? 'yes' : 'no'}`);
    cols.forEach((c, i) => {
      c.forEach((v, j) =>
        svg
          .append('text')
          .attr('class', j === 0 ? 'mv-fh-text' : 'mv-fh-col')
          .attr('x', x0 + (i + 0.5) * cw)
          .attr('y', 14 + j * 12)
          .attr('text-anchor', 'middle')
          .text(j === 0 ? `n_hvg ${v}` : short(COL_F[j], v)),
      );
    });
    rowsL.forEach((r, i) => svg.append('text').attr('class', 'mv-fh-text').attr('x', 4).attr('y', 104 + i * 70).text(r.join(' · ')));

    const sel = state.selectedModels;
    const anySel = sel.size > 0;
    const cfgSelected = (id: string) => cfg.get(id)!.models.some((m) => sel.has(m));
    const gE = svg.append('g');
    manifest.pairs.forEach((p) => {
      const a = cfg.get(p.config_a)!;
      const b = cfg.get(p.config_b)!;
      const active = !anySel || (cfgSelected(p.config_a) && cfgSelected(p.config_b));
      const fi = factorOrder.indexOf(p.factor);
      let d: string;
      if (a.y === b.y) {
        const h = 10 + 7 * fi + Math.abs(a.x - b.x) * 0.08;
        d = `M${a.x},${a.y} Q${(a.x + b.x) / 2},${a.y - 2 * h} ${b.x},${b.y}`;
      } else {
        const h = 12 + 8 * fi;
        d = `M${a.x},${a.y} Q${a.x + 2 * h},${(a.y + b.y) / 2} ${b.x},${b.y}`;
      }
      gE.append('path')
        .attr('d', d)
        .attr('fill', 'none')
        .attr('stroke', active ? color(means[p.pair_id]) : ink.grid)
        .attr('stroke-width', active ? 2.2 : 1)
        .attr('stroke-opacity', active ? 0.95 : 0.6)
        .on('pointermove', function (ev: PointerEvent) {
          d3.select(this).attr('stroke-width', 4);
          setTip({
            x: ev.clientX,
            y: ev.clientY,
            title: `${p.factor}: ${p.level_a} ↔ ${p.level_b}`,
            rows: [
              { label: `mean Δ̄ over ${activeCells.length.toLocaleString()} cells`, value: fmt(means[p.pair_id]) },
              { label: 'config a', value: p.config_a },
              { label: 'config b', value: p.config_b },
            ],
          });
        })
        .on('pointerleave', function () {
          d3.select(this).attr('stroke-width', active ? 2.2 : 1);
          setTip(null);
        });
    });
    const methodCol = categorical(theme, levels('method'));
    for (const [id, c] of cfg) {
      const on = cfgSelected(id);
      svg
        .append('circle')
        .attr('cx', c.x)
        .attr('cy', c.y)
        .attr('r', on ? 7 : 5.5)
        .attr('fill', methodCol(c.f.method))
        .attr('stroke', on ? ink.primary : ink.surface)
        .attr('stroke-width', on ? 2 : 1.5)
        .style('cursor', 'pointer')
        .on('click', (ev: MouseEvent) => dispatch({ type: 'selectModels', ids: c.models, additive: ev.shiftKey }))
        .on('pointermove', (ev: PointerEvent) => setTip({ x: ev.clientX, y: ev.clientY, title: id, rows: [{ label: 'click to select its seed replicates', value: `${c.models.length} models` }] }))
        .on('pointerleave', () => setTip(null));
    }
  }, [means, cfg, color, theme, state.selectedModels, width]);

  if (!cells) return <p className="mv-loading">Loading…</p>;
  return (
    <div ref={wrap} className="mv-chart">
      <svg ref={ref} width={width} height={height} role="img" aria-label="Design lattice: configurations connected by one-factor contrasts" />
      <div className="mv-legend-row">
        <Colorbar color={color} domain={domain} label={`edge: mean Δ̄ over ${state.selectedCells ? 'selected' : 'all'} cells`} width={260} />
        <Swatches items={levels('method').map((l) => ({ key: l, label: `node: ${l}`, color: categorical(theme, levels('method'))(l) }))} />
        <span className="mv-small mv-muted">
          Column codes: BA = hvg_batch_aware, exIGX = exclude_igx. Arcs within a row change {COL_F.join(' / ')}; arcs within a column change {ROW_F.join(' / ')}. Recomputed in {computeMs.toFixed(0)} ms.
          {state.selectedModels.size ? ' Only edges between configurations with selected models are coloured.' : ''}
        </span>
      </div>
      <Tooltip tip={tip} />
    </div>
  );
}
