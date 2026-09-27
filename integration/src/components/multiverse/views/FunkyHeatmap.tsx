// Model-level funky heatmap (after Ext. Fig. 3A of the scAtlasTb preprint), drawn with D3 in SVG:
// rows = models, column groups setup | overall | batch | bio; bars for aggregates, circles for metrics.
import * as d3 from 'd3';
import { useEffect, useRef, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { INK, categorical, fmt, sequential, useTheme } from '../d3/colors';
import { Tooltip, type Tip } from '../d3/Tooltip';
import type { Model } from '../data/loader';
import type { LeaderMetric } from './Leaderboard';

const ROW = 26;
const TOP = 118;
const SETUP = ['method', 'batch_key', 'n_hvg', 'hvg_batch_aware', 'exclude_igx'];

export default function FunkyHeatmap({ rows, metric }: { rows: Model[]; metric: LeaderMetric }) {
  const { entry } = useData();
  const { manifest } = entry;
  const { state, dispatch } = useSelection();
  const theme = useTheme();
  const ref = useRef<SVGSVGElement>(null);
  const [tip, setTip] = useState<Tip | null>(null);
  const batch = manifest.metrics.filter((m) => m.group === 'batch').map((m) => m.name);
  const bio = manifest.metrics.filter((m) => m.group === 'bio').map((m) => m.name);

  type Col = { key: string; group: string; kind: 'text' | 'bar' | 'circle'; w: number };
  const cols: Col[] = [
    { key: 'model', group: 'setup', kind: 'text', w: 30 },
    ...SETUP.map((f) => ({ key: f, group: 'setup', kind: 'text' as const, w: f === 'method' ? 82 : f === 'batch_key' ? 56 : 44 })),
    { key: 'seed', group: 'setup', kind: 'text', w: 30 },
    { key: 'overall', group: 'overall', kind: 'bar', w: 70 },
    { key: 'batch', group: 'batch', kind: 'bar', w: 60 },
    ...batch.map((b) => ({ key: b, group: 'batch', kind: 'circle' as const, w: 22 })),
    { key: 'bio', group: 'bio', kind: 'bar', w: 60 },
    ...bio.map((b) => ({ key: b, group: 'bio', kind: 'circle' as const, w: 22 })),
  ];
  const width = cols.reduce((a, c) => a + c.w, 0) + 20;
  const height = TOP + rows.length * ROW + 8;

  useEffect(() => {
    const ink = INK[theme];
    const svg = d3.select(ref.current!);
    svg.selectAll('*').remove();
    const seq = sequential(theme, [0, 1]);
    const methodColor = categorical(theme, manifest.factors.find((f) => f.name === 'method')!.levels.map(String));
    let x = 10;
    const colX: Record<string, number> = {};
    cols.forEach((c) => {
      colX[c.key] = x;
      x += c.w;
    });
    // group headers
    for (const g of ['setup', 'overall', 'batch', 'bio']) {
      const gc = cols.filter((c) => c.group === g);
      const x0 = colX[gc[0].key];
      const x1 = colX[gc[gc.length - 1].key] + gc[gc.length - 1].w;
      svg.append('line').attr('x1', x0 + 2).attr('x2', x1 - 4).attr('y1', 16).attr('y2', 16).attr('stroke', ink.axis);
      svg.append('text').attr('class', 'mv-fh-group').attr('x', x0 + 2).attr('y', 12).text(g);
    }
    // column labels
    cols.forEach((c) => {
      svg
        .append('text')
        .attr('class', 'mv-fh-col')
        .attr('transform', `translate(${colX[c.key] + c.w / 2},${TOP - 6}) rotate(-50)`)
        .text(c.key === 'model' ? 'rank' : c.key);
    });
    const selected = state.selectedModels;
    rows.forEach((m, r) => {
      const y = TOP + r * ROW;
      const g = svg.append('g').attr('class', 'mv-fh-row').attr('transform', `translate(0,${y})`).style('cursor', 'pointer');
      if (r % 2 === 0) g.append('rect').attr('x', 6).attr('width', width - 12).attr('height', ROW).attr('fill', ink.grid).attr('opacity', 0.35);
      if (selected.has(m.model_id)) g.append('rect').attr('x', 6).attr('width', width - 12).attr('height', ROW).attr('fill', 'none').attr('stroke', ink.primary).attr('stroke-width', 1.5);
      g.on('click', (ev: MouseEvent) =>
        dispatch(ev.shiftKey || ev.metaKey ? { type: 'toggleModel', id: m.model_id } : { type: 'selectModels', ids: [m.model_id] }),
      );
      for (const c of cols) {
        const cx = colX[c.key];
        if (c.kind === 'text') {
          const val = c.key === 'model' ? String(r + 1) : c.key === 'seed' ? String(m.seed) : String(m.factors[c.key]);
          if (c.key === 'method') g.append('rect').attr('x', cx).attr('y', 8).attr('width', 4).attr('height', 10).attr('rx', 1).attr('fill', methodColor(val));
          g.append('text').attr('class', 'mv-fh-text').attr('x', cx + (c.key === 'method' ? 8 : 2)).attr('y', ROW / 2 + 4).text(val);
        } else if (c.kind === 'bar') {
          const v = (m as unknown as Record<string, number>)[c.key];
          const bw = Math.max(1, (c.w - 12) * v);
          g.append('rect').attr('x', cx).attr('y', 6).attr('width', c.w - 12).attr('height', ROW - 12).attr('fill', ink.grid).attr('rx', 2);
          g.append('rect').attr('x', cx).attr('y', 6).attr('width', bw).attr('height', ROW - 12).attr('fill', seq(v)).attr('rx', 2);
          g.append('rect')
            .attr('x', cx)
            .attr('y', 0)
            .attr('width', c.w)
            .attr('height', ROW)
            .attr('fill', 'transparent')
            .on('pointermove', (ev: PointerEvent) =>
              setTip({ x: ev.clientX, y: ev.clientY, title: m.model_id, rows: [{ label: `${c.key} (seed sd ${fmt(m.seed_sd[c.key])})`, value: fmt(v) }] }),
            )
            .on('pointerleave', () => setTip(null));
        } else {
          const v = m.scaled[c.key];
          g.append('circle')
            .attr('cx', cx + c.w / 2)
            .attr('cy', ROW / 2)
            .attr('r', 2 + 7 * Math.max(0, v))
            .attr('fill', seq(v))
            .attr('stroke', ink.surface)
            .attr('stroke-width', 1);
          g.append('rect')
            .attr('x', cx)
            .attr('y', 0)
            .attr('width', c.w)
            .attr('height', ROW)
            .attr('fill', 'transparent')
            .on('pointermove', (ev: PointerEvent) =>
              setTip({
                x: ev.clientX,
                y: ev.clientY,
                title: `${m.model_id} · ${c.key}`,
                rows: [
                  { label: 'scaled', value: fmt(v) },
                  { label: 'raw', value: fmt(m.raw[c.key]) },
                  { label: 'seed sd (raw)', value: fmt(m.seed_sd[c.key]) },
                ],
              }),
            )
            .on('pointerleave', () => setTip(null));
        }
      }
    });
  }, [rows, theme, state.selectedModels, width]);

  return (
    <div className="mv-table-wrap">
      <svg ref={ref} width={width} height={height} role="img" aria-label={`Funky heatmap of the top ${rows.length} models by ${metric.label}`} />
      <p className="mv-muted mv-small">Bars: aggregate scores (0–1). Circles: scaled individual metrics (size and colour). Hover for raw values and seed sd. Same data as the table view.</p>
      <Tooltip tip={tip} />
    </div>
  );
}
