// V4 — Regions × decisions fingerprint. Colour = mean E over the region's cells (diverging),
// dot size = mean H (interaction: how much the effect varies across matched pairs).
import * as d3 from 'd3';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { INK, diverging, fmt, useTheme } from '../d3/colors';
import { Colorbar } from '../d3/Legend';
import { Tooltip, type Tip } from '../d3/Tooltip';
import { regionCells, type RegionEffect } from '../compute/selection';

const ROW = 24;
const COL = 64;
const LEFT = 190;
const TOP = 64;

export default function Fingerprint({ effects, relative, setRelative }: { effects: RegionEffect[]; relative: boolean; setRelative: (v: boolean) => void }) {
  const { entry, cells } = useData();
  const { manifest } = entry;
  const { state, dispatch } = useSelection();
  const theme = useTheme();
  const ref = useRef<SVGSVGElement>(null);
  const [tip, setTip] = useState<Tip | null>(null);
  const F = manifest.measure_factors;
  const labels = cells!.regions.by_mode[state.mode].regions;
  // memoised so tooltip re-renders do not rebuild the SVG (which would swallow clicks)
  const rows = useMemo(() => [...effects].sort((a, b) => b.total - a.total), [effects]);
  const maxAbs = useMemo(() => Math.max(...effects.flatMap((e) => e.E.map(Math.abs)), 0.5), [effects]);
  const maxH = useMemo(() => Math.max(...effects.flatMap((e) => e.H), 0.1), [effects]);
  const color = useMemo(() => diverging(theme, maxAbs), [theme, maxAbs]);
  const width = LEFT + F.length * COL + 10;
  const height = TOP + rows.length * ROW + 6;

  useEffect(() => {
    const ink = INK[theme];
    const svg = d3.select(ref.current!);
    svg.selectAll('*').remove();
    F.forEach((f, j) => {
      svg
        .append('text')
        .attr('class', 'mv-fh-col')
        .attr('transform', `translate(${LEFT + j * COL + COL / 2},${TOP - 8}) rotate(-35)`)
        .text(f === 'seed' ? 'seed (null)' : f);
    });
    rows.forEach((r, i) => {
      const y = TOP + i * ROW;
      const sel = state.selectedRegion === r.region;
      const g = svg.append('g').attr('transform', `translate(0,${y})`).style('cursor', 'pointer');
      g.on('click', () =>
        dispatch(sel ? { type: 'selectRegion', region: null } : { type: 'selectRegion', region: r.region, cells: regionCells(cells!, manifest, state.mode, r.region) }),
      );
      if (sel) g.append('rect').attr('x', 2).attr('y', 1).attr('width', width - 4).attr('height', ROW - 2).attr('fill', 'none').attr('stroke', ink.primary).attr('stroke-width', 1.5).attr('rx', 3);
      g.append('text')
        .attr('class', 'mv-fh-text')
        .attr('x', 6)
        .attr('y', ROW / 2 + 4)
        .text(`${labels[r.region]?.label ?? `R${r.region}`} · n=${r.size}`);
      F.forEach((f, j) => {
        const x = LEFT + j * COL;
        g.append('rect').attr('x', x + 1).attr('y', 2).attr('width', COL - 2).attr('height', ROW - 4).attr('rx', 2).attr('fill', color(r.E[j]));
        g.append('circle')
          .attr('cx', x + COL / 2)
          .attr('cy', ROW / 2)
          .attr('r', 1.5 + 6 * Math.min(1, r.H[j] / maxH))
          .attr('fill', 'none')
          .attr('stroke', ink.primary)
          .attr('stroke-width', 1.2);
        g.append('rect')
          .attr('x', x)
          .attr('width', COL)
          .attr('height', ROW)
          .attr('fill', 'transparent')
          .on('pointermove', (ev: PointerEvent) =>
            setTip({
              x: ev.clientX,
              y: ev.clientY,
              title: `R${r.region} × ${f}`,
              rows: [
                { label: relative ? 'mean E − factor median' : 'mean E (z units of seed noise)', value: fmt(r.E[j], 2) },
                { label: 'mean H (sd over matched pairs)', value: fmt(r.H[j], 2) },
                { label: 'cells', value: String(r.size) },
              ],
            }),
          )
          .on('pointerleave', () => setTip(null));
      });
    });
  }, [rows, theme, state.selectedRegion, state.mode, relative, color, maxH, width]);

  return (
    <div className="mv-table-wrap">
      <div className="mv-toolbar">
        <label className="mv-small">
          <input type="checkbox" checked={relative} onChange={(e) => setRelative(e.target.checked)} /> Show E relative to each factor's median over all
          cells (display transform, not validated)
        </label>
      </div>
      <svg ref={ref} width={width} height={height} role="img" aria-label="Regions by decisions fingerprint: mean effect per region and factor" />
      <div className="mv-legend-row">
        <Colorbar color={color} domain={[-maxAbs, maxAbs]} label={relative ? 'mean E − factor median' : 'mean E (seed-noise z)'} width={220} />
        <span className="mv-small mv-muted">Ring size = mean H. Rows sorted by total |E| (seed excluded). Click a row to select the region.</span>
      </div>
      <Tooltip tip={tip} />
    </div>
  );
}
