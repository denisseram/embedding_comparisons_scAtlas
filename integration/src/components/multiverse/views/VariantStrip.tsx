// V5 — Local variant strip: regions × models coloured by how each model structures the region
// (typical / merged / split / joined to a neighbour). Columns in the V3 (factor) order;
// filtered to the selected models when a model selection exists.
import * as d3 from 'd3';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { INK, categorical, fmt, useTheme } from '../d3/colors';
import { Swatches } from '../d3/Legend';
import { Tooltip, type Tip } from '../d3/Tooltip';
import { useWidth } from '../d3/useSize';
import { lexOrder, regionCells, type RegionEffect } from '../compute/selection';

const ROW = 18;
const LEFT = 160;

export default function VariantStrip({ effects }: { effects: RegionEffect[] }) {
  const { entry, cells } = useData();
  const { manifest, models } = entry;
  const { state, dispatch } = useSelection();
  const theme = useTheme();
  const wrap = useRef<HTMLDivElement>(null);
  const ref = useRef<SVGSVGElement>(null);
  const width = useWidth(wrap);
  const [tip, setTip] = useState<Tip | null>(null);
  const reg = cells!.regions;
  const levels = reg.variant_levels;
  const regs = reg.by_mode[state.mode].regions;
  // memoised so tooltip re-renders do not rebuild the SVG (which would swallow clicks)
  const rows = useMemo(() => [...effects].sort((a, b) => b.total - a.total).map((e) => e.region), [effects]);
  const order = useMemo(() => {
    const o = lexOrder(manifest, models);
    return state.selectedModels.size ? o.filter((i) => state.selectedModels.has(models[i].model_id)) : o;
  }, [manifest, models, state.selectedModels]);
  const colW = Math.max(3, Math.min(22, (width - LEFT - 10) / Math.max(order.length, 1)));
  const height = rows.length * ROW + 30;
  const pal = useMemo(() => {
    const cat = categorical(theme, ['merged', 'split', 'joined']);
    return (v: string) => (v === 'typical' ? INK[theme].grid : cat(v));
  }, [theme]);

  useEffect(() => {
    const svg = d3.select(ref.current!);
    svg.selectAll('*').remove();
    rows.forEach((r, i) => {
      const rec = regs[r];
      const y = i * ROW;
      svg
        .append('text')
        .attr('class', 'mv-fh-text')
        .attr('x', 4)
        .attr('y', y + ROW / 2 + 4)
        .style('cursor', 'pointer')
        .style('font-weight', state.selectedRegion === r ? 700 : 400)
        .text(rec.label)
        .on('click', () => dispatch({ type: 'selectRegion', region: r, cells: regionCells(cells!, manifest, state.mode, r) }));
      order.forEach((mi, j) => {
        const v = levels[rec.variants.code[mi]];
        svg
          .append('rect')
          .attr('x', LEFT + j * colW)
          .attr('y', y + 1)
          .attr('width', Math.max(1, colW - 1))
          .attr('height', ROW - 2)
          .attr('fill', pal(v))
          .style('cursor', 'pointer')
          .on('pointermove', (ev: PointerEvent) =>
            setTip({
              x: ev.clientX,
              y: ev.clientY,
              title: `${rec.label} in ${models[mi].model_id}`,
              rows: [
                { label: 'variant', value: v },
                { label: `sub-clusters (modal ${rec.variants.modal_n_sub})`, value: String(rec.variants.n_sub[mi]) },
                { label: 'neighbours outside region', value: fmt(rec.variants.outside_frac[mi], 2) },
                { label: 'most absorbing region', value: rec.variants.joined_to[mi] >= 0 ? `R${rec.variants.joined_to[mi]}` : '–' },
              ],
            }),
          )
          .on('pointerleave', () => setTip(null))
          .on('click', (ev: MouseEvent) =>
            dispatch(ev.shiftKey ? { type: 'toggleModel', id: models[mi].model_id } : { type: 'selectModels', ids: [models[mi].model_id] }),
          );
      });
    });
    svg
      .append('text')
      .attr('class', 'mv-fh-col')
      .attr('x', LEFT)
      .attr('y', rows.length * ROW + 16)
      .text(`${order.length} models in factor order (method → batch_key → n_hvg → hvg_batch_aware → exclude_igx → seed)`);
  }, [rows, order, colW, pal, theme, state.selectedRegion, state.mode, regs]);

  return (
    <div ref={wrap} className="mv-chart">
      <svg ref={ref} width={width} height={height} role="img" aria-label="Local variant strip: regions by models coloured by variant" />
      <div className="mv-legend-row">
        <Swatches items={levels.map((l) => ({ key: l, label: l, color: pal(l) }))} />
        <span className="mv-small mv-muted">
          {state.selectedModels.size ? 'Showing selected models only.' : 'Select models (V0a, V0b, V3, V6) to filter columns.'} Click a column to select a
          model, a row label to select a region.
        </span>
      </div>
      <Tooltip tip={tip} />
    </div>
  );
}
