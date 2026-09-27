// V0a — Model map: 2D layout of all models from A(a,b). Location only; nothing is computed from 2D.
import * as d3 from 'd3';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { INK, fmt, useTheme } from '../d3/colors';
import { attachLasso, inPolygon } from '../d3/lasso';
import { Colorbar, Swatches } from '../d3/Legend';
import { Tooltip, type Tip } from '../d3/Tooltip';
import { useWidth } from '../d3/useSize';
import { resolveColor } from './modelColor';

const H = 440;
const PAD = 22;

export default function ModelMap() {
  const { entry, fracZ } = useData();
  const { manifest, models } = entry;
  const { state, dispatch } = useSelection();
  const theme = useTheme();
  const wrap = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const width = useWidth(wrap);
  const [tip, setTip] = useState<Tip | null>(null);

  const color = useMemo(
    () => resolveColor(state.colorBy, manifest, models, theme, state.mode, fracZ),
    [state.colorBy, manifest, models, theme, state.mode, fracZ],
  );

  const pos = useMemo(() => {
    const pts = models.map((m) => m.layout[state.mode][state.layout]);
    const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0]) as [number, number]).nice().range([PAD, width - PAD]);
    const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1]) as [number, number]).nice().range([H - PAD, PAD]);
    return pts.map((p) => [x(p[0]), y(p[1])] as [number, number]);
  }, [models, state.mode, state.layout, width]);

  // draw
  useEffect(() => {
    const svg = d3.select(svgRef.current!);
    const ink = INK[theme];
    svg.selectAll('g.mv-layer').remove();
    const gLines = svg.insert('g', '.mv-lasso').attr('class', 'mv-layer');
    const byCfg = d3.group(models.map((m, i) => ({ m, i })), (d) => d.m.config_id);
    for (const reps of byCfg.values()) {
      const ps = reps.map((r) => pos[r.i]);
      gLines
        .append('path')
        .attr('d', `M${ps.map((p) => p.join(',')).join('L')}Z`)
        .attr('fill', 'none')
        .attr('stroke', ink.muted)
        .attr('stroke-width', 1)
        .attr('stroke-opacity', 0.55);
    }
    const sel = state.selectedModels;
    const any = sel.size > 0;
    const gPts = svg.insert('g', '.mv-lasso').attr('class', 'mv-layer');
    const order = d3.range(models.length).sort((a, b) => Number(sel.has(models[a].model_id)) - Number(sel.has(models[b].model_id)));
    gPts
      .selectAll('circle')
      .data(order)
      .join('circle')
      .attr('cx', (i) => pos[i][0])
      .attr('cy', (i) => pos[i][1])
      .attr('r', (i) => (sel.has(models[i].model_id) ? 6.5 : 5))
      .attr('fill', (i) => color.color(i))
      .attr('fill-opacity', (i) => (!any || sel.has(models[i].model_id) ? 1 : 0.3))
      .attr('stroke', (i) => (sel.has(models[i].model_id) ? ink.primary : ink.surface))
      .attr('stroke-width', (i) => (sel.has(models[i].model_id) ? 2 : 1.5));
    const ref = models.findIndex((m) => m.model_id === state.referenceModel);
    if (ref >= 0) {
      gPts
        .append('path')
        .attr('d', d3.symbol(d3.symbolDiamond, 90)())
        .attr('transform', `translate(${pos[ref][0]},${pos[ref][1] - 13})`)
        .attr('fill', ink.primary)
        .append('title')
        .text('reference model');
    }
  }, [pos, color, state.selectedModels, state.referenceModel, theme, models]);

  // interaction: nearest-point hover/click + lasso
  const delaunay = useMemo(() => d3.Delaunay.from(pos), [pos]);
  // attach the lasso once; read current positions through refs so re-renders never
  // re-bind the drag behaviour in the middle of a gesture
  const posRef = useRef(pos);
  posRef.current = pos;
  useEffect(() => {
    const svg = svgRef.current!;
    return attachLasso(svg, (poly, ev) => {
      const p = posRef.current;
      const ids = models.filter((_, i) => inPolygon(poly, p[i][0], p[i][1])).map((m) => m.model_id);
      dispatch({ type: 'selectModels', ids, additive: ev?.shiftKey });
    });
  }, [models, dispatch]);

  const nearest = (ev: React.PointerEvent | React.MouseEvent) => {
    const r = svgRef.current!.getBoundingClientRect();
    const x = ev.clientX - r.left;
    const y = ev.clientY - r.top;
    const i = delaunay.find(x, y);
    return Math.hypot(pos[i][0] - x, pos[i][1] - y) <= 14 ? i : -1;
  };

  const onMove = (ev: React.PointerEvent) => {
    const i = nearest(ev);
    if (i < 0) return setTip(null);
    const m = models[i];
    const v = color.value(i);
    const rows = [
      { label: color.label, value: typeof v === 'number' ? fmt(v) + (color.kind === 'continuous' && color.sd?.(i) != null ? ` ± ${fmt(color.sd(i)!)}` : '') : String(v) },
      ...Object.entries(m.factors).map(([k, val]) => ({ label: k, value: String(val) })),
      { label: 'seed', value: String(m.seed) },
      { label: 'overall', value: fmt(m.overall) },
    ];
    setTip({ x: ev.clientX, y: ev.clientY, title: m.config_id, rows });
  };

  const onClick = (ev: React.MouseEvent) => {
    const i = nearest(ev);
    if (i < 0) return;
    if (ev.shiftKey || ev.metaKey) dispatch({ type: 'toggleModel', id: models[i].model_id });
    else dispatch({ type: 'selectModels', ids: [models[i].model_id] });
  };

  return (
    <div ref={wrap} className="mv-chart">
      <svg
        ref={svgRef}
        width={width}
        height={H}
        className="mv-svg mv-lassoable"
        role="img"
        aria-label={`Model map (${state.layout.toUpperCase()} of model agreement), ${models.length} models coloured by ${color.label}`}
        onPointerMove={onMove}
        onPointerLeave={() => setTip(null)}
        onClick={onClick}
      />
      <div className="mv-legend-row">
        {color.kind === 'continuous' ? (
          color.missing ? (
            <span className="mv-muted">{color.missing}</span>
          ) : (
            <Colorbar color={color.scale} domain={color.domain} label={color.label} />
          )
        ) : (
          <Swatches items={color.levels.map((l) => ({ key: l, label: `${color.label} = ${l}`, color: color.colorOf(l) }))} />
        )}
        <span className="mv-muted mv-small">Thin outlines join seed replicates of one configuration (the noise scale). Drag to lasso, shift+click to add, ◆ = reference.</span>
      </div>
      <Tooltip tip={tip} />
    </div>
  );
}

