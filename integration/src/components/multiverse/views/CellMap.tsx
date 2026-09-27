// V1 — Fixed cell map: one UMAP of the consensus kNN graph (location only), drawn on <canvas>.
// Quadtree hover, lasso selection (SVG overlay), wheel zoom / pan, hex-binned density above 50k cells.
import * as d3 from 'd3';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { INK, fmt, useTheme } from '../d3/colors';
import { attachLasso, inPolygon } from '../d3/lasso';
import { Colorbar, Swatches } from '../d3/Legend';
import { Tooltip, type Tip } from '../d3/Tooltip';
import { useWidth } from '../d3/useSize';
import { resolveCellColor } from './cellColor';
import type { RegionEffect } from '../compute/selection';

const HEX_THRESHOLD = 50_000;

interface Props {
  colorKey: string;
  height?: number;
  title?: string;
  regionEffects?: RegionEffect[];
  zSel?: Float32Array | null;
  zLabel?: string;
  compact?: boolean;
}

export default function CellMap({ colorKey, height = 480, title, regionEffects, zSel, zLabel, compact }: Props) {
  const { entry, cells } = useData();
  const cd = cells!;
  const { manifest } = entry;
  const { state, dispatch } = useSelection();
  const theme = useTheme();
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<SVGSVGElement>(null);
  const width = useWidth(wrap, 500);
  const [tip, setTip] = useState<Tip | null>(null);
  const [tool, setTool] = useState<'lasso' | 'pan'>('lasso');
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);
  const [highlight, setHighlight] = useState<string | null>(null);
  const N = manifest.dataset.n_cells;

  const color = useMemo(
    () => resolveCellColor(colorKey, cd, manifest, theme, { mode: state.mode, regionTotals: regionEffects?.map((r) => r.total), zSel, zLabel }),
    [colorKey, cd, manifest, theme, state.mode, regionEffects, zSel, zLabel],
  );

  const base = useMemo(() => {
    const x = d3.scaleLinear().domain(d3.extent(cd.cells.x) as [number, number]).range([12, width - 12]);
    const y = d3.scaleLinear().domain(d3.extent(cd.cells.y) as [number, number]).range([height - 12, 12]);
    return { x, y };
  }, [cd, width, height]);

  // screen positions under the current zoom
  const screen = useMemo(() => {
    const px = new Float32Array(N);
    const py = new Float32Array(N);
    for (let c = 0; c < N; c++) {
      px[c] = transform.applyX(base.x(cd.cells.x[c]));
      py[c] = transform.applyY(base.y(cd.cells.y[c]));
    }
    return { px, py };
  }, [base, transform, cd, N]);

  const selectedMask = useMemo(() => {
    if (!state.selectedCells) return null;
    const m = new Uint8Array(N);
    state.selectedCells.forEach((c) => (m[c] = 1));
    return m;
  }, [state.selectedCells, N]);

  const hlMask = useMemo(() => {
    if (!highlight || color.kind !== 'categorical') return null;
    const code = color.levels.indexOf(highlight);
    const m = new Uint8Array(N);
    for (let c = 0; c < N; c++) m[c] = color.code(c) === code ? 1 : 0;
    return m;
  }, [highlight, color, N]);

  // draw
  useEffect(() => {
    const t0 = performance.now();
    const cv = canvas.current!;
    const dpr = window.devicePixelRatio || 1;
    cv.width = width * dpr;
    cv.height = height * dpr;
    const ctx = cv.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = INK[theme].surface;
    ctx.fillRect(0, 0, width, height);
    const { px, py } = screen;
    const dim = INK[theme].neutral;
    const focus = selectedMask ?? hlMask;
    if (N > HEX_THRESHOLD) {
      drawHex(ctx, px, py, color, focus, dim, width, height);
    } else {
      const r = compact ? 1.6 : 2.2;
      const pass = (want: boolean) => {
        for (let c = 0; c < N; c++) {
          const inFocus = !focus || focus[c] === 1;
          if (inFocus !== want) continue;
          ctx.fillStyle = inFocus ? color.color(c) : dim;
          ctx.globalAlpha = inFocus ? 0.9 : 0.25;
          ctx.fillRect(px[c] - r, py[c] - r, 2 * r, 2 * r);
        }
      };
      pass(false);
      pass(true);
      ctx.globalAlpha = 1;
    }
    // region labels (direct labels at region centroids)
    if (colorKey === 'region' || state.selectedRegion !== null) {
      const reg = cd.measures.subarray(cd.measureIndex[`region.${state.mode}`] * N, (cd.measureIndex[`region.${state.mode}`] + 1) * N);
      const acc = new Map<number, [number, number, number]>();
      for (let c = 0; c < N; c++) {
        const a = acc.get(reg[c]) ?? [0, 0, 0];
        a[0] += px[c];
        a[1] += py[c];
        a[2]++;
        acc.set(reg[c], a);
      }
      ctx.font = '600 11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      for (const [r, [sx, sy, n]] of acc) {
        if (colorKey !== 'region' && r !== state.selectedRegion) continue;
        ctx.lineWidth = 3;
        ctx.strokeStyle = INK[theme].surface;
        ctx.strokeText(`R${r}`, sx / n, sy / n);
        ctx.fillStyle = INK[theme].primary;
        ctx.fillText(`R${r}`, sx / n, sy / n);
      }
    }
    const dt = performance.now() - t0;
    if (dt > 100) console.info(`[multiverse] cell map redraw ${dt.toFixed(0)} ms`);
  }, [screen, color, selectedMask, hlMask, width, height, theme, N, compact, colorKey, state.selectedRegion, state.mode, cd]);

  // hover via quadtree in screen space
  const tree = useMemo(
    () =>
      d3
        .quadtree<number>()
        .x((c) => screen.px[c])
        .y((c) => screen.py[c])
        .addAll(d3.range(N)),
    [screen, N],
  );

  // lasso / zoom: bound once per tool; current tree/positions/selection are read through refs
  // so re-renders (hover tooltips, recolouring) never re-bind mid-gesture
  const live = useRef({ tree, screen, selected: state.selectedCells, transform });
  live.current = { tree, screen, selected: state.selectedCells, transform };
  useEffect(() => {
    const svg = overlay.current!;
    if (tool === 'lasso') {
      return attachLasso(svg, (poly, ev) => {
        const { tree: t, screen: sc, selected } = live.current;
        const [x0, y0] = [d3.min(poly, (p) => p[0])!, d3.min(poly, (p) => p[1])!];
        const [x1, y1] = [d3.max(poly, (p) => p[0])!, d3.max(poly, (p) => p[1])!];
        const hits: number[] = [];
        t.visit((node, ax, ay, bx, by) => {
          if (!node.length) {
            let n: d3.QuadtreeLeaf<number> | undefined = node as d3.QuadtreeLeaf<number>;
            do {
              const c = n.data;
              if (inPolygon(poly, sc.px[c], sc.py[c])) hits.push(c);
            } while ((n = n.next));
          }
          return ax > x1 || bx < x0 || ay > y1 || by < y0;
        });
        let sel = Uint32Array.from(hits);
        if (ev?.shiftKey && selected) sel = Uint32Array.from(new Set([...selected, ...hits]));
        dispatch({ type: 'selectCells', cells: sel });
      });
    }
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 40])
      .on('zoom', (ev) => setTransform(ev.transform));
    const s = d3.select(svg).call(zoom);
    s.call(zoom.transform, live.current.transform);
    return () => {
      s.on('.zoom', null);
    };
  }, [tool, dispatch]);

  const onMove = (ev: React.PointerEvent) => {
    const r = overlay.current!.getBoundingClientRect();
    const c = tree.find(ev.clientX - r.left, ev.clientY - r.top, 8);
    if (c === undefined) return setTip(null);
    const cat = cd.cells.categorical;
    const rows = [
      { label: color.label, value: color.kind === 'categorical' ? color.levels[color.code(c)] : fmt(color.value(c)) },
      { label: 'cell type', value: cat.cell_type.levels[cat.cell_type.codes[c]] },
      { label: 'sample / study', value: `${cat.sample.levels[cat.sample.codes[c]]} / ${cat.study.levels[cat.study.codes[c]]}` },
      { label: 'planted effect', value: cat.planted_effect.levels[cat.planted_effect.codes[c]] },
    ];
    setTip({ x: ev.clientX, y: ev.clientY, title: cd.cells.id[c], rows });
  };

  return (
    <div ref={wrap} className="mv-chart">
      {!compact && (
        <div className="mv-toolbar">
          <span className="mv-small mv-muted">{title}</span>
          <div className="mv-seg" role="group" aria-label="Pointer tool">
            <button type="button" aria-pressed={tool === 'lasso'} onClick={() => setTool('lasso')}>Lasso</button>
            <button type="button" aria-pressed={tool === 'pan'} onClick={() => setTool('pan')}>Pan / zoom</button>
            <button type="button" onClick={() => setTransform(d3.zoomIdentity)}>Reset zoom</button>
          </div>
        </div>
      )}
      <div style={{ position: 'relative', width, height }}>
        <canvas ref={canvas} style={{ width, height, display: 'block' }} role="img" aria-label={`Cell map of ${N} cells coloured by ${color.label}`} />
        <svg
          ref={overlay}
          width={width}
          height={height}
          className={`mv-svg ${tool === 'lasso' && !compact ? 'mv-lassoable' : ''}`}
          style={{ position: 'absolute', inset: 0, pointerEvents: compact ? 'none' : 'auto' }}
          onPointerMove={onMove}
          onPointerLeave={() => setTip(null)}
        />
      </div>
      <div className="mv-legend-row">
        {color.kind === 'categorical' ? (
          <Swatches
            items={color.levels.map((l) => ({ key: l, label: l, color: color.colorOf(l) }))}
            active={highlight}
            onPick={(k) => setHighlight((h) => (h === k ? null : k))}
          />
        ) : color.note ? (
          <span className="mv-muted mv-small">{color.note}</span>
        ) : (
          <Colorbar color={color.scale} domain={color.domain} label={color.label} width={compact ? 180 : 240} />
        )}
      </div>
      {!compact && <Tooltip tip={tip} />}
    </div>
  );
}

// density fallback for large N: pointy-top hex bins; colour = mean value / majority category
function drawHex(
  ctx: CanvasRenderingContext2D,
  px: Float32Array,
  py: Float32Array,
  color: ReturnType<typeof resolveCellColor>,
  focus: Uint8Array | null,
  dim: string,
  width: number,
  height: number,
) {
  const R = 4;
  const w = Math.sqrt(3) * R;
  const h = 1.5 * R;
  const bins = new Map<string, { n: number; sum: number; counts: Map<number, number>; x: number; y: number; f: number }>();
  for (let c = 0; c < px.length; c++) {
    const row = Math.round(py[c] / h);
    const col = Math.round((px[c] - (row % 2 ? w / 2 : 0)) / w);
    const key = `${row},${col}`;
    let b = bins.get(key);
    if (!b) bins.set(key, (b = { n: 0, sum: 0, counts: new Map(), x: col * w + (row % 2 ? w / 2 : 0), y: row * h, f: 0 }));
    b.n++;
    if (!focus || focus[c]) b.f++;
    if (color.kind === 'continuous') b.sum += color.value(c);
    else b.counts.set(color.code(c), (b.counts.get(color.code(c)) ?? 0) + 1);
  }
  const maxN = Math.max(...Array.from(bins.values(), (b) => b.n));
  for (const b of bins.values()) {
    if (b.x < -w || b.y < -h || b.x > width + w || b.y > height + h) continue;
    let fill: string;
    if (color.kind === 'continuous') fill = color.scale(b.sum / b.n);
    else {
      const top = [...b.counts.entries()].sort((a, z) => z[1] - a[1])[0][0];
      fill = color.colorOf(color.levels[top]);
    }
    ctx.globalAlpha = focus && b.f === 0 ? 0.2 : 0.35 + 0.65 * Math.sqrt(b.n / maxN);
    ctx.fillStyle = focus && b.f === 0 ? dim : fill;
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (Math.PI / 3) * i + Math.PI / 6;
      ctx.lineTo(b.x + R * Math.cos(a), b.y + R * Math.sin(a));
    }
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
