// V3 — Model agreement matrix A(a,b), ordered lexicographically by factors, with factor strips.
// Matrix body on <canvas> (M² cells), strips and labels in SVG.
import * as d3 from 'd3';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { INK, categorical, fmt, sequential, useTheme } from '../d3/colors';
import { Colorbar, Swatches } from '../d3/Legend';
import { Tooltip, type Tip } from '../d3/Tooltip';
import { useWidth } from '../d3/useSize';
import { lexOrder } from '../compute/selection';

const STRIPS = ['method', 'batch_key', 'n_hvg', 'hvg_batch_aware', 'exclude_igx'];
const STRIP_W = 9;

export default function AgreementMatrix() {
  const { entry } = useData();
  const { manifest, models, agreement } = entry;
  const { state, dispatch } = useSelection();
  const theme = useTheme();
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const width = useWidth(wrap);
  const [tip, setTip] = useState<Tip | null>(null);
  const M = models.length;
  const order = useMemo(() => lexOrder(manifest, models), [manifest, models]);
  const A = agreement[state.mode];
  const offset = STRIPS.length * STRIP_W + 6;
  const size = Math.max(200, Math.min(width - offset - 10, 620));
  const cell = size / M;
  const domain = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (let a = 0; a < M; a++) for (let b = 0; b < M; b++) if (a !== b) (lo = Math.min(lo, A[a * M + b])), (hi = Math.max(hi, A[a * M + b]));
    return [lo, hi] as [number, number];
  }, [A, M]);
  const color = useMemo(() => sequential(theme, domain), [theme, domain]);

  useEffect(() => {
    const cv = canvas.current!;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size * dpr;
    cv.height = size * dpr;
    const ctx = cv.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (let r = 0; r < M; r++) {
      for (let c = 0; c < M; c++) {
        const a = order[r];
        const b = order[c];
        ctx.fillStyle = a === b ? INK[theme].surface : color(A[a * M + b]);
        ctx.fillRect(c * cell, r * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }
    // selected models: outline rows/cols
    const sel = state.selectedModels;
    ctx.strokeStyle = INK[theme].primary;
    ctx.lineWidth = 1;
    order.forEach((mi, pos) => {
      if (!sel.has(models[mi].model_id)) return;
      ctx.strokeRect(0.5, pos * cell + 0.5, size - 1, cell);
      ctx.strokeRect(pos * cell + 0.5, 0.5, cell, size - 1);
    });

    const s = d3.select(svg.current!);
    s.selectAll('*').remove();
    STRIPS.forEach((f, j) => {
      const levels = manifest.factors.find((x) => x.name === f)!.levels.map(String);
      const col = categorical(theme, levels);
      const x0 = j * STRIP_W;
      order.forEach((mi, pos) => {
        const v = String(models[mi].factors[f]);
        s.append('rect').attr('x', x0).attr('y', offset + pos * cell).attr('width', STRIP_W - 1).attr('height', Math.ceil(cell)).attr('fill', col(v));
        s.append('rect').attr('y', x0).attr('x', offset + pos * cell).attr('height', STRIP_W - 1).attr('width', Math.ceil(cell)).attr('fill', col(v));
      });
    });
  }, [A, order, size, cell, color, theme, state.selectedModels, models, manifest, offset, M]);

  const locate = (ev: React.PointerEvent | React.MouseEvent) => {
    const r = canvas.current!.getBoundingClientRect();
    const c = Math.floor((ev.clientX - r.left) / cell);
    const row = Math.floor((ev.clientY - r.top) / cell);
    if (c < 0 || row < 0 || c >= M || row >= M) return null;
    return { a: order[row], b: order[c] };
  };

  return (
    <div ref={wrap} className="mv-chart">
      <div style={{ position: 'relative', width: offset + size, height: offset + size }}>
        <svg ref={svg} width={offset + size} height={offset + size} style={{ position: 'absolute', left: 0, top: 0 }} aria-hidden="true" />
        <canvas
          ref={canvas}
          style={{ position: 'absolute', left: offset, top: offset, width: size, height: size, cursor: 'pointer' }}
          role="img"
          aria-label={`Agreement matrix of ${M} models; mean neighbour change between each pair`}
          onPointerMove={(ev) => {
            const p = locate(ev);
            if (!p) return setTip(null);
            setTip({
              x: ev.clientX,
              y: ev.clientY,
              title: 'A(a, b) — mean Δ over sampled cells',
              rows: [
                { label: 'A', value: fmt(A[p.a * M + p.b]) },
                { label: 'row', value: models[p.a].model_id },
                { label: 'column', value: models[p.b].model_id },
              ],
            });
          }}
          onPointerLeave={() => setTip(null)}
          onClick={(ev) => {
            const p = locate(ev);
            if (p) dispatch({ type: 'selectModels', ids: [models[p.a].model_id, models[p.b].model_id], additive: ev.shiftKey });
          }}
        />
      </div>
      <div className="mv-legend-row">
        <Colorbar color={color} domain={domain} label="A(a,b): mean neighbour change (0 = identical)" width={300} />
        <span className="mv-muted mv-small">Rows and columns ordered by the factors below (outer strip first), then seed. Click a cell to select both models.</span>
      </div>
      <div className="mv-legend-row">
        {STRIPS.map((f) => {
          const levels = manifest.factors.find((x) => x.name === f)!.levels.map(String);
          const col = categorical(theme, levels);
          return <Swatches key={f} items={levels.map((l) => ({ key: l, label: `${f}=${l}`, color: col(l) }))} />;
        })}
      </div>
      <Tooltip tip={tip} />
    </div>
  );
}
