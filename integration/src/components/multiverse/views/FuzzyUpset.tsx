// Comparative fuzzy UpSet: which labels of one column mix in each embedding's kNN graph, and how that mixing
// differs across embeddings. Columns = intersections; bars = fuzzy size in the focused model; heatmap strip =
// fuzzy size in every compared model; dot matrix = labels involved; attribute rows = composition by a second
// column and mean QC. Clicking a column selects its cells (focused model) in every linked view.
import * as d3 from 'd3';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { INK, categorical, fmt, sequential, useTheme } from '../d3/colors';
import { Swatches } from '../d3/Legend';
import { Tooltip, type Tip } from '../d3/Tooltip';
import { loadFuzzy, loadFuzzyRegions, loadModelUmaps, modelUmap, type ModelUmaps } from '../data/loader';
import {
  attributes,
  cachedSignatures,
  cellsFor,
  compareModels,
  difference,
  DIFFUSE_KEY,
  regionSignatures,
  type DiffKind,
  type FuzzyInfo,
  type Intersection,
} from '../compute/fuzzyUpset';
import CellMap from './CellMap';

const COL = 22;
const LEFT = 250;
const BAR_H = 120;
const DOT_ROW = 16;
const ATTR_ROW = 14;
const MAX_COLS = 60;

/** Funky-heatmap glyph for a fuzzy size: area grows with the value, and the shape goes from a small circle
 * (small values) to a full square (the largest value in the view). null for 0. */
function fuzzyGlyph(v: number, max: number, box: number): { s: number; r: number } | null {
  if (!(v > 0) || !(max > 0)) return null;
  const t = Math.min(1, v / max);
  const s = 2.5 + (box - 2.5) * Math.sqrt(t);
  return { s, r: (s / 2) * (1 - t) };
}

function GlyphLegend({ max, color }: { max: number; color: (v: number) => string }) {
  const steps = [0.03, 0.2, 0.5, 1].map((t) => t * max);
  const box = 16;
  return (
    <svg width={4 * 46 + 10} height={46} role="img" aria-label={`Glyph size and colour for fuzzy size, 0 to ${max.toFixed(0)}`}>
      <text className="mv-legend-title" x={4} y={11}>
        glyph: fuzzy size (area and colour)
      </text>
      {steps.map((v, i) => {
        const g = fuzzyGlyph(v, max, box)!;
        const cx = 16 + i * 46;
        return (
          <g key={i}>
            <rect x={cx - g.s / 2} y={24 - g.s / 2} width={g.s} height={g.s} rx={g.r} fill={color(v)} />
            <text className="mv-fh-col" x={cx} y={43} textAnchor="middle">
              {v >= 10 ? Math.round(v) : v.toFixed(1)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

let countsCache: Promise<Uint8Array | null> | null = null;
let regionsCache: Promise<Int16Array | null> | null = null;
let umapCache: Promise<ModelUmaps | null> | null = null;

type SortKey = 'difference' | 'fuzzy' | 'n_cells';
type Mode = 'cell' | 'region';

export default function FuzzyUpset() {
  const { entry, cells, cellsStatus, cellsError, requestCells } = useData();
  const { manifest } = entry;
  const info = manifest.fuzzy_upset;
  const [counts, setCounts] = useState<Uint8Array | null | undefined>(undefined);
  const [regions, setRegions] = useState<Int16Array | null>(null);
  const [regionsError, setRegionsError] = useState<string | null>(null);
  const [umaps, setUmaps] = useState<ModelUmaps | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (cellsStatus === 'idle') requestCells();
    if (!info) return;
    countsCache ??= loadFuzzy(manifest);
    countsCache.then(setCounts).catch((e: Error) => {
      countsCache = null;
      setError(e.message);
    });
    regionsCache ??= loadFuzzyRegions(manifest);
    regionsCache.then(setRegions).catch((e: Error) => {
      regionsCache = null;
      setRegionsError(e.message); // per-cell mode still works
    });
    umapCache ??= loadModelUmaps(manifest);
    umapCache.then(setUmaps).catch(() => {
      umapCache = null;
      setUmaps(null); // the UpSet works without the linked UMAP
    });
  }, [cellsStatus, requestCells, manifest, info]);

  if (!info)
    return (
      <p className="mv-muted">
        This export has no fuzzy UpSet data. Add a <code>fuzzy_upset:</code> block to config.yaml and run <code>make export-fuzzy</code>.
      </p>
    );
  if (error || cellsStatus === 'error')
    return (
      <div className="mv-error" role="alert">
        <strong>Could not load the label-mixing data.</strong> {error ?? cellsError}
      </div>
    );
  if (!counts || !cells || umaps === undefined) return <p className="mv-loading" role="status">Loading neighbour-label memberships (≈ 0.8 MB) and cell annotations…</p>;
  return <UpsetPanel info={info} counts={counts} regions={regions} regionsError={regionsError} umaps={umaps} />;
}

function UpsetPanel({
  info,
  counts,
  regions,
  regionsError,
  umaps,
}: {
  info: FuzzyInfo;
  counts: Uint8Array;
  regions: Int16Array | null;
  regionsError: string | null;
  umaps: ModelUmaps | null;
}) {
  const { entry, cells } = useData();
  const cd = cells!;
  const { manifest, models } = entry;
  const { state, dispatch } = useSelection();
  const theme = useTheme();
  const N = manifest.dataset.n_cells;
  const configOf = useMemo(() => new Map(models.map((m) => [m.model_id, m.config_id])), [models]);
  const short = (id: string) => configOf.get(id) ?? id;

  // ---- defaults: the reference setup (seed 0) with every method x batch key ----------------------------
  const defaults = useMemo(() => {
    const ref = state.referenceModel;
    const rest = ref.slice(ref.indexOf('.')).replace(/\.s\d+$/, '.s0');
    const focus = `${ref.slice(0, ref.indexOf('.'))}${rest}`;
    const methods = manifest.factors.find((f) => f.name === 'method')!.levels.map(String);
    const bks = manifest.factors.find((f) => f.name === 'batch_key')!.levels.map(String);
    const bk = String(models.find((m) => m.model_id === ref)?.factors.batch_key ?? '');
    const set = methods.flatMap((m) => bks.map((b) => `${m}${rest.replace(`.${bk}.`, `.${b}.`)}`)).filter((id) => info.models.includes(id));
    return { focus: info.models.includes(focus) ? focus : info.models[0], compared: set.length >= 2 ? set : info.models.slice(0, 6) };
  }, [state.referenceModel, manifest, models, info]);

  const [colIdx, setColIdx] = useState(0);
  const [mode, setMode] = useState<Mode>('cell');
  const regionTaus = info.regions?.taus ?? [];
  const [regionTauIdx, setRegionTauIdx] = useState(Math.max(0, regionTaus.indexOf(info.defaults.tau)));
  const regionMode = mode === 'region' && !!regions;
  // raw fractions and enrichment live on different scales, so each keeps its own threshold
  const [tauByMode, setTauByMode] = useState({ raw: info.defaults.tau, enrich: 1 });
  const [maxSize, setMaxSize] = useState(info.defaults.max_size);
  const [normalize, setNormalize] = useState(info.defaults.normalize);
  const tau = normalize ? tauByMode.enrich : tauByMode.raw;
  const setTau = (v: number) => setTauByMode((t) => (normalize ? { ...t, enrich: v } : { ...t, raw: v }));
  const [focus, setFocus] = useState(defaults.focus);
  const [compared, setCompared] = useState<string[]>(defaults.compared);
  const [second, setSecond] = useState(info.second_default ?? info.second_columns[0]);
  const [sortKey, setSortKey] = useState<SortKey>('difference');
  const [diffKind, setDiffKind] = useState<DiffKind>('range');
  const [minSize, setMinSize] = useState(1);
  const [showPure, setShowPure] = useState(!info.defaults.hide_pure);
  const [search, setSearch] = useState('');
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [tip, setTip] = useState<Tip | null>(null);
  useEffect(() => {
    setFocus(defaults.focus);
    setCompared(defaults.compared);
  }, [defaults]);
  useEffect(() => {
    if (!state.selectedCells) setActiveKey(null);
  }, [state.selectedCells]);

  const col = info.columns[colIdx];
  // the focused model is always part of the comparison (first row)
  const rowsModels = useMemo(() => [focus, ...compared.filter((m) => m !== focus)], [focus, compared]);
  const canLogRatio = rowsModels.length === 2;
  const diff: DiffKind = diffKind === 'log_ratio' && !canLogRatio ? 'range' : diffKind;

  const cmp = useMemo(() => {
    if (regionMode) {
      const per = rowsModels.map((id) => regionSignatures(regions!, counts, info, N, info.models.indexOf(id), colIdx, regionTauIdx));
      return { ...compareModels(per), nRegions: per[0].nRegions };
    }
    const p = { tau, maxSize, normalize };
    const per = rowsModels.map((id) => cachedSignatures(counts, info, N, info.models.indexOf(id), col, p));
    return { ...compareModels(per, true), nRegions: 0 };
  }, [counts, regions, regionMode, regionTauIdx, info, N, col, colIdx, tau, maxSize, normalize, rowsModels]);
  const focusSig = cmp.perModel[0];

  // composition column: never the label column itself
  const secondOpts = info.second_columns.filter((c) => c !== col.name && cd.cells.categorical[c]);
  const secondName = secondOpts.includes(second) ? second : secondOpts[0];
  const secondCol = secondName ? cd.cells.categorical[secondName] : undefined;
  const qcCols = info.qc_columns.filter((q) => cd.cells.numeric[q]);
  const attrs = useMemo(
    () => attributes(focusSig, secondCol?.codes ?? null, secondCol?.levels.length ?? 0, qcCols.map((q) => cd.cells.numeric[q])),
    [focusSig, secondCol, qcCols.join(), cd],
  );

  // ---- filter + sort ------------------------------------------------------------------------------------
  const q = search.trim().toLowerCase();
  const { shown, total } = useMemo(() => {
    const score = (it: Intersection) =>
      sortKey === 'fuzzy' ? it.fuzzy[0] : sortKey === 'n_cells' ? it.nCells[0] : Math.abs(difference(it, diff, [0, 1]));
    // regions are mixed by construction, so the pure filter does not apply to them; diffuse is never "pure"
    const keep = cmp.intersections.filter(
      (it) =>
        (regionMode || showPure || it.labels.length !== 1) &&
        Math.max(...it.fuzzy) >= minSize &&
        (!q || (it.key === DIFFUSE_KEY ? 'diffuse'.includes(q) : it.labels.some((l) => col.levels[l].toLowerCase().includes(q)))),
    );
    const sorted = keep.map((it) => ({ it, s: score(it) })).sort((a, b) => b.s - a.s).map((x) => x.it);
    return { shown: sorted.slice(0, MAX_COLS), total: sorted.length };
  }, [cmp, regionMode, showPure, minSize, q, col, sortKey, diff]);

  const maxFuzzy = useMemo(() => Math.max(1, ...shown.flatMap((it) => Array.from(it.fuzzy))), [shown]);
  const heat = useMemo(() => sequential(theme, [0, maxFuzzy]), [theme, maxFuzzy]);
  const barColor = categorical(theme, ['bar'])('bar'); // single series: first categorical hue, no legend
  const compColor = useMemo(() => categorical(theme, secondCol?.levels ?? []), [theme, secondCol]);
  const qcRange = useMemo(
    () =>
      qcCols.map((_, i) => {
        const v = shown.map((it) => attrs.qcMean.get(it.key)?.[i] ?? NaN).filter(Number.isFinite);
        return v.length ? ([Math.min(...v), Math.max(...v)] as [number, number]) : ([0, 1] as [number, number]);
      }),
    [shown, attrs, qcCols.join()],
  );

  const select = (it: Intersection, mIdx = 0) => {
    if (mIdx !== 0) setFocus(rowsModels[mIdx]);
    const key = it.key;
    if (activeKey === key && mIdx === 0) {
      setActiveKey(null);
      dispatch({ type: 'selectCells', cells: null });
      return;
    }
    setActiveKey(key);
    dispatch({ type: 'selectCells', cells: cellsFor(cmp.perModel[mIdx], key) });
  };
  const names = (it: Intersection) =>
    it.key === DIFFUSE_KEY
      ? `diffuse: more than ${maxSize} labels ≥ τ`
      : (regionMode ? 'mixed region(s): ' : '') + (it.labels.map((l) => col.levels[l]).join(' & ') || 'no label ≥ τ on average');

  // ---- SVG ----------------------------------------------------------------------------------------------
  const ref = useRef<SVGSVGElement>(null);
  const R = rowsModels.length;
  const heatRow = R > 12 ? 12 : 18;
  const L = col.levels.length;
  const yBars = 10;
  const yHeat = yBars + BAR_H + 8;
  const yDots = yHeat + R * heatRow + 12;
  const yComp = yDots + L * DOT_ROW + 12;
  const yQc = yComp + 44 + 8;
  const height = yQc + qcCols.length * ATTR_ROW + 8;
  const width = LEFT + Math.max(shown.length, 1) * COL + 10;

  useEffect(() => {
    const ink = INK[theme];
    const svg = d3.select(ref.current!);
    svg.selectAll('*').remove();
    const y = d3.scaleLinear().domain([0, maxFuzzy]).nice().range([yBars + BAR_H, yBars]);
    // left labels
    const lab = (yy: number, text: string, cls = 'mv-fh-text') =>
      svg.append('text').attr('class', cls).attr('x', LEFT - 8).attr('y', yy).attr('text-anchor', 'end').text(text);
    svg.append('g').attr('class', 'mv-axis').attr('transform', `translate(${LEFT - 2},0)`).call(d3.axisLeft(y).ticks(4).tickSize(3));
    lab(yBars + BAR_H / 2, 'fuzzy size, focused model', 'mv-fh-col').attr('x', LEFT - 34);
    rowsModels.forEach((id, j) => {
      const t = lab(yHeat + j * heatRow + heatRow - 3, (j === 0 ? '▶ ' : '') + short(id), 'mv-fh-col');
      if (j === 0) t.style('font-weight', '700');
      if (heatRow < 14) t.style('font-size', '9px');
    });
    col.levels.forEach((lv, l) => {
      if (l % 2 === 0)
        svg.append('rect').attr('x', LEFT).attr('y', yDots + l * DOT_ROW).attr('width', width - LEFT - 10).attr('height', DOT_ROW).attr('fill', ink.mid).attr('opacity', 0.6);
      lab(yDots + l * DOT_ROW + DOT_ROW - 4, lv);
    });
    lab(yComp + 26, `composition by ${secondName}`, 'mv-fh-col');
    qcCols.forEach((qn, i) => lab(yQc + i * ATTR_ROW + ATTR_ROW - 3, `mean ${qn}`, 'mv-fh-col'));

    shown.forEach((it, i) => {
      const x = LEFT + i * COL;
      const cx = x + COL / 2;
      const g = svg.append('g').style('cursor', 'pointer');
      const active = activeKey === it.key;
      if (active)
        g.append('rect').attr('x', x + 1).attr('y', 2).attr('width', COL - 2).attr('height', height - 4).attr('rx', 3).attr('fill', 'none').attr('stroke', ink.primary).attr('stroke-width', 1.5);
      // bar (focused model): 4px rounded data end, square at the baseline
      const v = it.fuzzy[0];
      const bw = Math.min(COL - 6, 16);
      const top = y(v);
      const h = yBars + BAR_H - top;
      if (h > 0) {
        const r = Math.min(4, h, bw / 2);
        g.append('path')
          .attr('d', `M${cx - bw / 2},${yBars + BAR_H}V${top + r}q0,${-r} ${r},${-r}H${cx + bw / 2 - r}q${r},0 ${r},${r}V${yBars + BAR_H}Z`)
          .attr('fill', barColor);
      }
      // strip: one funky-heatmap glyph per compared model on a faint tile (the tile is the hit target)
      rowsModels.forEach((id, j) => {
        g.append('rect')
          .attr('x', x + 1)
          .attr('y', yHeat + j * heatRow + 1)
          .attr('width', COL - 2)
          .attr('height', heatRow - 2)
          .attr('rx', 2)
          .attr('fill', ink.mid)
          .attr('opacity', 0.7)
          .on('click', (ev: MouseEvent) => {
            ev.stopPropagation();
            select(it, j);
          })
          .on('pointermove', (ev: PointerEvent) => {
            ev.stopPropagation();
            setTip({
              x: ev.clientX,
              y: ev.clientY,
              title: `${names(it)} · ${id}`,
              rows: [
                { label: 'fuzzy size (sum of strengths)', value: fmt(it.fuzzy[j], 2) },
                { label: 'cells', value: String(it.nCells[j]) },
                { label: 'mean strength', value: fmt(it.nCells[j] ? it.fuzzy[j] / it.nCells[j] : 0, 3) },
                { label: 'click: focus this model and select its cells', value: '' },
              ],
            });
          })
          .on('pointerleave', () => setTip(null));
      });
      rowsModels.forEach((_, j) => {
        const v = it.fuzzy[j];
        const glyph = fuzzyGlyph(v, maxFuzzy, Math.min(COL, heatRow) - 2);
        if (!glyph) return;
        g.append('rect')
          .attr('x', cx - glyph.s / 2)
          .attr('y', yHeat + j * heatRow + heatRow / 2 - glyph.s / 2)
          .attr('width', glyph.s)
          .attr('height', glyph.s)
          .attr('rx', glyph.r)
          .attr('fill', heat(v))
          .style('pointer-events', 'none');
      });
      // dot matrix
      const member = new Set(it.labels);
      const ys = it.labels.map((l) => yDots + l * DOT_ROW + DOT_ROW / 2);
      if (ys.length > 1) g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', Math.min(...ys)).attr('y2', Math.max(...ys)).attr('stroke', ink.primary).attr('stroke-width', 2);
      if (it.key === DIFFUSE_KEY) {
        // no fixed label set: name the column inside the dot matrix
        g.append('text')
          .attr('class', 'mv-fh-col')
          .attr('transform', `translate(${cx + 4},${yDots + (L * DOT_ROW) / 2}) rotate(-90)`)
          .attr('text-anchor', 'middle')
          .style('font-weight', '700')
          .text(`> ${maxSize} labels`);
      } else
        for (let l = 0; l < L; l++)
          g.append('circle')
            .attr('cx', cx)
            .attr('cy', yDots + l * DOT_ROW + DOT_ROW / 2)
            .attr('r', 4.5)
            .attr('fill', member.has(l) ? ink.primary : ink.grid)
            .attr('stroke', member.has(l) ? ink.surface : 'none')
            .attr('stroke-width', 2);
      // composition (stacked, 2px surface gaps)
      const comp = attrs.composition.get(it.key);
      if (comp && secondCol) {
        let acc = 0;
        const hTot = 44;
        comp.forEach((f, gIdx) => {
          if (f <= 0) return;
          const hh = f * hTot;
          g.append('rect')
            .attr('x', x + 3)
            .attr('y', yComp + acc + 1)
            .attr('width', COL - 6)
            .attr('height', Math.max(hh - 2, 0.5))
            .attr('fill', compColor(secondCol.levels[gIdx]));
          acc += hh;
        });
      }
      // QC rows (colour normalised per row over the shown intersections; exact values in the tooltip)
      const qm = attrs.qcMean.get(it.key);
      qcCols.forEach((_, k) => {
        const val = qm?.[k] ?? NaN;
        const [lo, hi] = qcRange[k];
        g.append('rect')
          .attr('x', x + 1)
          .attr('y', yQc + k * ATTR_ROW + 1)
          .attr('width', COL - 2)
          .attr('height', ATTR_ROW - 2)
          .attr('rx', 2)
          .attr('fill', Number.isFinite(val) ? sequential(theme, [lo, hi])(val) : ink.grid);
      });
      // hit target for the whole column except the heatmap strip
      g.insert('rect', ':first-child').attr('x', x).attr('y', 0).attr('width', COL).attr('height', height).attr('fill', 'transparent');
      g.on('click', () => select(it)).on('pointermove', (ev: PointerEvent) => {
        const rows = [
          { label: `fuzzy size in ${short(focus)}`, value: fmt(it.fuzzy[0], 2) },
          { label: 'cells', value: String(it.nCells[0]) },
          { label: 'mean strength', value: fmt(it.nCells[0] ? it.fuzzy[0] / it.nCells[0] : 0, 3) },
          { label: diff === 'log_ratio' ? `log2 ratio ${short(rowsModels[0])} / ${short(rowsModels[1])}` : diff === 'var' ? 'variance across models' : 'max − min across models', value: fmt(difference(it, diff, [0, 1]), 2) },
        ];
        if (comp && secondCol)
          Array.from(comp)
            .map((f, gi) => [f, gi] as const)
            .filter(([f]) => f > 0)
            .sort((a, b) => b[0] - a[0])
            .slice(0, 4)
            .forEach(([f, gi]) => rows.push({ label: `${secondName} = ${secondCol.levels[gi]}`, value: `${(100 * f).toFixed(0)}%` }));
        qcCols.forEach((qn, k) => rows.push({ label: `mean ${qn}`, value: fmt(qm?.[k], 3) }));
        setTip({ x: ev.clientX, y: ev.clientY, title: names(it), rows });
      });
      g.on('pointerleave', () => setTip(null));
    });
  }, [shown, cmp, theme, activeKey, rowsModels, col, attrs, secondCol, secondName, barColor, qcCols.join(), qcRange, heat, compColor, maxFuzzy, width, height, heatRow, diff, focus, maxSize, regionMode]);

  // ---- linked UMAP of the focused model -------------------------------------------------------------------
  const focusIdx = models.findIndex((m) => m.model_id === focus);
  const coords = useMemo(() => (umaps && focusIdx >= 0 ? modelUmap(umaps, focusIdx, N) : undefined), [umaps, focusIdx, N]);
  const notExported = [...state.selectedModels].filter((m) => !info.models.includes(m));
  const tauMax = normalize ? 5 : 1;

  return (
    <div className="mv-grid">
      <section className="mv-card mv-span-2" aria-labelledby="fu-title">
        <h2 id="fu-title">Label mixing · comparative fuzzy UpSet</h2>
        <p className="mv-small mv-muted">
          For every cell, the fraction of its neighbourhood (its {info.k} nearest neighbours in the model's own latent space, plus itself) that carries
          each label. A cell belongs to every label with fraction ≥ τ; its <em>signature</em> is that set of labels and its <em>strength</em> is the
          smallest of those fractions. Each column is one signature (intersection); its <em>fuzzy size</em> is the sum of strengths. Mixing of
          cell-type labels usually means lost biology; mixing of sample / study labels usually means successful batch correction.
        </p>

        <nav className="mv-seg" aria-label="Label column" style={{ margin: '8px 0' }}>
          {info.columns.map((c, i) => (
            <button key={c.name} type="button" aria-pressed={i === colIdx} onClick={() => (setColIdx(i), setActiveKey(null))}>
              {c.name} ({c.levels.length})
            </button>
          ))}
        </nav>

        <div className="mv-inline" style={{ marginBottom: 8 }}>
          <div className="mv-seg" role="group" aria-label="Group cells by" style={{ alignSelf: 'center' }}>
            <button type="button" aria-pressed={mode === 'cell'} onClick={() => (setMode('cell'), setActiveKey(null))}>
              Per cell
            </button>
            <button
              type="button"
              aria-pressed={mode === 'region'}
              disabled={!regions}
              title={regions ? undefined : regionsError ?? 'This export has no mixed regions (run make export-fuzzy).'}
              onClick={() => (setMode('region'), setActiveKey(null))}
            >
              Mixed regions
            </button>
          </div>
          {regionMode ? (
            <label>
              τ (threshold)
              <select value={regionTauIdx} onChange={(e) => setRegionTauIdx(+e.target.value)} aria-label="Membership threshold tau for regions">
                {regionTaus.map((t, i) => (
                  <option key={t} value={i}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <>
              <label>
                τ (threshold) = {tau.toFixed(2)}
                <input type="range" min={0.01} max={tauMax} step={0.01} value={tau} onChange={(e) => setTau(+e.target.value)} aria-label="Membership threshold tau" />
              </label>
              <label>
                Max labels per intersection
                <select value={maxSize} onChange={(e) => setMaxSize(+e.target.value)}>
                  {[1, 2, 3, 4, 5, 6, 7].map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
              <label className="mv-small mv-check">
                <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} /> Threshold
                enrichment (fraction ÷ label's global frequency)
              </label>
            </>
          )}
          <label>
            Focused model
            <select value={focus} onChange={(e) => setFocus(e.target.value)}>
              {info.models.map((m) => (
                <option key={m} value={m}>
                  {short(m)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Composition by
            <select value={secondName} onChange={(e) => setSecond(e.target.value)}>
              {secondOpts.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
            </select>
          </label>
        </div>
        <div className="mv-inline" style={{ marginBottom: 8 }}>
          <label>
            Sort by
            <select value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}>
              <option value="difference">difference across models</option>
              <option value="fuzzy">fuzzy size (focused model)</option>
              <option value="n_cells">cells (focused model)</option>
            </select>
          </label>
          <label>
            Difference
            <select value={diff} onChange={(e) => setDiffKind(e.target.value as DiffKind)}>
              <option value="range">max − min</option>
              <option value="var">variance</option>
              <option value="log_ratio" disabled={!canLogRatio}>
                |log2 ratio| (exactly two models)
              </option>
            </select>
          </label>
          <label>
            Min fuzzy size
            <input type="number" min={0} step={1} value={minSize} onChange={(e) => setMinSize(Math.max(0, +e.target.value))} style={{ width: 70 }} />
          </label>
          {!regionMode && (
            <label className="mv-small mv-check">
              <input type="checkbox" checked={showPure} onChange={(e) => setShowPure(e.target.checked)} /> Show pure (single-label) intersections
            </label>
          )}
          <label>
            Search labels
            <input type="text" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="e.g. T-A" style={{ width: 110 }} />
          </label>
        </div>

        <ModelPicker info={info} compared={compared} setCompared={setCompared} defaults={defaults.compared} short={short} />

        {regionMode && (
          <p className="mv-small mv-muted">
            <strong>Mixed regions.</strong> A cell is mixed when at least two labels reach τ in its neighbourhood (no maximum). Mixed cells that are
            neighbours in the model's kNN graph are joined into one region (regions under {info.regions?.min_size} cells are left out). Each region gets
            one signature: the labels whose average fraction over the region reaches τ. A heterogeneous cluster, which the per-cell view splits into many
            small columns, becomes one column here. Regions with the same signature are pooled.
          </p>
        )}
        {!regionMode && normalize && (
          <p className="mv-warning">
            Enrichment mode: τ applies to fraction ÷ global label frequency (τ = 1 means “as common as in the whole dataset”). Strengths and fuzzy sizes
            still use the raw fractions (0–1).
          </p>
        )}
        {col.warnings.map((w) => (
          <p key={w} className="mv-warning">
            {col.name}: {w}
          </p>
        ))}
        {notExported.length > 0 && (
          <p className="mv-warning">
            {notExported.length} model(s) selected in other views have no exported graph for this view (it covers {info.models.length} seed-0 models;
            change <code>fuzzy_upset.models</code> in config.yaml to add them).
          </p>
        )}

        <p className="mv-small" aria-live="polite">
          <strong>{short(focus)}</strong>:{' '}
          {regionMode
            ? `${cmp.nRegions} mixed region${cmp.nRegions === 1 ? '' : 's'}, ${(N - focusSig.nNone).toLocaleString()} cells in them, ${focusSig.nNone.toLocaleString()} cells outside`
            : `${focusSig.nDiffuse.toLocaleString()} diffuse cells (> ${maxSize} labels ≥ τ, shown as their own column), ${focusSig.nNone.toLocaleString()} with no label ≥ τ`}{' '}
          · showing {shown.length} of {total} {regionMode ? 'region signatures' : 'intersections'}
          {total > MAX_COLS ? ` (top ${MAX_COLS}; raise the min size or search to narrow)` : ''}.{' '}
          <span className="mv-muted">Click a column to select its cells in the focused model; click a heatmap cell to focus that model.</span>
        </p>
        <div className="mv-table-wrap">
          {shown.length ? (
            <svg ref={ref} width={width} height={height} role="img" aria-label={`Fuzzy UpSet of ${col.name} labels across ${rowsModels.length} models`} onPointerLeave={() => setTip(null)} />
          ) : (
            <p className="mv-muted">No intersection passes the filters{!showPure ? ' (pure intersections are hidden)' : ''}.</p>
          )}
        </div>
        <div className="mv-legend-row">
          <GlyphLegend max={maxFuzzy} color={heat} />
          {secondCol && <Swatches items={secondCol.levels.map((l) => ({ key: l, label: l, color: compColor(l) }))} />}
          <span className="mv-small mv-muted">QC rows: darker = higher mean among the shown intersections (exact values on hover).</span>
        </div>
        <details className="mv-explainer">
          <summary>Table view</summary>
          <table className="mv-table">
            <thead>
              <tr>
                <th>intersection</th>
                <th className="num">fuzzy size ({short(focus)})</th>
                <th className="num">cells</th>
                <th className="num">mean strength</th>
                <th className="num">difference</th>
                {rowsModels.slice(1).map((m) => (
                  <th key={m} className="num mono">
                    {short(m)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {shown.map((it) => (
                <tr key={it.key} className={activeKey === it.key ? 'is-selected' : ''} onClick={() => select(it)}>
                  <td>{names(it)}</td>
                  <td className="num">{fmt(it.fuzzy[0], 2)}</td>
                  <td className="num">{it.nCells[0]}</td>
                  <td className="num">{fmt(it.nCells[0] ? it.fuzzy[0] / it.nCells[0] : 0, 3)}</td>
                  <td className="num">{fmt(difference(it, diff, [0, 1]), 2)}</td>
                  {rowsModels.slice(1).map((m, j) => (
                    <td key={m} className="num">
                      {fmt(it.fuzzy[j + 1], 2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </details>
        <Tooltip tip={tip} />
      </section>

      <section className="mv-card mv-span-2" aria-labelledby="fu-umap">
        <h2 id="fu-umap">UMAP of the focused model · {short(focus)}</h2>
        {coords ? (
          <CellMap colorKey={`cat:${col.name}`} height={380} coords={coords} ariaLabel={`UMAP of ${focus}, coloured by ${col.name}`} />
        ) : (
          <p className="mv-muted">Per-model UMAPs are not in this export; the selection still highlights cells in the Cells and Embeddings tabs.</p>
        )}
      </section>
    </div>
  );
}

function ModelPicker({
  info,
  compared,
  setCompared,
  defaults,
  short,
}: {
  info: FuzzyInfo;
  compared: string[];
  setCompared: (v: string[]) => void;
  defaults: string[];
  short: (id: string) => string;
}) {
  const { state } = useSelection();
  const fromSel = [...state.selectedModels]
    .map((id) => (info.models.includes(id) ? id : id.replace(/\.s\d+$/, '.s0')))
    .filter((id, i, a) => info.models.includes(id) && a.indexOf(id) === i);
  const toggle = (id: string) => setCompared(compared.includes(id) ? compared.filter((m) => m !== id) : [...compared, id]);
  return (
    <details className="mv-explainer" style={{ marginBottom: 8 }}>
      <summary>
        Compared models: {compared.length} of {info.models.length} (the focused model is always the first row)
      </summary>
      <div className="mv-inline" style={{ margin: '6px 0' }}>
        <button type="button" className="mv-button" onClick={() => setCompared(defaults)}>
          Reference setup × all methods and batch keys
        </button>
        <button type="button" className="mv-button" onClick={() => setCompared(info.models)}>
          All {info.models.length}
        </button>
        <button type="button" className="mv-button" disabled={!fromSel.length} onClick={() => setCompared(fromSel)}>
          Models selected in other views{fromSel.length ? ` (${fromSel.length})` : ''}
        </button>
      </div>
      <div style={{ columns: '3 260px', fontSize: 12 }}>
        {info.models.map((m) => (
          <label key={m} style={{ display: 'block' }}>
            <input type="checkbox" checked={compared.includes(m)} onChange={() => toggle(m)} /> <span className="mono">{short(m)}</span>
          </label>
        ))}
      </div>
    </details>
  );
}
