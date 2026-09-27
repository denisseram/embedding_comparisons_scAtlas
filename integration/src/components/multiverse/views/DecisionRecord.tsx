// V8 — Decision record: the chosen model plus a snapshot of the evidence, downloaded in the
// browser as JSON and Markdown (Blob; no server).
import { useEffect, useMemo, useState } from 'react';
import { useData } from '../data/DataContext';
import { useSelection } from '../state';
import { fmt } from '../d3/colors';
import { cellArray } from '../data/loader';
import { median, ranks } from '../compute/selection';

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function DecisionRecord() {
  const { entry, cells, cellsStatus, requestCells, z, zStatus, requestZ, fracZ, activeCells } = useData();
  const { manifest, models } = entry;
  const { state } = useSelection();
  const firstSel = [...state.selectedModels][0];
  const [choice, setChoice] = useState(firstSel ?? state.referenceModel);
  const [rationale, setRationale] = useState('');
  const [author, setAuthor] = useState('');

  useEffect(() => {
    if (cellsStatus === 'idle') requestCells();
    if (!zStatus[state.mode]) requestZ(state.mode);
  }, [cellsStatus, requestCells, zStatus, requestZ, state.mode]);
  useEffect(() => {
    if (firstSel) setChoice(firstSel);
  }, [firstSel]);

  const record = useMemo(() => {
    const mi = models.findIndex((m) => m.model_id === choice);
    const m = models[mi];
    if (!m) return null;
    const names = [...manifest.metrics.map((x) => x.name)];
    const metricRows = [
      ...['overall', 'bio', 'batch'].map((n) => {
        const vals = models.map((x) => (x as unknown as Record<string, number>)[n]);
        return { metric: n, group: 'aggregate', raw: vals[mi], scaled: vals[mi], rank: ranks(vals)[mi], seed_sd: m.seed_sd[n] };
      }),
      ...names.map((n) => {
        const vals = models.map((x) => x.raw[n]);
        return { metric: n, group: manifest.metrics.find((x) => x.name === n)!.group, raw: m.raw[n], scaled: m.scaled[n], rank: ranks(vals)[mi], seed_sd: m.seed_sd[n] };
      }),
    ];
    let noise: Record<string, number> | null = null;
    let region = null;
    if (cells) {
      const N = manifest.dataset.n_cells;
      noise = Object.fromEntries(
        manifest.measure_factors.map((f) => [f, median(Array.from(cellArray(cells, `E.${state.mode}.${f}`, N), Math.abs))]),
      );
      if (state.selectedRegion !== null) {
        const r = cells.regions.by_mode[state.mode].regions[state.selectedRegion];
        region = {
          region: r.region,
          label: r.label,
          size: r.size,
          E_mean: r.E_mean,
          H_mean: r.H_mean,
          stability_all: r.stability_all,
          stability_seed: r.stability_seed,
          qc_deviation: r.qc_deviation,
          composition_cell_type: r.comp_cell_type,
          composition_sample: r.comp_sample,
          top_genes: r.top_genes.slice(0, 5).map((g) => g.gene),
          local_variant_of_chosen_model: cells.regions.variant_levels[r.variants.code[mi]],
        };
      }
    }
    return {
      record_type: 'embedding-multiverse-decision',
      created_at: new Date().toISOString(),
      author: author || null,
      chosen_model: { model_id: m.model_id, config_id: m.config_id, factors: m.factors, seed: m.seed },
      rationale,
      context: {
        measure_mode: state.mode,
        measure_label: manifest.measure.mode_labels[state.mode],
        reference_model: state.referenceModel,
        data_generated_at: manifest.generated_at,
        schema_version: manifest.schema_version,
        dataset: manifest.dataset,
        weights: manifest.weights,
      },
      metrics: metricRows,
      selection: {
        selected_models: [...state.selectedModels],
        selected_cells: state.selectedCells ? state.selectedCells.length : 'all',
        selected_region: region,
      },
      calibrated_change_vs_reference: fracZ
        ? { fraction_cells_abs_z_gt_2: fracZ[mi], over_cells: activeCells.length, reference: state.referenceModel }
        : 'not available (z vs this reference not loaded or not exported for this mode)',
      noise_levels: noise ? { median_abs_E_per_factor: noise, note: 'seed = null reference (seed pseudo-factor)' } : 'cell data not loaded',
      caveats: [
        'Regions, effects and variants are evidence, not verdicts: the dashboard does not decide biology vs artifact.',
        'Validation of the pre-registered measure failed checks V2 (P2 attribution) and V5 (P3/P4 stability); see VALIDATION.md.',
      ],
    };
  }, [choice, models, manifest, cells, state, rationale, author, fracZ, activeCells]);

  const markdown = useMemo(() => {
    if (!record) return '';
    const r = record;
    const lines = [
      `# Decision record — ${r.chosen_model.model_id}`,
      '',
      `Created ${r.created_at}${r.author ? ` by ${r.author}` : ''}. Measure: ${r.context.measure_label}. Reference: \`${r.context.reference_model}\`. Data generated ${r.context.data_generated_at}.`,
      '',
      '## Chosen model',
      '',
      ...Object.entries(r.chosen_model.factors).map(([k, v]) => `- **${k}**: ${v}`),
      `- **seed**: ${r.chosen_model.seed}`,
      '',
      '## Rationale',
      '',
      r.rationale || '_(none given)_',
      '',
      '## Metrics (rank among all models, seed-replicate sd)',
      '',
      '| metric | group | raw | scaled | rank | seed sd |',
      '|---|---|---|---|---|---|',
      ...r.metrics.map((x) => `| ${x.metric} | ${x.group} | ${fmt(x.raw)} | ${fmt(x.scaled)} | ${x.rank} | ${fmt(x.seed_sd)} |`),
      '',
      '## Evidence snapshot',
      '',
      `- Selected models: ${r.selection.selected_models.length ? r.selection.selected_models.join(', ') : 'none'}`,
      `- Selected cells: ${r.selection.selected_cells}`,
      typeof r.calibrated_change_vs_reference === 'string'
        ? `- Calibrated change vs reference: ${r.calibrated_change_vs_reference}`
        : `- Fraction of cells with |z| > 2 vs reference: ${fmt(r.calibrated_change_vs_reference.fraction_cells_abs_z_gt_2)} over ${r.calibrated_change_vs_reference.over_cells} cells`,
    ];
    if (r.selection.selected_region) {
      const g = r.selection.selected_region;
      lines.push(
        `- Region ${g.label} (${g.size} cells): local variant in chosen model = **${g.local_variant_of_chosen_model}**, stability ${fmt(g.stability_all)} (all) / ${fmt(g.stability_seed)} (seed), QC deviation ${fmt(g.qc_deviation, 2)}`,
        `  - mean E: ${Object.entries(g.E_mean).map(([k, v]) => `${k}=${fmt(v as number, 2)}`).join(', ')}`,
        `  - top genes: ${g.top_genes.join(', ')}`,
      );
    }
    if (typeof r.noise_levels !== 'string')
      lines.push(`- Median |E| per factor: ${Object.entries(r.noise_levels.median_abs_E_per_factor).map(([k, v]) => `${k}=${fmt(v, 2)}`).join(', ')}`);
    lines.push('', '## Caveats', '', ...r.caveats.map((c) => `- ${c}`), '');
    return lines.join('\n');
  }, [record]);

  if (!record) return <p className="mv-error">Unknown model {choice}.</p>;
  const base = `decision-${record.chosen_model.model_id}`;
  return (
    <div className="mv-card">
      <h2>V8 · Decision record</h2>
      <div className="mv-inline">
        <label>
          Chosen model
          <select value={choice} onChange={(e) => setChoice(e.target.value)}>
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.model_id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Author (optional)
          <input type="text" value={author} onChange={(e) => setAuthor(e.target.value)} />
        </label>
      </div>
      <label className="mv-small" style={{ display: 'block', marginTop: 10 }}>
        Rationale
        <textarea value={rationale} onChange={(e) => setRationale(e.target.value)} placeholder="Why this model, given the evidence below?" />
      </label>
      <div className="mv-inline" style={{ margin: '10px 0' }}>
        <button type="button" className="mv-button primary" onClick={() => download(`${base}.json`, JSON.stringify(record, null, 2), 'application/json')}>
          Download JSON
        </button>
        <button type="button" className="mv-button" onClick={() => download(`${base}.md`, markdown, 'text/markdown')}>
          Download Markdown
        </button>
        {(!cells || !z[state.mode]) && <span className="mv-small mv-muted">Loading cell-level evidence…</span>}
      </div>
      <h3>Preview</h3>
      <pre className="mv-small" style={{ whiteSpace: 'pre-wrap', maxHeight: 420, overflow: 'auto', background: 'var(--page)', padding: 10, borderRadius: 6 }}>
        {markdown}
      </pre>
    </div>
  );
}
