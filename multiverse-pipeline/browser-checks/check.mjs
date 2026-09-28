// End-to-end browser checks for the /multiverse page (36 checks).
// Usage: npm install && npx playwright install firefox webkit
//        node check.mjs <url> <chrome|chromium|firefox|webkit> [light|dark]
// 'chrome' uses the locally installed Google Chrome; the others use Playwright's browsers.
import { chromium, firefox, webkit } from 'playwright';
const [url, which = 'chrome', scheme = 'light'] = process.argv.slice(2);
const launcher = { chrome: chromium, chromium, firefox, webkit }[which];
const browser = await launcher.launch(which === 'chrome' ? { channel: 'chrome' } : {});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 1000 }, colorScheme: scheme, acceptDownloads: true });
const page = await ctx.newPage();
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('requestfailed', (r) => errors.push('requestfailed: ' + r.url()));
const results = [];
const check = (name, ok, detail = '') => { results.push({ name, ok: !!ok, detail }); };
const timed = async (fn, until) => { const t0 = Date.now(); await fn(); await page.waitForFunction(until, null, { timeout: 5000, polling: 'raf' }); return Date.now() - t0; };
const summary = () => page.locator('.mv-selection-summary').innerText();

await page.goto(url, { waitUntil: 'networkidle' });
await page.waitForSelector('#v0a');
for (const h of ['v0a', 'v0b', 'v3']) check(`renders ${h}`, await page.locator(`#${h}`).count());
const circles = () => page.evaluate(() => [...document.querySelectorAll('section[aria-labelledby=v0a] svg circle')].map((c) => c.getAttribute('fill')).join());
const nCircles = await page.evaluate(() => document.querySelectorAll('section[aria-labelledby=v0a] svg circle').length);
check('V0a draws 144 models', nCircles === 144, `${nCircles}`);
check('V3 canvas drawn', await page.evaluate(() => { const c = document.querySelector('section[aria-labelledby=v3] canvas'); return c && c.width > 0; }));

// dropdown recolour + leaderboard update
const before = await circles();
const colorSel = page.getByLabel('Colour models by');
const colorKeys = await colorSel.locator('option').evaluateAll((os) => os.map((o) => o.value));
const nMetrics = await page.evaluate(async () => (await (await fetch(new URL('../multiverse-data/manifest.json', location.href))).json()).metrics.length);
const hasAll = ['mv:', 'factor:', 'agg:', 'raw:', 'scaled:'].every((p) => colorKeys.some((k) => k.startsWith(p)));
check('colour dropdown offers multiverse measures, decisions and all metrics', hasAll && colorKeys.filter((k) => k.startsWith('raw:')).length === nMetrics && colorKeys.filter((k) => k.startsWith('scaled:')).length === nMetrics, `${colorKeys.length} options, ${nMetrics} metrics`);
const msMetric = await timed(() => colorSel.selectOption('raw:graph_iLISI'), () => document.querySelector('section[aria-labelledby=v0a] .mv-legend-title')?.textContent?.includes('graph_iLISI'));
check('metric colour recolours V0a', true, `${msMetric} ms`);
let ms = await timed(() => colorSel.selectOption('mv:consensus_share'), () => document.querySelector('section[aria-labelledby=v0a] .mv-legend-title')?.textContent?.includes('consensus'));
const after = await circles();
check('dropdown recolours V0a', before !== after);
results.push({ name: 'timing: recolour V0a', ok: ms < 200, detail: `${ms} ms` });
const lbBefore = await page.locator('.mv-leaderboard tbody tr').first().innerText();
ms = await timed(() => page.getByLabel('Rank by').selectOption('raw:ARI'), () => document.querySelector('.mv-leaderboard strong')?.textContent?.includes('ARI'));
const lbAfter = await page.locator('.mv-leaderboard tbody tr').first().innerText();
check('Rank by updates V0b', lbBefore !== lbAfter || (await page.locator('.mv-leaderboard').innerText()).includes('ARI (raw)'), `${ms} ms`);
results.push({ name: 'timing: leaderboard re-rank', ok: ms < 200, detail: `${ms} ms` });
ms = await timed(() => colorSel.selectOption('factor:batch_key'), () => document.querySelectorAll('section[aria-labelledby=v0a] .mv-swatches li').length >= 2);
check('categorical colour + legend', true, `${ms} ms`);
const pos = () => page.evaluate(() => [...document.querySelectorAll('section[aria-labelledby=v0a] svg circle')].slice(0, 5).map((c) => c.getAttribute('cx')).join());
for (const lay of ['tsne', 'mds', 'umap']) {
  const p0 = await pos();
  await page.getByLabel('Layout').selectOption(lay);
  await page.waitForTimeout(50);
  check(`layout ${lay} moves models`, (await pos()) !== p0);
}
await colorSel.selectOption('mv:frac_z');
await page.waitForFunction(() => !document.body.innerText.includes('loading calibrated change'), null, { timeout: 15000 });
check('frac |z|>2 colour loads z and recolours', (await circles()) !== after);
await colorSel.selectOption('factor:method');
await page.getByRole('button', { name: 'Funky heatmap' }).click();
check('funky heatmap renders', await page.locator('.mv-leaderboard svg circle').count() > 50);
await page.getByRole('button', { name: 'Table', exact: true }).click();

// linking: leaderboard row → model selection
ms = await timed(() => page.locator('.mv-leaderboard tbody tr').first().click(), () => document.querySelector('.mv-selection-summary')?.textContent?.startsWith('1 model'));
check('leaderboard click selects model', (await summary()).startsWith('1 model'), `${ms} ms`);
const big = await page.evaluate(() => [...document.querySelectorAll('section[aria-labelledby=v0a] svg circle')].filter((c) => c.getAttribute('r') === '6.5').length);
check('V0a highlights selected model', big === 1);
// lasso on V0a
await page.locator('section[aria-labelledby=v0a] svg').first().evaluate((e) => e.scrollIntoView({ block: 'center' }));
const svgBox = await page.locator('section[aria-labelledby=v0a] svg').first().boundingBox();
await page.mouse.move(svgBox.x + 5, svgBox.y + 5); await page.mouse.down();
for (const [fx, fy] of [[0.55, 0.02], [0.55, 0.98], [0.02, 0.98], [0.01, 0.03]]) await page.mouse.move(svgBox.x + fx * svgBox.width, svgBox.y + fy * svgBox.height, { steps: 4 });
const t0 = Date.now();
await page.mouse.up();
await page.waitForFunction(() => !document.querySelector('.mv-selection-summary')?.textContent?.startsWith('1 model'), null, { timeout: 3000 }).catch(() => {});
const lassoMs = Date.now() - t0;
results.push({ name: 'timing: model lasso', ok: lassoMs < 200, detail: `${lassoMs} ms` });
check('V0a lasso selects models', /^\d+ models/.test(await summary()) && !(await summary()).startsWith('0 '), `${await summary()} (${lassoMs} ms mouse-up → selection rendered)`);
await page.getByRole('button', { name: 'Reset selection' }).click();
check('reset clears selection', (await summary()).startsWith('0 models · all cells'));
// V3 click selects two models
await page.locator('section[aria-labelledby=v3] canvas').evaluate((e) => e.scrollIntoView({ block: 'center' }));
const mBox = await page.locator('section[aria-labelledby=v3] canvas').boundingBox();
await page.mouse.click(mBox.x + mBox.width * 0.3, mBox.y + mBox.height * 0.6);
check('V3 click selects two models', (await summary()).startsWith('2 models'));
await page.getByRole('button', { name: 'Reset selection' }).click();

// embeddings tab: per-model UMAPs
await page.getByRole('tab', { name: /Embeddings/ }).click();
await page.waitForSelector('[aria-label^="UMAP of embedding"]', { timeout: 20000 });
const nPanels = await page.locator('[aria-label^="UMAP of embedding"]').count();
check('embedding viewer shows per-model UMAPs', nPanels >= 2, `${nPanels} panels`);
await page.getByLabel('Model for panel 4').selectOption({ index: 5 });
await page.waitForTimeout(200);
check('embedding panel model picker', (await page.locator('[aria-label^="UMAP of embedding"]').count()) === nPanels + 1);
await page.getByLabel('Colour cells by').selectOption('cat:cell_type');
check('embedding viewer recolours', (await page.locator('.mv-embed-panel .mv-swatches').first().innerText()).includes('T-A'));

// cell tab
await page.getByRole('tab', { name: /Cells/ }).click();
await page.waitForSelector('#v1', { timeout: 15000 });
for (const h of ['v1', 'v2', 'v4', 'v5', 'v6', 'v7']) check(`renders ${h}`, await page.locator(`#${h}`).count());
const cellSel = page.getByLabel('Colour cells by');
ms = await timed(() => cellSel.selectOption('measure:E.jaccard.batch_key'), () => document.querySelector('section[aria-labelledby=v1] .mv-legend-title')?.textContent?.includes('batch_key'));
results.push({ name: 'timing: cell map recolour', ok: ms < 200, detail: `${ms} ms` });
// hover tooltip on the cell map
await page.locator('section[aria-labelledby=v1] canvas').evaluate((e) => e.scrollIntoView({ block: 'center' }));
const cBox = await page.locator('section[aria-labelledby=v1] canvas').boundingBox();
let tipOk = false;
for (let i = 0; i < 40 && !tipOk; i++) {
  await page.mouse.move(cBox.x + cBox.width * (0.1 + 0.02 * i), cBox.y + cBox.height * 0.5);
  tipOk = (await page.locator('.mv-tooltip').count()) > 0;
}
check('cell map hover tooltip', tipOk);
// lasso cells
const edgeBefore = await page.evaluate(() => [...document.querySelectorAll('section[aria-labelledby=v6] path')].slice(0, 20).map((p) => p.getAttribute('stroke')).join());
await page.mouse.move(cBox.x + cBox.width * 0.1, cBox.y + cBox.height * 0.1); await page.mouse.down();
for (const [fx, fy] of [[0.6, 0.1], [0.6, 0.6], [0.1, 0.6], [0.1, 0.12]]) await page.mouse.move(cBox.x + fx * cBox.width, cBox.y + fy * cBox.height, { steps: 4 });
const t1 = Date.now();
await page.mouse.up();
await page.waitForFunction(() => /cells/.test(document.querySelector('.mv-selection-summary')?.textContent ?? '') && !/all cells/.test(document.querySelector('.mv-selection-summary')?.textContent ?? ''), null, { timeout: 3000 }).catch(() => {});
const cellLassoMs = Date.now() - t1;
results.push({ name: 'timing: cell lasso (incl. V6/V0a/V7 recompute)', ok: cellLassoMs < 200, detail: `${cellLassoMs} ms` });
check('V1 lasso selects cells', !(await summary()).includes('all cells'), `${await summary()} (${cellLassoMs} ms mouse-up → selection + linked views rendered)`);
const edgeAfter = await page.evaluate(() => [...document.querySelectorAll('section[aria-labelledby=v6] path')].slice(0, 20).map((p) => p.getAttribute('stroke')).join());
check('V6 edges recoloured for cell selection', edgeBefore !== edgeAfter);
check('V7 inspector shows selection', (await page.locator('section[aria-labelledby=v7]').innerText()).includes('Lasso selection'));
// region via fingerprint row
await page.locator('section[aria-labelledby=v4] svg g').nth(2).click();
await page.waitForTimeout(100);
check('V4 row click selects region', (await summary()).includes('region R'), await summary());
check('V7 shows top genes for region', (await page.locator('section[aria-labelledby=v7] .mv-genes li').count()) > 0);
// model selection filters V5
await page.getByRole('button', { name: 'Reset selection' }).click();
const colsAll = await page.locator('section[aria-labelledby=v5] svg rect').count();
await page.locator('section[aria-labelledby=v6] circle').first().click();
await page.waitForTimeout(100);
const colsSel = await page.locator('section[aria-labelledby=v5] svg rect').count();
check('V6 node click selects models and filters V5', colsSel < colsAll && (await summary()).startsWith('3 models'), `${colsAll} → ${colsSel} cells`);
// measure mode switch
await page.getByLabel('Measure').selectOption('composition');
await page.waitForTimeout(300);
check('measure mode switch shows exploratory banner', (await page.locator('.mv-warning').innerText()).includes('Exploratory'));
await page.getByLabel('Measure').selectOption('jaccard');

// decision record
await page.getByRole('tab', { name: /Decision/ }).click();
await page.waitForSelector('text=Download JSON');
await page.waitForFunction(() => !document.body.innerText.includes('Loading cell-level evidence'), null, { timeout: 15000 });
const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download JSON' }).click()]);
const json = JSON.parse(await (await import('node:fs/promises')).readFile(await dl.path(), 'utf8'));
check('V8 JSON download', json.record_type === 'embedding-multiverse-decision' && json.metrics.length > 10, dl.suggestedFilename());
const [dl2] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download Markdown' }).click()]);
check('V8 Markdown download', dl2.suggestedFilename().endsWith('.md'));

check('no console errors', errors.length === 0, errors.join(' | '));
await page.screenshot({ path: process.env.SHOT ?? '/dev/null', fullPage: true }).catch(() => {});
await browser.close();
const failed = results.filter((r) => !r.ok);
for (const r of results) console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}${r.detail ? '  — ' + r.detail : ''}`);
console.log(`\n${which}/${scheme}: ${results.length - failed.length}/${results.length} passed`);
process.exit(failed.length ? 1 : 0);
