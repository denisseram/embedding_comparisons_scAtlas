// Runs the browser implementation (integration/.../compute/fuzzyUpset.ts) on a fixture written by
// tests/test_fuzzy_upset.py and prints its results as JSON. Needs Node >= 22.6 (TypeScript type stripping).
import { readFileSync } from 'node:fs';
import { compareModels, modelSignatures, regionSignatures } from '../../integration/src/components/multiverse/compute/fuzzyUpset.ts';

const fx = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const counts = new Uint8Array(Buffer.from(fx.counts, 'base64'));
const dump = (ms) => ({
  sig: Array.from(ms.sig, (s) => (s >= 0 ? ms.keys[s] : s)),
  strength: Array.from(ms.strength),
  n_none: ms.nNone,
  n_diffuse: ms.nDiffuse,
});
const cases = fx.cases.map((cs) => {
  const col = fx.info.columns[cs.col];
  const p = { tau: cs.tau, maxSize: cs.max_size, normalize: cs.normalize };
  const per = fx.info.models.map((_, m) => modelSignatures(counts, fx.info, fx.N, m, col, p));
  const cmp = compareModels(per, true);
  return { models: per.map(dump), fuzzy: Object.fromEntries(cmp.intersections.map((it) => [it.key, Array.from(it.fuzzy)])) };
});
let regions = [];
if (fx.regions) {
  const reg = new Int16Array(new Uint8Array(Buffer.from(fx.regions, 'base64')).buffer);
  regions = fx.region_cases.map((rc) => fx.info.models.map((_, m) => dump(regionSignatures(reg, counts, fx.info, fx.N, m, rc.col, rc.tau_idx))));
}
process.stdout.write(JSON.stringify({ cases, regions }));
