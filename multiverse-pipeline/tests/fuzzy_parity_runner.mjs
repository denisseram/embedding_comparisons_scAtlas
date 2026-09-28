// Runs the browser implementation (integration/.../compute/fuzzyUpset.ts) on a fixture written by
// tests/test_fuzzy_upset.py and prints its results as JSON. Needs Node >= 22.6 (TypeScript type stripping).
import { readFileSync } from 'node:fs';
import { compareModels, modelSignatures } from '../../integration/src/components/multiverse/compute/fuzzyUpset.ts';

const fx = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const counts = new Uint8Array(Buffer.from(fx.counts, 'base64'));
const out = fx.cases.map((cs) => {
  const col = fx.info.columns[cs.col];
  const p = { tau: cs.tau, maxSize: cs.max_size, normalize: cs.normalize };
  const per = fx.info.models.map((_, m) => modelSignatures(counts, fx.info, fx.N, m, col, p));
  const cmp = compareModels(per);
  return {
    models: per.map((ms) => ({
      sig: Array.from(ms.sig, (s) => (s >= 0 ? ms.keys[s] : s)),
      strength: Array.from(ms.strength),
      n_none: ms.nNone,
      n_diffuse: ms.nDiffuse,
    })),
    fuzzy: Object.fromEntries(cmp.intersections.map((it) => [it.key, Array.from(it.fuzzy)])),
  };
});
process.stdout.write(JSON.stringify(out));
