// @ts-check
import { defineConfig } from 'astro/config';

import react from "@astrojs/react";

// https://astro.build/config
// The deploy target is not decided yet. Set ASTRO_BASE (e.g. "/embedding_comparison_scAtlas")
// and ASTRO_SITE at build time for subpath hosting such as GitHub Pages.
export default defineConfig({
  site: process.env.ASTRO_SITE || undefined,
  base: process.env.ASTRO_BASE || undefined,
  integrations: [react()],
});