// Palettes (reference dataviz palette, validated light + dark) and theme detection.
import * as d3 from 'd3';
import { useEffect, useState } from 'react';

export type Theme = 'light' | 'dark';

const CAT = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
};
const SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
export const INK = {
  light: { surface: '#fcfcfb', primary: '#0b0b0b', secondary: '#52514e', muted: '#898781', grid: '#e1e0d9', axis: '#c3c2b7', neutral: '#c3c2b7', mid: '#f0efec' },
  dark: { surface: '#1a1a19', primary: '#ffffff', secondary: '#c3c2b7', muted: '#898781', grid: '#2c2c2a', axis: '#383835', neutral: '#5b5a55', mid: '#383835' },
};

export function useTheme(): Theme {
  const get = (): Theme => {
    if (typeof window === 'undefined') return 'light';
    const t = document.documentElement.dataset.theme;
    if (t === 'dark' || t === 'light') return t;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  };
  const [theme, setTheme] = useState<Theme>(get);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const on = () => setTheme(get());
    mq.addEventListener('change', on);
    const mo = new MutationObserver(on);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => {
      mq.removeEventListener('change', on);
      mo.disconnect();
    };
  }, []);
  return theme;
}

export function categorical(theme: Theme, domain: string[]) {
  const pal = CAT[theme];
  // fixed order, never cycled: >8 levels fold into the neutral colour
  return (v: string) => {
    const i = domain.indexOf(v);
    return i >= 0 && i < pal.length ? pal[i] : INK[theme].neutral;
  };
}

/** Sequential single-hue ramp; low values recede toward the surface in both modes. */
export function sequential(theme: Theme, domain: [number, number]) {
  const steps = theme === 'light' ? SEQ : [...SEQ].reverse();
  const lo = domain[0];
  const hi = domain[1] === domain[0] ? domain[0] + 1 : domain[1];
  return d3.scaleSequential(d3.interpolateRgbBasis(steps)).domain([lo, hi]).clamp(true);
}

/** Diverging blue <-> gray <-> red around 0 (for signed effects). */
export function diverging(theme: Theme, maxAbs: number) {
  const m = maxAbs || 1;
  const mid = INK[theme].mid;
  const interp = d3.interpolateRgbBasis(['#184f95', '#6da7ec', mid, '#ec8f8e', '#b8302f']);
  return d3.scaleDiverging(interp).domain([-m, 0, m]).clamp(true);
}

export function fmt(v: number | null | undefined, d = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(d);
}
