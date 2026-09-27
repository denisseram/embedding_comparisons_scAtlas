// Small legend components: continuous colour bar (D3 axis in a ref) and categorical swatches.
import * as d3 from 'd3';
import { useEffect, useRef } from 'react';

export function Colorbar({ color, domain, label, width = 200 }: { color: (v: number) => string; domain: [number, number]; label: string; width?: number }) {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const svg = d3.select(ref.current!);
    svg.selectAll('*').remove();
    const h = 10;
    const id = `grad-${Math.random().toString(36).slice(2)}`;
    const grad = svg.append('defs').append('linearGradient').attr('id', id);
    const [lo, hi] = domain;
    d3.range(0, 1.0001, 0.1).forEach((t) =>
      grad.append('stop').attr('offset', `${t * 100}%`).attr('stop-color', color(lo + t * (hi - lo))),
    );
    svg.append('rect').attr('x', 6).attr('y', 16).attr('width', width - 12).attr('height', h).attr('rx', 2).attr('fill', `url(#${id})`);
    const x = d3.scaleLinear().domain(domain).range([6, width - 6]);
    svg
      .append('g')
      .attr('class', 'mv-axis')
      .attr('transform', `translate(0,${16 + h})`)
      .call(d3.axisBottom(x).ticks(4).tickSize(3).tickFormat(d3.format('.3~g')));
    svg.append('text').attr('class', 'mv-legend-title').attr('x', 6).attr('y', 11).text(label);
  }, [color, domain[0], domain[1], label, width]);
  return <svg ref={ref} width={width} height={46} role="img" aria-label={`Colour scale for ${label}, ${domain[0]} to ${domain[1]}`} />;
}

export function Swatches({
  items,
  onPick,
  active,
}: {
  items: { key: string; label: string; color: string }[];
  onPick?: (key: string) => void;
  active?: string | null;
}) {
  return (
    <ul className="mv-swatches" aria-label="Legend">
      {items.map((it) => (
        <li key={it.key}>
          <button
            type="button"
            className={active === it.key ? 'is-active' : ''}
            onClick={() => onPick?.(it.key)}
            disabled={!onPick}
            aria-pressed={active === it.key}
          >
            <span className="mv-swatch" style={{ background: it.color }} />
            {it.label}
          </button>
        </li>
      ))}
    </ul>
  );
}
