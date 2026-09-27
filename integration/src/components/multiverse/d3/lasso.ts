// Freehand lasso on an SVG overlay. Calls onEnd(polygon) in the overlay's pixel coordinates.
import * as d3 from 'd3';

export type Polygon = [number, number][];

export function attachLasso(
  svg: SVGSVGElement,
  onEnd: (poly: Polygon, ev: MouseEvent) => void,
  opts: { minPoints?: number; filter?: (ev: PointerEvent) => boolean } = {},
) {
  const sel = d3.select(svg);
  const path = sel.append('path').attr('class', 'mv-lasso').attr('fill-rule', 'evenodd');
  let pts: Polygon = [];
  const drag = d3
    .drag<SVGSVGElement, unknown>()
    .filter((ev: PointerEvent) => (opts.filter ? opts.filter(ev) : !ev.button))
    .on('start', (ev) => {
      pts = [[ev.x, ev.y]];
      path.attr('d', null);
    })
    .on('drag', (ev) => {
      pts.push([ev.x, ev.y]);
      path.attr('d', `M${pts.map((p) => p.join(',')).join('L')}Z`);
    })
    .on('end', (ev) => {
      path.attr('d', null);
      if (pts.length >= (opts.minPoints ?? 4)) onEnd(pts, ev.sourceEvent as MouseEvent);
      pts = [];
    });
  sel.call(drag);
  return () => {
    sel.on('.drag', null);
    path.remove();
  };
}

export const inPolygon = (poly: Polygon, x: number, y: number) => d3.polygonContains(poly, [x, y]);
