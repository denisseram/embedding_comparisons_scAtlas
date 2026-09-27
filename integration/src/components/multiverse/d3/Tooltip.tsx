// One shared tooltip, positioned in page coordinates. Content is plain text rows (never innerHTML).
export interface TipRow {
  label: string;
  value: string;
}
export interface Tip {
  x: number;
  y: number;
  title: string;
  rows: TipRow[];
}

export function Tooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null;
  const left = Math.min(tip.x + 14, (typeof window !== 'undefined' ? window.innerWidth : 1200) - 280);
  return (
    <div className="mv-tooltip" style={{ left, top: tip.y + 14 }} role="status">
      <div className="mv-tooltip-title">{tip.title}</div>
      {tip.rows.map((r) => (
        <div className="mv-tooltip-row" key={r.label}>
          <strong>{r.value}</strong> <span>{r.label}</span>
        </div>
      ))}
    </div>
  );
}
