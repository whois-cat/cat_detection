// Colour per identity label, derived from the label string: stable per name,
// no hardcoded cat names. 'cat' (no identity) uses the neutral default.
export const DEFAULT_CAT_COLOR = '#00ff88';

export function catColor(c) {
  if (!c || c === 'cat') return DEFAULT_CAT_COLOR;
  if (c === 'unknown') return '#cccccc';
  let h = 0;
  for (let i = 0; i < c.length; i++) h = (h * 31 + c.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360} ${55 + (h >> 9) % 25}% ${50 + (h >> 17) % 20}%)`;
}
