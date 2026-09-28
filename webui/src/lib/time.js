// Time formatting and lookup helpers.

export const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
export const DAY_MS = 86_400_000;

export const pad2 = n => n.toString().padStart(2, '0');

export function fmtTimeOfDay(ms, seconds = false) {
  const d = new Date(ms);
  const hm = `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
  return seconds ? `${hm}:${pad2(d.getSeconds())}` : hm;
}

export function fmtDate(ms) {
  const d = new Date(ms);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}`;
}

export function fmtDateTime(ms) {
  return `${fmtDate(ms)}, ${fmtTimeOfDay(ms, true)}`;
}

export function isLocalMidnight(ms) {
  const d = new Date(ms);
  return d.getHours() === 0 && d.getMinutes() === 0 && d.getSeconds() === 0 && d.getMilliseconds() === 0;
}

// fmtDuration gives a compact approximate duration: 850ms, 4.2s, 12min, 3.5h, 2.0d.
export function fmtDuration(ms) {
  const abs = Math.abs(ms);
  if (abs < 1000) return `${Math.round(ms)}ms`;
  if (abs < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)}s`;
  if (abs < 3600_000) return `${(ms / 60_000).toFixed(ms < 600_000 ? 1 : 0)}min`;
  if (abs < DAY_MS) return `${(ms / 3600_000).toFixed(ms < 36_000_000 ? 1 : 0)}h`;
  return `${(ms / DAY_MS).toFixed(1)}d`;
}

// fmtGap formats a gap length as MM:SS, or HH:MM:SS from one hour
// (two-digit groups; e.g. 09:00, 01:05:00).
export function fmtGap(ms) {
  const s = Math.round(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rest = `${pad2(m)}:${pad2(s % 60)}`;
  return h > 0 ? `${pad2(h)}:${rest}` : rest;
}

// Ranges are sorted, non-overlapping [startMs, endMs] pairs.

// rangeIndexAt returns the index of the range containing t, or -1.
export function rangeIndexAt(ranges, t) {
  let lo = 0, hi = ranges.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ranges[mid][1] <= t) lo = mid + 1; else hi = mid;
  }
  return lo < ranges.length && ranges[lo][0] <= t ? lo : -1;
}

// nextRangeStart returns the start of the first range starting after t, or null.
export function nextRangeStart(ranges, t) {
  for (const [s] of ranges) if (s > t) return s;
  return null;
}
