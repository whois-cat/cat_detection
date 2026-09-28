// CV results per camera, looked up by frame time.

import { CLOCK_RATE } from './mp4.js';

const KEEP_MS = 5 * 60_000;

// Result: {ms, dets: [{box: [x, y, w, h] fractions, score, cats: {name: p}}], model, infer_ms}
export class LabelStore {
  constructor() {
    this.items = []; // sorted by ms
  }

  // add accepts a hub/sidecar result (pts in 90 kHz ticks).
  add(r) {
    const item = { ms: r.pts / (CLOCK_RATE / 1000), dets: r.dets || [], model: r.model, infer_ms: r.infer_ms };
    const a = this.items;
    if (!a.length || a[a.length - 1].ms < item.ms) { a.push(item); return; }
    const i = this.index(item.ms);
    if (a[i]?.ms !== item.ms) a.splice(i, 0, item); // skip duplicates
  }

  // at returns the newest result for a frame at or before ms, or null.
  at(ms) {
    const i = this.index(ms + 0.5);
    return i > 0 ? this.items[i - 1] : null;
  }

  // prune drops results far from ms (either side).
  prune(ms) {
    this.items = this.items.filter(r => Math.abs(r.ms - ms) < KEEP_MS);
  }

  clear() { this.items = []; }

  // index of the first item with ms >= v
  index(v) {
    let lo = 0, hi = this.items.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.items[m].ms < v) lo = m + 1; else hi = m; }
    return lo;
  }
}

// topCat mirrors streamhub's labels.Det.TopCat: the likely identity for display.
export function topCat(det) {
  const cats = Object.entries(det.cats || {});
  if (!cats.length) return 'cat';
  cats.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  return cats[0][1] >= 0.5 ? cats[0][0] : 'unknown';
}

// parseSidecar parses a segment's .labels.jsonl into results.
export function parseSidecar(text) {
  const out = [];
  for (const line of text.split('\n')) {
    if (!line) continue;
    try {
      const r = JSON.parse(line);
      if (r.t === 'cv') out.push(r);
    } catch { /* partial last line while the segment is still being labeled */ }
  }
  return out;
}
