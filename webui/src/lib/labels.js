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
    const item = {
      ms: r.pts / (CLOCK_RATE / 1000), dets: r.dets || [], model: r.model, worker: r.worker,
      infer_ms: r.infer_ms, decision: r.decision,
    };
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

// DecisionStore keeps decider decisions per feeder, looked up by frame time.
export class DecisionStore {
  constructor() {
    this.feeders = new Map(); // feeder id -> LabelStore of decisions
  }

  add(d) {
    let s = this.feeders.get(d.feeder);
    if (!s) this.feeders.set(d.feeder, (s = new LabelStore()));
    s.add({ pts: d.pts, dets: [], decision: d });
  }

  // at returns the latest decision of each feeder at or before ms.
  at(ms) {
    const out = [];
    for (const s of this.feeders.values()) {
      const r = s.at(ms);
      if (r) out.push(r.decision);
    }
    return out.sort((a, b) => a.feeder.localeCompare(b.feeder));
  }

  prune(ms) { for (const s of this.feeders.values()) s.prune(ms); }
  clear() { this.feeders.clear(); }
}

// describeDecision renders a decision as one short line.
export function describeDecision(d) {
  const who = d.identity ? `${d.identity}${d.conf != null ? ` ${Math.round(d.conf * 100)}%` : ''}` : '';
  const why = d.action === 'open' ? '' : d.reason;
  return [`${d.feeder}: ${d.state}`, who, why, d.display ? `[${d.display}]` : '']
    .filter(Boolean).join(' · ');
}

// parseSidecar parses a segment's .labels.jsonl into CV results and decisions.
export function parseSidecar(text) {
  const results = [], decisions = [];
  for (const line of text.split('\n')) {
    if (!line) continue;
    try {
      const r = JSON.parse(line);
      if (r.t === 'cv') results.push(r);
      else if (r.t === 'decision') decisions.push(r);
    } catch { /* partial last line while the segment is still being labeled */ }
  }
  return { results, decisions };
}
