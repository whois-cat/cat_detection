// Shared UI state. All timelines and players read the same view and
// playback, so they stay in sync (pan/zoom, hover, playhead).

export const cams = $state({
  list: [],      // [{id, label}]
  status: {},    // id -> ingest status from /api/status
  ranges: {},    // id -> [[startMs, endMs], ...] recorded spans
});

export const view = $state({
  from: 0,        // timeline viewport, wall-clock ms
  to: 0,
  hoverMs: null,  // hovered time, shown on every timeline
  follow: true,   // viewport's right edge tracks now
  dragging: false, // a timeline is being panned/pinched
  selected: 'all', // 'all' (grid) or a camera id
});

export const play = $state({
  live: true,
  playing: true,
  rate: 1,
  playheadMs: 0,  // history: time being shown; live: ~now
  waiting: {},    // camera id -> true while history data is loading at the playhead
  gap: null,      // {lengthMs, target} while showing the "skipping a gap" overlay
});

export const clock = $state({ now: Date.now() });

export const LIVE_SNAP_MS = 5_000;
export const ZOOM_MIN_MS = 30_000;
export const ZOOM_MAX_MS = 30 * 86_400_000;

export function visibleCameras() {
  return view.selected === 'all' ? cams.list.map(c => c.id) : [view.selected];
}

// ---- viewport animation (one at a time, shared by all timelines) ----

let anim = null;

export function animateView(from, to, durationMs = 380) {
  anim = { start: performance.now(), dur: durationMs, from0: view.from, to0: view.to, from, to };
  requestAnimationFrame(tick);
}

export function cancelAnimation() { anim = null; }
export function animating() { return anim !== null; }

function tick(now) {
  if (!anim) return;
  const t = Math.min(1, (now - anim.start) / anim.dur);
  const e = 1 - Math.pow(1 - t, 3);
  view.from = anim.from0 + (anim.from - anim.from0) * e;
  view.to = anim.to0 + (anim.to - anim.to0) * e;
  if (t < 1) requestAnimationFrame(tick); else anim = null;
}

// ---- mode switches ----

export function goLive() {
  play.live = true;
  play.playing = true;
  play.gap = null;
  view.follow = true;
  const span = view.to - view.from;
  const to = clock.now + span * 0.05;
  animateView(to - span, to);
}

export function seek(ms) {
  if (ms >= clock.now - LIVE_SNAP_MS) { goLive(); return; }
  play.live = false;
  play.gap = null;
  play.playheadMs = ms;
  view.follow = false;
}

// ---- URL hash: #cam=…&from=…&to=…&t=… ----

export function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const num = k => { const v = parseInt(p.get(k), 10); return Number.isNaN(v) ? null : v; };
  return { cam: p.get('cam'), from: num('from'), to: num('to'), t: num('t') };
}

let hashTimer = 0;
export function writeHash() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    const p = new URLSearchParams();
    p.set('cam', view.selected);
    p.set('from', Math.round(view.from).toString());
    p.set('to', Math.round(view.to).toString());
    if (!play.live) p.set('t', Math.round(play.playheadMs).toString());
    const next = '#' + p.toString();
    if (location.hash !== next) history.replaceState(null, '', next);
  }, 300);
}
