// Video sources for a camera: live (WebSocket) and history (segment files).
// Both feed an MseSink attached to the camera's <video>.

import { MseSink } from './mse.js';
import { rangeIndexAt } from './time.js';
import { parseSidecar } from './labels.js';

// Live latency: jump to the live edge when further behind than this…
const LIVE_MAX_LAG_S = 1.0;
// …landing this far behind the newest frame.
const LIVE_TARGET_LAG_S = 0.3;
const TRIM_BEHIND_MS = 30_000;
const RECONNECT_MAX_MS = 10_000;

export class LiveSource {
  constructor(video, camera, labels, decisions) {
    this.video = video;
    this.camera = camera;
    this.labels = labels;
    this.decisions = decisions;
    this.sink = new MseSink(video);
    this.connected = false;
    this.closed = false;
    this.backoff = 500;
    this.lastTrim = 0;
    this.connect();
  }

  connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/api/live/${encodeURIComponent(this.camera)}`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { this.connected = true; this.backoff = 500; };
    ws.onmessage = ev => {
      if (typeof ev.data === 'string') {
        const m = JSON.parse(ev.data);
        if (m.type === 'labels') this.labels.add(m);
        else if (m.type === 'decision') this.decisions.add(m);
        return;
      }
      this.sink.append(new Uint8Array(ev.data)).then(() => this.keepUp(), err => console.warn(this.camera, err));
    };
    ws.onclose = () => {
      this.connected = false;
      if (this.closed) return;
      setTimeout(() => !this.closed && this.connect(), this.backoff);
      this.backoff = Math.min(this.backoff * 2, RECONNECT_MAX_MS);
    };
    this.ws = ws;
  }

  // keepUp holds latency near the live edge and skips over gaps.
  keepUp() {
    const b = this.video.buffered;
    if (!b.length) return;
    const start = b.start(b.length - 1), end = b.end(b.length - 1);
    const t = this.video.currentTime;
    if (t < start || end - t > LIVE_MAX_LAG_S) this.video.currentTime = Math.max(start, end - LIVE_TARGET_LAG_S);
    if (this.video.paused) this.video.play().catch(() => {});
    const now = this.sink.wallMs();
    if (now - this.lastTrim > TRIM_BEHIND_MS) {
      this.lastTrim = now;
      this.sink.remove(0, now - TRIM_BEHIND_MS);
      this.labels.prune(now);
      this.decisions.prune(now);
    }
  }

  wallMs() { return this.sink.wallMs(); }

  destroy() {
    this.closed = true;
    this.ws.close();
    this.sink.destroy();
  }
}

// History buffering around the playhead.
const AHEAD_MS = 30_000;
const KEEP_BEHIND_MS = 60_000;
const LIST_WINDOW_MS = 10 * 60_000;
// Drift between this video and the shared playhead: seek above HARD,
// otherwise nudge the playback rate.
const DRIFT_HARD_MS = 1000;
const DRIFT_GAIN = 0.0002; // rate change per ms of drift
const DRIFT_MAX_NUDGE = 0.1;

export class HistorySource {
  constructor(video, camera, labels, decisions) {
    this.video = video;
    this.camera = camera;
    this.labels = labels;
    this.decisions = decisions;
    this.sink = new MseSink(video);
    this.segments = [];       // sorted by start: {start, end, url}
    this.listed = null;       // [from, to] covered by this.segments
    this.listing = null;      // in-flight list request
    this.appended = new Set();
    this.loading = false;
    this.destroyed = false;
  }

  // update aligns this camera with the shared playhead. Returns
  // {waiting, hasData}: waiting while data exists at the playhead but isn't
  // buffered yet.
  update(playheadMs, playing, rate) {
    this.ensureList(playheadMs);
    this.fill(playheadMs);
    this.trim(playheadMs);

    const spans = this.segments.map(s => [s.start, s.end]);
    const hasData = rangeIndexAt(spans, playheadMs) >= 0;
    const buffered = rangeIndexAt(this.sink.buffered(), playheadMs) >= 0;
    const v = this.video;
    if (!hasData || !buffered) {
      if (!v.paused) v.pause();
      return { waiting: hasData && !buffered, hasData };
    }
    const drift = this.sink.wallMs() - playheadMs;
    if (Math.abs(drift) > DRIFT_HARD_MS) {
      v.currentTime = this.sink.mediaTime(playheadMs);
    }
    const nudge = Math.max(-DRIFT_MAX_NUDGE, Math.min(DRIFT_MAX_NUDGE, -drift * DRIFT_GAIN));
    v.playbackRate = rate * (1 + nudge);
    if (playing && v.paused) v.play().catch(() => {});
    if (!playing && !v.paused) v.pause();
    return { waiting: false, hasData };
  }

  ensureList(t) {
    const want = [t - KEEP_BEHIND_MS, t + AHEAD_MS];
    const stale = !this.listed || want[0] < this.listed[0] || want[1] > this.listed[1] ||
      // the newest segments keep appearing near now
      this.listed[1] > Date.now() - LIST_WINDOW_MS / 2;
    if (!stale || this.listing) return;
    const from = Math.round(t - LIST_WINDOW_MS / 2), to = Math.round(t + LIST_WINDOW_MS / 2);
    this.listing = fetch(`/api/recordings/${encodeURIComponent(this.camera)}?from=${from}&to=${to}`)
      .then(r => r.json())
      .then(list => {
        if (this.destroyed) return;
        this.segments = list;
        this.listed = [from, Math.min(to, Date.now())];
      })
      .catch(err => console.warn(this.camera, err))
      .finally(() => { setTimeout(() => { this.listing = null; }, 2000); });
  }

  // fill appends the next missing segment near the playhead (one at a time,
  // in order).
  fill(t) {
    if (this.loading) return;
    const next = this.segments.find(s => s.end > t - 2000 && s.start < t + AHEAD_MS && !this.appended.has(s.url));
    if (!next) return;
    this.loading = true;
    this.appended.add(next.url);
    this.loadLabels(next);
    fetch(next.url)
      .then(r => { if (!r.ok) throw new Error(`${r.status} ${next.url}`); return r.arrayBuffer(); })
      .then(buf => !this.destroyed && this.sink.append(new Uint8Array(buf)))
      .catch(err => { this.appended.delete(next.url); console.warn(this.camera, err); })
      .finally(() => { this.loading = false; });
  }

  // loadLabels fetches a segment's CV results (a missing sidecar just means
  // no CV ran then).
  loadLabels(seg) {
    fetch(seg.labels)
      .then(r => (r.ok ? r.text() : ''))
      .then(text => {
        if (this.destroyed) return;
        const { results, decisions } = parseSidecar(text);
        for (const r of results) this.labels.add(r);
        for (const d of decisions) this.decisions.add(d);
      })
      .catch(() => {});
  }

  trim(t) {
    const now = performance.now();
    if (this.lastTrim && now - this.lastTrim < 5000) return;
    this.lastTrim = now;
    const lo = t - KEEP_BEHIND_MS, hi = t + AHEAD_MS * 4;
    this.sink.remove(0, lo);
    this.sink.remove(hi, Number.MAX_SAFE_INTEGER / 2);
    // Partly removed segments must be fetched again if needed.
    for (const s of this.segments) if (s.start < lo || s.end > hi) this.appended.delete(s.url);
    this.labels.prune(t);
    this.decisions.prune(t);
  }

  wallMs() { return this.sink.wallMs(); }

  destroy() {
    this.destroyed = true;
    this.sink.destroy();
  }
}
