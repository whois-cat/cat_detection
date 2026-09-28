<script>
  // One timeline for all shown cameras: per camera a row with its recorded
  // spans and detection density, one shared time scale, playhead and hover.
  // Viewport, hover and playhead live in lib/state.svelte.js.
  //
  // Props:
  //   rows: [{label, ranges, events}]
  //     ranges: [[startMs, endMs], ...] recorded spans
  //     events: [{wall_ms, cat, n, dur}] detection counts over [wall_ms, wall_ms + dur)
  //             (density bars, per-cat colours)
  import { onMount, onDestroy } from 'svelte';
  import { view, play, clock, seek, cancelAnimation, writeHash, ZOOM_MIN_MS, ZOOM_MAX_MS } from './lib/state.svelte.js';
  import { DAY_MS, fmtDate, fmtTimeOfDay, fmtDateTime, fmtDuration, isLocalMidnight } from './lib/time.js';
  import { catColor, DEFAULT_CAT_COLOR } from './lib/colors.js';

  let { rows = [] } = $props();

  // Row geometry: recorded strip, then density; the scale below all rows.
  const TOP = 4, AV_H = 5, DEN_GAP = 2, DEN_H = 22, ROW_GAP = 7, SCALE_H = 18;
  const ROW_PITCH = AV_H + DEN_GAP + DEN_H + ROW_GAP;
  const rowY = i => TOP + i * ROW_PITCH;
  const HOVER_WINDOW_PX = 18;
  const TOOLTIP_W = 260;

  let canvas;
  let width = $state(0);

  const sorted = $derived(rows.map(r => ({
    ...r,
    ranges: r.ranges || [],
    events: (r.events || []).slice().sort((a, b) => a.wall_ms - b.wall_ms),
  })));
  const scaleY = $derived(TOP + sorted.length * ROW_PITCH - ROW_GAP + 4);
  const height = $derived(scaleY + SCALE_H);

  const timeToX = ms => (ms - view.from) / (view.to - view.from) * width;
  const xToTime = x => view.from + (x / width) * (view.to - view.from);

  const hoverX = $derived(view.hoverMs === null || !width ? null : timeToX(view.hoverMs));
  const hoverInfo = $derived.by(() => {
    if (view.hoverMs === null || !width) return null;
    const t = view.hoverMs;
    const winMs = HOVER_WINDOW_PX * (view.to - view.from) / width;
    return {
      timeStr: fmtDateTime(t),
      windowStr: `±${fmtDuration(winMs)}`,
      rows: sorted.map(r => {
        const counts = Object.create(null);
        for (let i = lowerBound(r.events, t - winMs), hi = lowerBound(r.events, t + winMs); i < hi; i++) {
          const c = r.events[i].cat || '(unlabelled)';
          counts[c] = (counts[c] || 0) + (r.events[i].n ?? 1);
        }
        return {
          label: r.label,
          recorded: r.ranges.some(([s, e]) => s <= t && t < e),
          cats: Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([cat, n]) => ({ cat, n, color: catColor(cat) })),
        };
      }),
    };
  });

  // ---- non-reactive interaction state ----
  const pointers = new Map();
  let panStart = null;
  let pinchStart = null;
  let dragMoved = false;
  let drawPending = false;
  let ro;

  function scheduleDraw() {
    if (drawPending) return;
    drawPending = true;
    requestAnimationFrame(() => { drawPending = false; draw(); });
  }

  function lowerBound(arr, v) {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m].wall_ms < v) lo = m + 1; else hi = m; }
    return lo;
  }

  function pickTickStepMs(spanMs, targetTicks) {
    const candidates = [
      10_000, 30_000, 60_000, 5 * 60_000, 10 * 60_000, 15 * 60_000, 30 * 60_000,
      3600_000, 2 * 3600_000, 4 * 3600_000, 6 * 3600_000, 12 * 3600_000,
      DAY_MS, 2 * DAY_MS, 7 * DAY_MS, 14 * DAY_MS, 30 * DAY_MS,
    ];
    const ideal = spanMs / targetTicks;
    for (const c of candidates) if (c >= ideal) return c;
    return candidates[candidates.length - 1];
  }

  function makeTicks(stepMs) {
    const out = [];
    if (stepMs >= DAY_MS) {
      const stepDays = Math.round(stepMs / DAY_MS);
      const cur = new Date(view.from);
      cur.setHours(0, 0, 0, 0);
      while (cur.getTime() < view.from) cur.setDate(cur.getDate() + 1);
      while (cur.getTime() <= view.to) { out.push(cur.getTime()); cur.setDate(cur.getDate() + stepDays); }
    } else {
      const midnight = new Date(view.from);
      midnight.setHours(0, 0, 0, 0);
      let t = midnight.getTime() + Math.ceil((view.from - midnight.getTime()) / stepMs) * stepMs;
      for (; t <= view.to; t += stepMs) out.push(t);
    }
    return out;
  }

  // ---- draw ----
  function draw() {
    if (!canvas || width === 0) return;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    sorted.forEach((r, i) => drawRow(ctx, r, rowY(i)));
    drawScaleAndMarkers(ctx);
  }

  function drawRow(ctx, r, y0) {
    // Recorded spans.
    ctx.fillStyle = '#2a1f1f';
    ctx.fillRect(0, y0, width, AV_H);
    ctx.fillStyle = '#3a6e3a';
    for (const [s, e] of r.ranges) {
      const x1 = Math.max(0, timeToX(s));
      const x2 = Math.min(width, timeToX(e));
      if (x2 > x1) ctx.fillRect(x1, y0, Math.max(1, x2 - x1), AV_H);
    }

    // Detection density: time-aligned buckets (bucket boundaries pinned to
    // absolute time so panning doesn't make bars wobble), stacked by cat.
    const ev = r.events;
    const denY = y0 + AV_H + DEN_GAP;
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, denY, width, DEN_H);
    if (!ev.length) return;
    const startIdx = lowerBound(ev, view.from);
    const endIdx = lowerBound(ev, view.to);
    const msPerPx = (view.to - view.from) / width;
    // One bar per pixel, but never narrower than the counts' own buckets
    // (else a steady presence would draw as a comb of thin bars).
    let evDur = 1;
    for (let i = startIdx; i < endIdx; i++) evDur = Math.max(evDur, ev[i].dur ?? 1);
    const bucketMs = Math.max(evDur, Math.round(msPerPx));
    const alignedFloor = Math.floor(view.from / bucketMs) * bucketMs;
    const numBuckets = Math.ceil((view.to - alignedFloor) / bucketMs) + 1;
    const present = new Set();
    for (let i = startIdx; i < endIdx; i++) if (ev[i].cat) present.add(ev[i].cat);
    const order = [...present].sort();
    const catIdx = Object.create(null);
    order.forEach((c, i) => { catIdx[c] = i; });
    const numCats = order.length + 1;
    const buckets = new Uint32Array(numBuckets * numCats);
    const totals = new Uint32Array(numBuckets);
    let maxCount = 1;
    for (let i = startIdx; i < endIdx; i++) {
      const b = Math.floor((ev[i].wall_ms - alignedFloor) / bucketMs);
      if (b < 0 || b >= numBuckets) continue;
      const n = ev[i].n ?? 1;
      buckets[b * numCats + (catIdx[ev[i].cat] ?? order.length)] += n;
      if ((totals[b] += n) > maxCount) maxCount = totals[b];
    }
    const barW = bucketMs / msPerPx;
    const colors = order.map(catColor).concat([DEFAULT_CAT_COLOR]);
    for (let b = 0; b < numBuckets; b++) {
      const t = totals[b];
      if (!t) continue;
      const x = timeToX(alignedFloor + b * bucketMs);
      const fullH = Math.max(1, (t / maxCount) * DEN_H);
      let yBottom = denY + DEN_H;
      for (let c = 0; c < numCats; c++) {
        const n = buckets[b * numCats + c];
        if (!n) continue;
        const segH = (n / t) * fullH;
        ctx.fillStyle = colors[c];
        ctx.fillRect(x, yBottom - segH, barW, segH);
        yBottom -= segH;
      }
    }
  }

  function drawScaleAndMarkers(ctx) {
    const SCL_Y = scaleY, SCL_H = 12;
    const step = pickTickStepMs(view.to - view.from, width / 110);
    const ticks = makeTicks(step);
    ctx.strokeStyle = '#444';
    ctx.font = '11px ui-monospace, monospace';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const t of ticks) {
      const x = timeToX(t);
      ctx.moveTo(x + 0.5, SCL_Y);
      ctx.lineTo(x + 0.5, SCL_Y + 4);
    }
    ctx.stroke();
    if (step < DAY_MS) {
      ctx.strokeStyle = 'rgba(160, 200, 255, 0.18)';
      ctx.beginPath();
      for (const t of ticks) {
        if (!isLocalMidnight(t)) continue;
        const x = timeToX(t);
        ctx.moveTo(x + 0.5, 0);
        ctx.lineTo(x + 0.5, SCL_Y);
      }
      ctx.stroke();
    }
    let datePrefixShown = false;
    for (const t of ticks) {
      let lbl, isDate;
      if (step >= DAY_MS) { lbl = fmtDate(t); isDate = true; }
      else if (isLocalMidnight(t)) { lbl = fmtDate(t); isDate = true; datePrefixShown = true; }
      else if (!datePrefixShown) { lbl = `${fmtDate(t)} ${fmtTimeOfDay(t, step < 60_000)}`; isDate = true; datePrefixShown = true; }
      else { lbl = fmtTimeOfDay(t, step < 60_000); isDate = false; }
      ctx.fillStyle = isDate ? '#cfe2ff' : '#888';
      ctx.fillText(lbl, timeToX(t) + 3, SCL_Y + SCL_H);
    }

    // Hover (shared across timelines).
    if (hoverX !== null && hoverX >= 0 && hoverX <= width) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.fillRect(hoverX - HOVER_WINDOW_PX, 0, HOVER_WINDOW_PX * 2, SCL_Y);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
      ctx.beginPath();
      ctx.moveTo(hoverX + 0.5, 0);
      ctx.lineTo(hoverX + 0.5, SCL_Y);
      ctx.stroke();
    }

    // "Now" marker and playhead.
    const nowX = timeToX(clock.now);
    if (nowX >= 0 && nowX <= width) {
      ctx.fillStyle = 'rgba(192, 57, 43, 0.5)';
      ctx.fillRect(nowX, 0, 1, SCL_Y);
    }
    const x = timeToX(play.playheadMs);
    if (x >= 0 && x <= width) {
      ctx.strokeStyle = ctx.fillStyle = play.live ? '#e04535' : '#ffcc00';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, SCL_Y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 5, 0);
      ctx.lineTo(x + 5, 0);
      ctx.lineTo(x, 6);
      ctx.closePath();
      ctx.fill();
    }
  }

  // ---- pointer / wheel ----
  function breakFollow() {
    view.follow = false;
  }

  function onPointerDown(e) {
    cancelAnimation();
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.offsetX });
    dragMoved = false;
    view.dragging = true;
    if (pointers.size === 1) {
      panStart = { x: e.offsetX, from: view.from, to: view.to };
      pinchStart = null;
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinchStart = { x0: a.x, x1: b.x, from: view.from, to: view.to };
      panStart = null;
      view.hoverMs = null;
    }
  }

  function onPointerMove(e) {
    if (!pointers.has(e.pointerId)) {
      view.hoverMs = xToTime(e.offsetX);
      return;
    }
    pointers.set(e.pointerId, { x: e.offsetX });
    view.hoverMs = null;
    if (panStart && Math.abs(e.offsetX - panStart.x) > 4) dragMoved = true;
    if (pinchStart) dragMoved = true;
    if (!dragMoved) return;
    breakFollow();
    if (pointers.size === 2 && pinchStart) {
      const [a, b] = [...pointers.values()];
      const startDist = Math.abs(pinchStart.x1 - pinchStart.x0);
      const currDist = Math.abs(b.x - a.x);
      if (startDist > 1 && currDist > 1) {
        const startMid = (pinchStart.x0 + pinchStart.x1) / 2;
        const currMid = (a.x + b.x) / 2;
        const startSpan = pinchStart.to - pinchStart.from;
        const span = Math.max(ZOOM_MIN_MS, Math.min(ZOOM_MAX_MS, startSpan * (startDist / currDist)));
        const midTime = pinchStart.from + (startMid / width) * startSpan;
        view.from = midTime - (currMid / width) * span;
        view.to = view.from + span;
      }
    } else if (pointers.size === 1 && panStart) {
      const dms = ((e.offsetX - panStart.x) / width) * (panStart.to - panStart.from);
      view.from = panStart.from - dms;
      view.to = panStart.to - dms;
    }
  }

  function onPointerLeave(e) {
    if (!pointers.has(e.pointerId)) view.hoverMs = null;
  }

  function onPointerUp(e) {
    const had = pointers.has(e.pointerId);
    pointers.delete(e.pointerId);
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
    if (!had) return;
    if (pointers.size === 1) {
      pinchStart = null;
      const [remaining] = [...pointers.values()];
      panStart = { x: remaining.x, from: view.from, to: view.to };
    } else if (pointers.size === 0) {
      const wasPanning = panStart !== null;
      panStart = null;
      pinchStart = null;
      view.dragging = false;
      if (wasPanning && !dragMoved) seek(xToTime(e.offsetX));
      writeHash();
    }
  }

  function onWheel(e) {
    e.preventDefault();
    cancelAnimation();
    const cursorT = xToTime(e.offsetX);
    const factor = Math.pow(1.15, e.deltaY > 0 ? 1 : -1);
    const span = Math.max(ZOOM_MIN_MS, Math.min(ZOOM_MAX_MS, (view.to - view.from) * factor));
    // Zooming keeps following live if the cursor is at the live edge.
    if (!(view.follow && cursorT > clock.now - span * 0.1)) breakFollow();
    view.from = cursorT - (e.offsetX / width) * span;
    view.to = view.from + span;
    writeHash();
  }

  function resize() {
    width = canvas.clientWidth;
    scheduleDraw();
  }

  onMount(() => {
    ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();
  });
  onDestroy(() => ro && ro.disconnect());

  $effect(() => {
    void sorted; void height; void play.playheadMs; void play.live; void clock.now;
    void view.from; void view.to; void view.hoverMs; void width;
    scheduleDraw();
  });
</script>

<div class="timeline">
  {#each sorted as r, i (r.label)}
    <div class="label" style="top: {rowY(i) + AV_H + DEN_GAP + 3}px">{r.label}</div>
  {/each}
  <canvas
    bind:this={canvas}
    style="height:{height}px;"
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    onpointercancel={onPointerUp}
    onpointerleave={onPointerLeave}
    onwheel={onWheel}
  ></canvas>
  {#if hoverInfo && hoverX !== null}
    <div class="tooltip" style={hoverX > width - TOOLTIP_W - 12 ? `right: ${width - hoverX + 8}px` : `left: ${hoverX + 8}px`}>
      <div class="time">{hoverInfo.timeStr} <span class="dim">{hoverInfo.windowStr}</span></div>
      {#each hoverInfo.rows as r (r.label)}
        <div class="row">
          <span class="cam">{r.label}</span>
          {#if !r.recorded}<span class="dim">no recording</span>{/if}
          {#each r.cats as { cat, n, color } (cat)}
            <span class="cat"><span class="swatch" style="background: {color};"></span>{cat} {n}</span>
          {/each}
          {#if r.recorded && !r.cats.length}<span class="dim">—</span>{/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .timeline {
    position: relative;
    background: #181818;
    border: 1px solid #333;
    border-radius: 4px;
  }
  .label {
    position: absolute;
    left: 6px;
    font: 0.75rem ui-monospace, monospace;
    color: #aaa;
    pointer-events: none;
    text-shadow: 0 0 3px #000, 0 0 3px #000;
  }
  canvas {
    width: 100%;
    display: block;
    cursor: grab;
    touch-action: none;
  }
  canvas:active { cursor: grabbing; }
  /* Inside the timeline, next to the hover line. */
  .tooltip {
    position: absolute;
    top: 2px;
    background: #1f1f1f;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 0.3rem 0.5rem;
    font: 0.75rem/1.3 ui-monospace, monospace;
    pointer-events: none;
    z-index: 5;
    width: 260px;
    box-sizing: border-box;
    opacity: 0.92;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
  }
  .tooltip .time { color: #cfe2ff; }
  .tooltip .dim { color: #888; }
  .tooltip .row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.2rem 0.6rem; }
  .tooltip .cam { color: #cfe2ff; min-width: 3.5em; }
  .tooltip .cat { display: inline-flex; align-items: center; gap: 0.3rem; }
  .tooltip .swatch {
    width: 0.8em;
    height: 0.8em;
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 2px;
  }
</style>
