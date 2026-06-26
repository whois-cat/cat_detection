<script>
  // Timeline component (Svelte 5, runes).
  //
  // Props:
  //   ranges:      Array<{from_ms, to_ms}>     intervals where data exists
  //   events:      Array<{wall_ms, cat?, ...}> detections (clear events filtered out)
  //   nowMs:       number                      wall-clock anchor for the right edge
  //   playheadMs:  number                      current playback time
  //   follow:      boolean                     when true, viewport advances with nowMs
  //   onseek:      (wallMs) => void            called on click-to-seek
  //   onenterLive: () => void                  called when user enters the LIVE dock
  //   onbreakFollow: () => void                called when user pans/zooms during follow
  //
  // URL hash: #from=<ms>&to=<ms> — pan/zoom state, restored on reload.
  import { onMount, onDestroy, untrack } from 'svelte';

  let {
    ranges = [],
    events = [],
    nowMs,
    playheadMs,
    follow = false,
    onseek,
    onenterLive,
    onbreakFollow,
  } = $props();

  // ---- constants ----
  const LIVE_SNAP_MS = 5_000;
  const ZOOM_MIN_MS = 30_000;
  const ZOOM_MAX_MS = 30 * 24 * 3600_000;
  const HOVER_WINDOW_PX = 18;
  const DAY_MS = 86_400_000;

  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  // Colour per identity label, derived deterministically from the (arbitrary)
  // label string — stable per name, distinct across names, no hardcoded cat
  // names/classes. Null/empty (no identity) uses the neutral default.
  const DEFAULT_CAT_COLOR = '#00ff88';
  function catColor(c) {
    if (!c) return DEFAULT_CAT_COLOR;
    let h = 0;
    for (let i = 0; i < c.length; i++) h = (h * 31 + c.charCodeAt(i)) >>> 0;
    return `hsl(${h % 360} ${55 + (h >> 9) % 25}% ${50 + (h >> 17) % 20}%)`;
  }

  // ---- DOM + reactive state ----
  let canvas;
  let container;
  let width = $state(0);
  const height = 80;

  let viewFromMs = $state(0);
  let viewToMs = $state(0);

  let hoverX = $state(null);
  let hoverInfo = $state(null);
  let tooltipVisible = $derived(hoverX !== null && hoverInfo !== null);

  // Detections sorted by wall_ms for binary-search bounding. Filters out
  // "clear" events (no boxes — those are overlay signals, not detections).
  const eventsSorted = $derived(
    events
      .filter(e => !e.boxes || e.boxes.length > 0)
      .slice()
      .sort((a, b) => a.wall_ms - b.wall_ms)
  );

  // ---- non-reactive (only touched in handlers; canvas redraws are imperative) ----
  const pointers = new Map();
  let panStart = null;
  let pinchStart = null;
  let dragMoved = false;
  let anim = null;
  let _lastFollowNow = 0;
  let hashTimer = 0;
  let drawPending = false;
  let ro;

  // ---- helpers ----
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

  function timeToX(ms) { return (ms - viewFromMs) / (viewToMs - viewFromMs) * width; }
  function xToTime(x) { return viewFromMs + (x / width) * (viewToMs - viewFromMs); }

  function pad2(n) { return n.toString().padStart(2, '0'); }
  function fmtTimeOfDay(ms) { const d = new Date(ms); return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`; }
  function fmtDate(ms)      { const d = new Date(ms); return `${MONTHS[d.getMonth()]} ${d.getDate()}`; }
  function isLocalMidnight(ms) {
    const d = new Date(ms);
    return d.getHours() === 0 && d.getMinutes() === 0 && d.getSeconds() === 0 && d.getMilliseconds() === 0;
  }
  function fmtFullDateTime(ms) {
    const d = new Date(ms);
    return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
  }
  function fmtDuration(ms) {
    const abs = Math.abs(ms);
    if (abs < 1000)       return `${Math.round(ms)}ms`;
    if (abs < 60_000)     return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)}s`;
    if (abs < 3600_000)   return `${(ms / 60_000).toFixed(ms < 600_000 ? 1 : 0)}min`;
    if (abs < 86_400_000) return `${(ms / 3600_000).toFixed(ms < 36_000_000 ? 1 : 0)}h`;
    return `${(ms / 86_400_000).toFixed(1)}d`;
  }

  function pickTickStepMs(spanMs, targetTicks) {
    const candidates = [
      60_000, 5*60_000, 10*60_000, 15*60_000, 30*60_000,
      3600_000, 2*3600_000, 4*3600_000, 6*3600_000, 12*3600_000,
      DAY_MS, 2*DAY_MS, 7*DAY_MS, 14*DAY_MS, 30*DAY_MS,
    ];
    const ideal = spanMs / targetTicks;
    for (const c of candidates) if (c >= ideal) return c;
    return candidates[candidates.length - 1];
  }

  function makeTicks(stepMs) {
    const out = [];
    if (stepMs >= DAY_MS) {
      const stepDays = Math.round(stepMs / DAY_MS);
      const cur = new Date(viewFromMs);
      cur.setHours(0, 0, 0, 0);
      while (cur.getTime() < viewFromMs) cur.setDate(cur.getDate() + 1);
      while (cur.getTime() <= viewToMs) {
        out.push(cur.getTime());
        cur.setDate(cur.getDate() + stepDays);
      }
    } else {
      const midnight = new Date(viewFromMs);
      midnight.setHours(0, 0, 0, 0);
      let t = midnight.getTime();
      while (t < viewFromMs) t += stepMs;
      while (t <= viewToMs) { out.push(t); t += stepMs; }
    }
    return out;
  }

  // ---- URL hash sync ----
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    const f = parseInt(p.get('from'), 10);
    const t = parseInt(p.get('to'), 10);
    if (!isNaN(f) && !isNaN(t) && t > f) return [f, t];
    return null;
  }
  function writeHash() {
    clearTimeout(hashTimer);
    hashTimer = setTimeout(() => {
      const p = new URLSearchParams(location.hash.slice(1));
      p.set('from', Math.round(viewFromMs).toString());
      p.set('to', Math.round(viewToMs).toString());
      const next = '#' + p.toString();
      if (location.hash !== next) history.replaceState(null, '', next);
    }, 150);
  }
  function onHashChange() {
    const h = readHash();
    if (h && (h[0] !== viewFromMs || h[1] !== viewToMs)) {
      viewFromMs = h[0];
      viewToMs = h[1];
      scheduleDraw();
    }
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

    const RET_Y = 2,  RET_H = 14;
    const DEN_Y = 18, DEN_H = 34;
    const SCL_Y = 56, SCL_H = 12;

    ctx.fillStyle = '#2a1f1f';
    ctx.fillRect(0, RET_Y, width, RET_H);
    ctx.fillStyle = '#3a6e3a';
    for (const r of ranges) {
      const x1 = Math.max(0, timeToX(r.from_ms));
      const x2 = Math.min(width, timeToX(r.to_ms));
      if (x2 > x1) ctx.fillRect(x1, RET_Y, x2 - x1, RET_H);
    }

    // Detection density. Two rendering paths — flip the flag to switch.
    //   true  → additive 1-px lines per event (smooth, soft)
    //   false → time-aligned buckets, stacked by cat (default)
    const USE_ADDITIVE_LINES = false;
    if (eventsSorted.length) {
      const startIdx = lowerBound(eventsSorted, viewFromMs);
      const endIdx   = lowerBound(eventsSorted, viewToMs);
      if (USE_ADDITIVE_LINES) {
        ctx.globalAlpha = 0.6;
        for (let i = startIdx; i < endIdx; i++) {
          const ev = eventsSorted[i];
          ctx.fillStyle = catColor(ev.cat);
          ctx.fillRect(timeToX(ev.wall_ms) - 0.5, DEN_Y, 1, DEN_H);
        }
        ctx.globalAlpha = 1;
      } else {
        const msPerPx = (viewToMs - viewFromMs) / width;
        const targetBucketPx = 1;
        const bucketMs = Math.max(1, Math.round(targetBucketPx * msPerPx));
        const alignedFloor = Math.floor(viewFromMs / bucketMs) * bucketMs;
        const numBuckets = Math.ceil((viewToMs - alignedFloor) / bucketMs) + 1;
        // Build a stable cat index from whatever labels are present in view —
        // no hardcoded class list. Sorted for deterministic stacking/colours.
        const present = new Set();
        for (let i = startIdx; i < endIdx; i++) {
          if (eventsSorted[i].cat) present.add(eventsSorted[i].cat);
        }
        const order = [...present].sort();
        const catIdx = Object.create(null);
        for (let i = 0; i < order.length; i++) catIdx[order[i]] = i;
        const UNK = order.length;
        const numCats = order.length + 1;
        const buckets = new Uint16Array(numBuckets * numCats);
        const totals  = new Uint16Array(numBuckets);
        let maxCount = 1;
        for (let i = startIdx; i < endIdx; i++) {
          const ev = eventsSorted[i];
          const bIdx = Math.floor((ev.wall_ms - alignedFloor) / bucketMs);
          if (bIdx < 0 || bIdx >= numBuckets) continue;
          const cIdx = catIdx[ev.cat] ?? UNK;
          buckets[bIdx * numCats + cIdx]++;
          totals[bIdx]++;
          if (totals[bIdx] > maxCount) maxCount = totals[bIdx];
        }
        const barW = bucketMs / msPerPx;
        const colors = order.map(catColor).concat([DEFAULT_CAT_COLOR]);
        for (let b = 0; b < numBuckets; b++) {
          const t = totals[b];
          if (!t) continue;
          const x = timeToX(alignedFloor + b * bucketMs);
          const fullH = Math.max(1, (t / maxCount) * DEN_H);
          let yBottom = DEN_Y + DEN_H;
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
    }

    const span = viewToMs - viewFromMs;
    const step = pickTickStepMs(span, width / 110);
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
      const x = timeToX(t);
      let label, isDate;
      if (step >= DAY_MS) {
        label = fmtDate(t); isDate = true;
      } else if (isLocalMidnight(t)) {
        label = fmtDate(t); isDate = true; datePrefixShown = true;
      } else if (!datePrefixShown) {
        label = `${fmtDate(t)} ${fmtTimeOfDay(t)}`; isDate = true; datePrefixShown = true;
      } else {
        label = fmtTimeOfDay(t); isDate = false;
      }
      ctx.fillStyle = isDate ? '#cfe2ff' : '#888';
      ctx.fillText(label, x + 3, SCL_Y + SCL_H);
    }

    if (hoverX !== null && hoverX >= 0 && hoverX <= width) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.fillRect(hoverX - HOVER_WINDOW_PX, 0, HOVER_WINDOW_PX * 2, SCL_Y);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hoverX + 0.5, 0);
      ctx.lineTo(hoverX + 0.5, SCL_Y);
      ctx.stroke();
    }

    if (playheadMs >= viewFromMs && playheadMs <= viewToMs) {
      const x = timeToX(playheadMs);
      ctx.strokeStyle = '#ffcc00';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, SCL_Y);
      ctx.stroke();
      ctx.fillStyle = '#ffcc00';
      ctx.beginPath();
      ctx.moveTo(x - 5, 0);
      ctx.lineTo(x + 5, 0);
      ctx.lineTo(x, 6);
      ctx.closePath();
      ctx.fill();
    }
  }

  // ---- hover ----
  function updateHover(x) {
    if (!eventsSorted.length || width === 0) { hoverInfo = null; return; }
    const t = xToTime(x);
    const winMs = HOVER_WINDOW_PX * (viewToMs - viewFromMs) / width;
    const lo = lowerBound(eventsSorted, t - winMs);
    const hi = lowerBound(eventsSorted, t + winMs);
    const counts = Object.create(null);
    let total = 0;
    for (let i = lo; i < hi; i++) {
      const c = eventsSorted[i].cat || '(unlabelled)';
      counts[c] = (counts[c] || 0) + 1;
      total++;
    }
    hoverInfo = {
      timeStr: fmtFullDateTime(t),
      windowStr: `±${fmtDuration(winMs)}`,
      total,
      cats: Object.entries(counts)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cat, n]) => ({ cat, n, color: catColor(cat) })),
    };
  }

  // ---- pointer / wheel ----
  function onMouseDown(e) {
    cancelAnim();
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.offsetX });
    dragMoved = false;
    if (pointers.size === 1) {
      panStart = { x: e.offsetX, viewFrom: viewFromMs, viewTo: viewToMs };
      pinchStart = null;
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinchStart = { x0: a.x, x1: b.x, viewFrom: viewFromMs, viewTo: viewToMs };
      panStart = null;
      hoverX = null;
    }
  }

  function onMouseMove(e) {
    if (!pointers.has(e.pointerId)) {
      hoverX = e.offsetX;
      updateHover(e.offsetX);
      scheduleDraw();
      return;
    }
    pointers.set(e.pointerId, { x: e.offsetX });
    hoverX = null;

    if (panStart && Math.abs(e.offsetX - panStart.x) > 4) dragMoved = true;
    if (pinchStart) dragMoved = true;

    if (follow) onbreakFollow?.();
    if (pointers.size === 2 && pinchStart) {
      const [a, b] = [...pointers.values()];
      const startDist = Math.abs(pinchStart.x1 - pinchStart.x0);
      const currDist  = Math.abs(b.x - a.x);
      if (startDist > 1 && currDist > 1) {
        const startMid = (pinchStart.x0 + pinchStart.x1) / 2;
        const currMid  = (a.x + b.x) / 2;
        const startSpan = pinchStart.viewTo - pinchStart.viewFrom;
        const newSpan = Math.max(ZOOM_MIN_MS, Math.min(ZOOM_MAX_MS, startSpan * (startDist / currDist)));
        const midTime = pinchStart.viewFrom + (startMid / width) * startSpan;
        const newFrom = midTime - (currMid / width) * newSpan;
        viewFromMs = newFrom;
        viewToMs = newFrom + newSpan;
        scheduleDraw();
      }
    } else if (pointers.size === 1 && panStart) {
      const dx = e.offsetX - panStart.x;
      const dms = (dx / width) * (panStart.viewTo - panStart.viewFrom);
      viewFromMs = panStart.viewFrom - dms;
      viewToMs = panStart.viewTo - dms;
      scheduleDraw();
    }
  }

  function onMouseLeave(e) {
    if (!pointers.has(e.pointerId)) {
      hoverX = null;
      scheduleDraw();
    }
  }

  function onMouseUp(e) {
    const had = pointers.has(e.pointerId);
    pointers.delete(e.pointerId);
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
    if (!had) return;
    if (pointers.size === 1) {
      pinchStart = null;
      const [remaining] = [...pointers.values()];
      panStart = { x: remaining.x, viewFrom: viewFromMs, viewTo: viewToMs };
    } else if (pointers.size === 0) {
      const wasPanning = panStart !== null;
      panStart = null; pinchStart = null;
      if (wasPanning && !dragMoved) {
        const t = xToTime(e.offsetX);
        maybeSeek(t);
      }
      writeHash();
    }
  }

  function onWheel(e) {
    e.preventDefault();
    cancelAnim();
    if (follow) onbreakFollow?.();
    const cursorT = xToTime(e.offsetX);
    const factor = Math.pow(1.15, e.deltaY > 0 ? 1 : -1);
    const span = Math.max(ZOOM_MIN_MS, Math.min(ZOOM_MAX_MS, (viewToMs - viewFromMs) * factor));
    const left = cursorT - (e.offsetX / width) * span;
    viewFromMs = left;
    viewToMs = left + span;
    scheduleDraw();
    writeHash();
  }

  function maybeSeek(t) {
    if (t >= nowMs - LIVE_SNAP_MS) goLive();
    else onseek?.(t);
  }

  // ---- animated view transition ----
  function animateTo(targetFromMs, targetToMs, durationMs = 380) {
    anim = {
      start: performance.now(),
      dur: durationMs,
      fromFrom: viewFromMs, fromTo: viewToMs,
      toFrom: targetFromMs, toTo: targetToMs,
    };
    requestAnimationFrame(tickAnim);
  }
  function tickAnim(now) {
    if (!anim) return;
    const t = Math.min(1, (now - anim.start) / anim.dur);
    const e = 1 - Math.pow(1 - t, 3);
    viewFromMs = anim.fromFrom + (anim.toFrom - anim.fromFrom) * e;
    viewToMs   = anim.fromTo   + (anim.toTo   - anim.fromTo)   * e;
    scheduleDraw();
    if (t < 1) requestAnimationFrame(tickAnim);
    else { anim = null; writeHash(); }
  }
  function cancelAnim() { anim = null; }

  function goLive() {
    const span = viewToMs - viewFromMs;
    const rightSlack = span * 0.05;
    const toTo = nowMs + rightSlack;
    animateTo(toTo - span, toTo);
    onenterLive?.();
  }

  // ---- lifecycle ----
  function resize() {
    if (!canvas) return;
    width = canvas.clientWidth;
    scheduleDraw();
  }

  onMount(() => {
    const hash = readHash();
    if (hash) {
      viewFromMs = hash[0];
      viewToMs = hash[1];
    } else {
      viewToMs = nowMs;
      viewFromMs = nowMs - 3600_000;
    }
    ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();
    window.addEventListener('hashchange', onHashChange);
  });
  onDestroy(() => {
    ro && ro.disconnect();
    window.removeEventListener('hashchange', onHashChange);
  });

  // ---- effects ----

  // Redraw on prop or state changes.
  $effect(() => {
    // Read everything we depend on so $effect picks them up.
    void events; void ranges; void playheadMs; void nowMs;
    void viewFromMs; void viewToMs; void hoverX;
    if (canvas) scheduleDraw();
  });

  // Auto-follow: when `follow` is true, shift the viewport by nowMs's delta
  // so the right edge tracks real time. View writes are untracked so the
  // effect doesn't re-fire on its own writes.
  $effect(() => {
    if (!follow || !nowMs) {
      _lastFollowNow = 0;
      return;
    }
    if (_lastFollowNow && nowMs > _lastFollowNow) {
      const delta = nowMs - _lastFollowNow;
      untrack(() => {
        viewFromMs += delta;
        viewToMs += delta;
      });
      scheduleDraw();
    }
    _lastFollowNow = nowMs;
  });

  // Keep playhead inside the view (for history playback advancing past the
  // right edge). Skipped during user pointer interaction or LIVE animation.
  $effect(() => {
    if (!playheadMs || !viewToMs) return;
    if (pointers.size > 0 || anim) return;
    const span = viewToMs - viewFromMs;
    const slack = span * 0.05;
    if (playheadMs > viewToMs - slack) {
      const shift = playheadMs - (viewToMs - slack);
      untrack(() => {
        viewFromMs += shift;
        viewToMs += shift;
      });
      scheduleDraw();
    } else if (playheadMs < viewFromMs + slack) {
      const shift = (viewFromMs + slack) - playheadMs;
      untrack(() => {
        viewFromMs -= shift;
        viewToMs -= shift;
      });
      scheduleDraw();
    }
  });
</script>

<div class="timeline" bind:this={container}>
  <canvas
    bind:this={canvas}
    style="height:{height}px;"
    onpointerdown={onMouseDown}
    onpointermove={onMouseMove}
    onpointerup={onMouseUp}
    onpointercancel={onMouseUp}
    onpointerleave={onMouseLeave}
    onwheel={onWheel}
  ></canvas>
  {#if tooltipVisible}
    <div class="tooltip"
         style="left: {Math.max(0, Math.min(width - 200, hoverX - 100))}px;">
      <div class="time">{hoverInfo.timeStr}</div>
      <div class="dim">{hoverInfo.windowStr}</div>
      {#if hoverInfo.total === 0}
        <div class="dim">no detections</div>
      {:else}
        <div>total: {hoverInfo.total}</div>
        {#each hoverInfo.cats as { cat, n, color } (cat)}
          <div class="row">
            <span class="swatch" style="background: {color};"></span>
            <span>{cat}: {n}</span>
          </div>
        {/each}
      {/if}
    </div>
  {/if}
  <button class="now" onclick={goLive} title="Jump to live">LIVE</button>
</div>

<style>
  .timeline {
    position: relative;
    display: flex;
    align-items: stretch;
    background: #181818;
    border: 1px solid #333;
    border-radius: 4px;
  }
  canvas {
    flex: 1;
    min-width: 0;
    display: block;
    cursor: grab;
    touch-action: none;
    border-radius: 4px 0 0 4px;
  }
  canvas:active { cursor: grabbing; }
  .now {
    background: #c0392b;
    color: white;
    border: none;
    padding: 0 1rem;
    font-weight: 700;
    cursor: pointer;
    letter-spacing: 1px;
    border-radius: 0 4px 4px 0;
  }
  .now:hover { background: #e04535; }

  .tooltip {
    position: absolute;
    bottom: calc(100% + 6px);
    background: #1f1f1f;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 0.4rem 0.6rem;
    font: 0.78rem/1.3 ui-monospace, monospace;
    pointer-events: none;
    z-index: 5;
    min-width: 140px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  .tooltip .time { color: #cfe2ff; }
  .tooltip .dim  { color: #888; }
  .tooltip .row  { display: flex; align-items: center; gap: 0.35rem; }
  .tooltip .swatch {
    display: inline-block;
    width: 0.8em; height: 0.8em;
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 2px;
    flex-shrink: 0;
  }
</style>
