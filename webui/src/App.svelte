<script>
  import { onMount, onDestroy } from 'svelte';
  import Player from './Player.svelte';
  import Timeline from './Timeline.svelte';
  import {
    cams, view, play, clock, visibleCameras, goLive, seek, animating, readHash, writeHash, savePref,
  } from './lib/state.svelte.js';
  import { fmtDateTime, fmtGap, rangeIndexAt, nextRangeStart } from './lib/time.js';

  // A gap in recordings shorter than this plays out in real time ("no
  // recording"); longer ones show their length for GAP_NOTICE_MS, then jump.
  const SMALL_GAP_MS = 2000;
  const GAP_NOTICE_MS = 2000;
  // History playback that catches up with now switches to live.
  const CATCH_UP_MS = 3000;
  const STATUS_EVERY_MS = 2000;
  const RANGES_EVERY_MS = 10_000;
  const RATES = [0.5, 1, 2, 4, 8, 16];

  const shown = $derived(cams.list.filter(c => view.selected === 'all' || c.id === view.selected));

  let raf, statusTimer, rangesTimer, rangesDebounce;
  let lastFrame = performance.now();

  // ---- data ----

  async function loadStatus() {
    try {
      cams.status = (await (await fetch('/api/status')).json()).cameras;
    } catch { /* keep last known */ }
  }

  // loadRanges loads recorded spans and detection counts around the viewport.
  async function loadRanges() {
    const span = view.to - view.from;
    const from = Math.round(view.from - span), to = Math.round(view.to + span);
    // About one detection bucket per timeline pixel.
    const bucket = Math.max(1000, Math.floor(span / 2000));
    const q = `from=${from}&to=${to}`;
    await Promise.all(cams.list.map(async c => {
      const id = encodeURIComponent(c.id);
      try {
        const [ranges, dets] = await Promise.all([
          fetch(`/api/ranges/${id}?${q}`).then(r => r.json()),
          fetch(`/api/detections/${id}?${q}&bucket=${bucket}`).then(r => r.json()),
        ]);
        cams.ranges[c.id] = ranges;
        cams.events[c.id] = dets.map(([wall_ms, cat, n]) => ({ wall_ms, cat, n, dur: bucket }));
      } catch { /* keep last known */ }
    }));
  }

  // ---- playback clock ----

  function frame(now) {
    const dt = now - lastFrame;
    lastFrame = now;
    clock.now = Date.now();

    if (play.live) {
      play.playheadMs = clock.now;
    } else if (play.playing && !play.gap) {
      const ids = visibleCameras();
      if (!ids.some(id => play.waiting[id])) play.playheadMs += dt * play.rate;
      skipGaps(ids);
      if (play.playheadMs >= clock.now - CATCH_UP_MS) goLive();
    }

    if (!animating() && !view.dragging) {
      const span = view.to - view.from;
      if (view.follow) {
        view.to = clock.now + span * 0.05;
        view.from = view.to - span;
      } else if (play.playheadMs > view.to - span * 0.05 || play.playheadMs < view.from) {
        // Keep the history playhead in view.
        view.from = play.playheadMs - span * 0.2;
        view.to = view.from + span;
      }
    }
    raf = requestAnimationFrame(frame);
  }

  function skipGaps(ids) {
    const t = play.playheadMs;
    if (ids.some(id => rangeIndexAt(cams.ranges[id] || [], t) >= 0)) return;
    const next = Math.min(...ids.map(id => nextRangeStart(cams.ranges[id] || [], t) ?? Infinity));
    if (!Number.isFinite(next) || next - t <= SMALL_GAP_MS) return;
    play.gap = { lengthMs: next - t, target: next };
    setTimeout(() => {
      if (play.gap?.target !== next) return;
      play.playheadMs = next;
      play.gap = null;
    }, GAP_NOTICE_MS);
  }

  // ---- controls ----

  function select(id) {
    view.selected = view.selected === id ? 'all' : id;
    writeHash();
  }

  function togglePlay() {
    if (play.live) {
      // Pausing live freezes the current moment as history.
      seek(clock.now - 1);
      play.playing = false;
    } else {
      play.playing = !play.playing;
    }
    writeHash();
  }

  function jump(deltaMs) {
    seek((play.live ? clock.now : play.playheadMs) + deltaMs);
    writeHash();
  }

  function onKey(e) {
    if (e.target.closest('input, select, textarea')) return;
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    else if (e.key === 'ArrowLeft') jump(e.shiftKey ? -60_000 : -10_000);
    else if (e.key === 'ArrowRight') jump(e.shiftKey ? 60_000 : 10_000);
    else if (e.key === 'l') goLive();
  }

  // ---- lifecycle ----

  onMount(async () => {
    cams.list = await (await fetch('/api/cameras')).json();
    const h = readHash();
    const now = Date.now();
    clock.now = now;
    view.selected = h.cam && (h.cam === 'all' || cams.list.some(c => c.id === h.cam)) ? h.cam : 'all';
    if (h.from !== null && h.to !== null && h.to > h.from) {
      view.from = h.from;
      view.to = h.to;
      view.follow = false;
    } else {
      view.to = now + 3 * 60_000;
      view.from = view.to - 60 * 60_000;
    }
    if (h.t !== null) seek(h.t); else play.playheadMs = now;
    loadStatus();
    loadRanges();
    statusTimer = setInterval(loadStatus, STATUS_EVERY_MS);
    rangesTimer = setInterval(loadRanges, RANGES_EVERY_MS);
    raf = requestAnimationFrame(frame);
  });

  onDestroy(() => {
    cancelAnimationFrame(raf);
    clearInterval(statusTimer);
    clearInterval(rangesTimer);
  });

  // Reload ranges when the viewport settles somewhere new.
  $effect(() => {
    void view.from; void view.to;
    if (view.follow) return; // the periodic refresh covers following live
    clearTimeout(rangesDebounce);
    rangesDebounce = setTimeout(loadRanges, 300);
  });
</script>

<svelte:window onkeydown={onKey} />

<header>
  <nav>
    <button class:active={view.selected === 'all'} onclick={() => { view.selected = 'all'; writeHash(); }}>All</button>
    {#each cams.list as c (c.id)}
      <button class:active={view.selected === c.id} onclick={() => select(c.id)}>
        <span class="dot" class:ok={cams.status[c.id]?.connected}></span>{c.id}
      </button>
    {/each}
  </nav>
  <div class="transport">
    <button onclick={() => jump(-10_000)} title="Back 10 s (←, shift: 1 min)">⏪</button>
    <button onclick={togglePlay} title="Play/pause (space)">{play.live || play.playing ? '⏸' : '▶'}</button>
    <button onclick={() => jump(10_000)} title="Forward 10 s (→, shift: 1 min)" disabled={play.live}>⏩</button>
    <select bind:value={play.rate} title="Playback speed" disabled={play.live}>
      {#each RATES as r (r)}<option value={r}>{r}×</option>{/each}
    </select>
    <span class="clock">{fmtDateTime(play.playheadMs)}</span>
    <label class="toggle" title="Per-camera CV results and decider state for the shown frame">
      <input type="checkbox" bind:checked={view.details} onchange={() => savePref('details', view.details)} /> details
    </label>
  </div>
</header>

<main class:grid={view.selected === 'all'} class:details={view.details}>
  {#each shown as c (c.id)}
    <Player camera={c.id} onselect={() => select(c.id)} side={view.selected !== 'all'} />
  {/each}
  {#if play.gap}
    <div class="gap">&gt;&gt; {fmtGap(play.gap.lengthMs)}</div>
  {/if}
</main>

<section class="timelines">
  <div class="stack">
    {#each shown as c (c.id)}
      <Timeline label={c.id} ranges={cams.ranges[c.id] || []} events={cams.events[c.id] || []} />
    {/each}
  </div>
  <!-- At the timelines' right end, where "now" is. -->
  <button class="live" class:on={play.live} onclick={goLive} title="Go live (l)">LIVE</button>
</section>

<style>
  :global(body) {
    margin: 0;
    background: #111;
    color: #ddd;
    font-family: system-ui, sans-serif;
  }
  :global(#app) {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 8px;
    min-height: 100vh;
    box-sizing: border-box;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 8px;
  }
  nav, .transport { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
  button, select {
    background: #222;
    color: #ddd;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 4px 10px;
    font: inherit;
    cursor: pointer;
  }
  button:disabled, select:disabled { opacity: 0.4; cursor: default; }
  button.active { background: #2f4f6f; border-color: #4f7faf; }
  .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #a33;
    margin-right: 6px;
  }
  .dot.ok { background: #3a3; }
  .clock { font-family: ui-monospace, monospace; padding: 0 6px; }
  .live { font-weight: 700; letter-spacing: 1px; padding: 0 1rem; }
  .live.on { background: #c0392b; border-color: #e04535; color: #fff; }
  main {
    position: relative;
    display: grid;
    gap: 8px;
    /* Single camera: as large as fits above the timeline. */
    max-height: calc(100vh - 200px);
    aspect-ratio: 16 / 9;
    margin: 0 auto;
    width: 100%;
    max-width: calc((100vh - 200px) * 16 / 9);
  }
  /* Single camera with details: the panel sits beside the video. */
  main.details:not(.grid) {
    aspect-ratio: auto;
    max-width: calc((100vh - 200px) * 16 / 9 + 23rem);
  }
  .toggle { display: flex; align-items: center; gap: 4px; padding: 0 6px; cursor: pointer; }
  main.grid {
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr));
    aspect-ratio: auto;
    max-height: none;
    max-width: none;
  }
  .gap {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    font: 700 4rem ui-monospace, monospace;
    color: #fff;
    background: rgba(0, 0, 0, 0.5);
    pointer-events: none;
  }
  .timelines { display: flex; gap: 4px; }
  .stack { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 4px; }
</style>
