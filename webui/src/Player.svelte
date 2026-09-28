<script>
  // One camera's video: live (WebSocket) or history (segments), following
  // the shared playback state.
  import { onMount, onDestroy, untrack } from 'svelte';
  import { LiveSource, HistorySource } from './lib/sources.js';
  import { play, cams } from './lib/state.svelte.js';
  import { fmtDateTime } from './lib/time.js';
  import { LabelStore, DecisionStore, topCat, describeDecision } from './lib/labels.js';
  import { catColor } from './lib/colors.js';

  let { camera, onselect } = $props();

  // History players re-sync to the playhead this often.
  const UPDATE_MS = 200;
  // A label older than this (relative to the frame shown) is drawn dashed:
  // the CV hasn't seen this frame, the box may have moved.
  const STALE_MS = 400;

  let video;
  let canvas;
  const labels = new LabelStore();
  const decisionStore = new DecisionStore();
  let decisions = $state([]); // latest per feeder at the shown frame
  let frameHandle;
  let source = null;
  let mode = null;
  let hasData = $state(true);
  let shownMs = $state(null);
  let timer;

  const status = $derived(cams.status[camera]);
  const offline = $derived(play.live && status && !status.connected);

  // (Re)create the source when switching between live and history.
  $effect(() => {
    const want = play.live ? 'live' : 'history';
    untrack(() => {
      if (!video || mode === want) return;
      source?.destroy();
      labels.clear();
      decisionStore.clear();
      const Source = want === 'live' ? LiveSource : HistorySource;
      source = new Source(video, camera, labels, decisionStore);
      mode = want;
      hasData = true;
    });
  });

  function update() {
    if (!source) return;
    if (mode === 'history') {
      const r = source.update(play.playheadMs, play.playing && !play.gap, play.rate);
      hasData = r.hasData;
      if (!!play.waiting[camera] !== r.waiting) play.waiting[camera] = r.waiting;
    } else {
      hasData = true;
      if (play.waiting[camera]) play.waiting[camera] = false;
    }
    shownMs = source.wallMs();
    const next = shownMs == null ? [] : decisionStore.at(shownMs);
    if (JSON.stringify(next) !== JSON.stringify(decisions)) decisions = next;
    // Also redraw here: labels can arrive after a paused frame was shown.
    drawOverlay(video.currentTime);
  }

  // drawOverlay draws the boxes of the newest result at or before the frame
  // at mediaTime, in the video's displayed (letterboxed) area.
  function drawOverlay(mediaTime) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const ew = canvas.clientWidth, eh = canvas.clientHeight;
    if (canvas.width !== ew * dpr || canvas.height !== eh * dpr) {
      canvas.width = ew * dpr;
      canvas.height = eh * dpr;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, ew, eh);
    const frameMs = source?.sink.frameWallMs(mediaTime);
    const r = frameMs != null && labels.at(frameMs);
    if (!r || !r.dets.length || !video.videoWidth) return;
    const scale = Math.min(ew / video.videoWidth, eh / video.videoHeight);
    const dw = video.videoWidth * scale, dh = video.videoHeight * scale;
    const ox = (ew - dw) / 2, oy = (eh - dh) / 2;
    ctx.setLineDash(frameMs - r.ms > STALE_MS ? [6, 4] : []);
    ctx.lineWidth = 2;
    ctx.font = '12px ui-monospace, monospace';
    for (const d of r.dets) {
      const [bx, by, bw, bh] = d.box;
      const x = ox + bx * dw, y = oy + by * dh;
      const cat = topCat(d);
      const p = d.cats?.[cat];
      // identity probability · detector's confidence that this is a cat at all
      const text = `${cat}${p != null ? ` ${Math.round(p * 100)}%` : ''} · det ${d.score.toFixed(2)}`;
      ctx.strokeStyle = ctx.fillStyle = catColor(cat);
      ctx.strokeRect(x, y, bw * dw, bh * dh);
      const tw = ctx.measureText(text).width + 6;
      const ty = y >= 16 ? y - 16 : y + bh * dh;
      ctx.fillRect(x - 1, ty, tw, 16);
      ctx.fillStyle = '#000';
      ctx.fillText(text, x + 2, ty + 12);
    }
  }

  function onVideoFrame(_now, meta) {
    drawOverlay(meta.mediaTime);
    frameHandle = video.requestVideoFrameCallback(onVideoFrame);
  }

  onMount(() => {
    timer = setInterval(update, UPDATE_MS);
    if ('requestVideoFrameCallback' in video) frameHandle = video.requestVideoFrameCallback(onVideoFrame);
  });
  onDestroy(() => {
    clearInterval(timer);
    if (frameHandle) video.cancelVideoFrameCallback(frameHandle);
    source?.destroy();
    delete play.waiting[camera];
  });
</script>

<div class="player">
  <!-- svelte-ignore a11y_media_has_caption -->
  <video bind:this={video} muted playsinline></video>
  <canvas class="overlay" bind:this={canvas}></canvas>
  <button class="label" onclick={onselect} title="Show only this camera">{camera}</button>
  {#if shownMs}<div class="time">{fmtDateTime(shownMs)}</div>{/if}
  {#if decisions.length}
    <div class="decisions">
      {#each decisions as d (d.feeder)}<div class:open={d.state === 'open'}>{describeDecision(d)}</div>{/each}
    </div>
  {/if}
  {#if offline}
    <div class="notice">camera offline{status.last_error ? `: ${status.last_error}` : ''}</div>
  {:else if !hasData}
    <div class="notice">no recording</div>
  {:else if play.waiting[camera]}
    <div class="notice dim">loading…</div>
  {/if}
</div>

<style>
  .player {
    position: relative;
    aspect-ratio: 16 / 9;
    background: #000;
    border-radius: 4px;
    overflow: hidden;
    min-width: 0;
  }
  video, .overlay {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  .overlay { pointer-events: none; }
  .label, .time {
    position: absolute;
    bottom: 6px;
    font: 0.8rem ui-monospace, monospace;
    color: #eee;
    background: rgba(0, 0, 0, 0.55);
    border-radius: 3px;
    padding: 2px 6px;
  }
  .label { left: 6px; border: none; cursor: pointer; }
  .label:hover { background: rgba(0, 0, 0, 0.8); }
  .time { right: 6px; }
  .decisions {
    position: absolute;
    top: 34px;
    right: 6px;
    font: 0.75rem ui-monospace, monospace;
    color: #eee;
    background: rgba(0, 0, 0, 0.55);
    border-radius: 3px;
    padding: 2px 6px;
    pointer-events: none;
    text-align: right;
  }
  .decisions .open { color: #6f6; }
  .notice {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    font: 1rem ui-monospace, monospace;
    color: #ccc;
    background: rgba(0, 0, 0, 0.6);
    pointer-events: none;
  }
  .notice.dim { background: rgba(0, 0, 0, 0.25); }
</style>
