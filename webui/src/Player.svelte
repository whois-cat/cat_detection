<script>
  // One camera's video: live (WebSocket) or history (segments), following
  // the shared playback state.
  import { onMount, onDestroy, untrack } from 'svelte';
  import { LiveSource, HistorySource } from './lib/sources.js';
  import { play, cams } from './lib/state.svelte.js';
  import { fmtDateTime } from './lib/time.js';

  let { camera, onselect } = $props();

  // History players re-sync to the playhead this often.
  const UPDATE_MS = 200;

  let video;
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
      source = want === 'live' ? new LiveSource(video, camera) : new HistorySource(video, camera);
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
  }

  onMount(() => { timer = setInterval(update, UPDATE_MS); });
  onDestroy(() => {
    clearInterval(timer);
    source?.destroy();
    delete play.waiting[camera];
  });
</script>

<div class="player">
  <!-- svelte-ignore a11y_media_has_caption -->
  <video bind:this={video} muted playsinline></video>
  <button class="label" onclick={onselect} title="Show only this camera">{camera}</button>
  {#if shownMs}<div class="time">{fmtDateTime(shownMs)}</div>{/if}
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
  video {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
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
