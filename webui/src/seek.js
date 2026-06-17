// Pure seek-planning logic, split out of App.svelte so it is unit-testable
// without a DOM. Decides whether a timeline seek can be served by moving the
// existing media element's currentTime (cheap, no network) or needs a fresh
// playback-window load (changes video.src).

export function planSeek({
  targetMs,
  windowStartMs,
  windowEndMs,
  hasWindow,
  marginMs = 500,
}) {
  // Inside the currently-loaded window → seek locally, never touch src.
  if (
    hasWindow &&
    windowStartMs != null &&
    windowEndMs != null &&
    targetMs >= windowStartMs - marginMs &&
    targetMs <= windowEndMs + marginMs
  ) {
    return {
      mode: 'currentTime',
      currentTime: Math.max(0, (targetMs - windowStartMs) / 1000),
    };
  }
  return { mode: 'reload' };
}

// True when the requested window start matches the one already loaded, so we
// can skip re-fetching the identical range.
export function sameWindowLoaded(startMs, loadedStartMs, hasSrc) {
  return !!hasSrc && loadedStartMs != null && startMs === loadedStartMs;
}
