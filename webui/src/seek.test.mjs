// Runnable with: node webui/src/seek.test.mjs   (no deps, no build step)
import assert from 'node:assert/strict';
import { planSeek, sameWindowLoaded } from './seek.js';

const winStart = 1_000_000;
const winEnd = winStart + 15 * 60 * 1000; // 15-min window

// In-window seek → currentTime only, no src reload.
let p = planSeek({ targetMs: winStart + 60_000, windowStartMs: winStart, windowEndMs: winEnd, hasWindow: true });
assert.equal(p.mode, 'currentTime');
assert.equal(p.currentTime, 60);

// Just before window start but within margin → clamped to 0, still local.
p = planSeek({ targetMs: winStart - 200, windowStartMs: winStart, windowEndMs: winEnd, hasWindow: true });
assert.equal(p.mode, 'currentTime');
assert.equal(p.currentTime, 0);

// Far outside the window → reload (src change).
p = planSeek({ targetMs: winEnd + 10 * 60_000, windowStartMs: winStart, windowEndMs: winEnd, hasWindow: true });
assert.equal(p.mode, 'reload');

// No window yet → reload.
p = planSeek({ targetMs: winStart, windowStartMs: null, windowEndMs: null, hasWindow: false });
assert.equal(p.mode, 'reload');

// Same range already loaded → skip refetch.
assert.equal(sameWindowLoaded(winStart, winStart, true), true);
assert.equal(sameWindowLoaded(winStart, winStart + 1, true), false);
assert.equal(sameWindowLoaded(winStart, winStart, false), false);

console.log('seek.test.mjs: all assertions passed');
