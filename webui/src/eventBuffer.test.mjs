import assert from 'node:assert/strict';
import { appendBoundedEvent, mergeBoundedEvents } from './eventBuffer.js';

{
  const events = [];
  for (let i = 0; i < 5; i++) appendBoundedEvent(events, { wall_ms: i }, 3);
  assert.deepEqual(events.map(e => e.wall_ms), [2, 3, 4]);
}

{
  const existing = [{ wall_ms: 10, pts: 1, model: 'a' }];
  const incoming = [
    { wall_ms: 5, pts: 1, model: 'a' },
    { wall_ms: 10, pts: 1, model: 'a' },
    { wall_ms: 15, pts: 1, model: 'a' },
  ];
  const merged = mergeBoundedEvents(existing, incoming, 10);
  assert.deepEqual(merged.map(e => e.wall_ms), [5, 10, 15]);
}

{
  const merged = mergeBoundedEvents(
    [{ wall_ms: 1 }, { wall_ms: 2 }],
    [{ wall_ms: 3 }, { wall_ms: 4 }],
    2,
  );
  assert.deepEqual(merged.map(e => e.wall_ms), [3, 4]);
}

console.log('eventBuffer.test.mjs: all assertions passed');
