import assert from 'node:assert/strict';
import {
  createJsonRequestCache,
  normalizeAvailabilityRanges,
  normalizeEpochMs,
  rangeCacheKey,
} from './requestCache.js';

function jsonResponse(body, { status = 200 } = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: () => 'application/json' },
    async json() { return body; },
    async text() { return JSON.stringify(body); },
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

assert.equal(
  rangeCacheKey('grey', 1000.2, 2000.8),
  'grey:1000:2001',
);
assert.equal(normalizeEpochMs(123.49), 123);
assert.equal(normalizeEpochMs(123.5), 124);
assert.deepEqual(
  normalizeAvailabilityRanges([
    [1000.2, 2000.8],
    { from_ms: 3000.1, to_ms: 4000.6, path: '/too/much/data.mp4' },
    [5000, 5000],
  ]),
  [
    { from_ms: 1000, to_ms: 2001 },
    { from_ms: 3000, to_ms: 4001 },
  ],
);

{
  let calls = 0;
  let now = 0;
  const cache = createJsonRequestCache({
    ttlMs: 100,
    now: () => now,
    fetchImpl: async () => {
      calls++;
      return jsonResponse({ ok: true, calls });
    },
  });

  assert.deepEqual(await cache.get('grey:1', '/ranges'), { ok: true, calls: 1 });
  assert.deepEqual(await cache.get('grey:1', '/ranges'), { ok: true, calls: 1 });
  assert.equal(calls, 1);

  now = 101;
  assert.deepEqual(await cache.get('grey:1', '/ranges'), { ok: true, calls: 2 });
  assert.equal(calls, 2);
}

{
  let calls = 0;
  const d = deferred();
  const cache = createJsonRequestCache({
    fetchImpl: async () => {
      calls++;
      return d.promise;
    },
  });
  const a = cache.get('same', '/slow');
  const b = cache.get('same', '/slow');
  assert.equal(calls, 1);
  d.resolve(jsonResponse({ done: true }));
  assert.deepEqual(await a, { done: true });
  assert.deepEqual(await b, { done: true });
}

{
  let calls = 0;
  const cache = createJsonRequestCache({
    fetchImpl: async () => {
      calls++;
      return jsonResponse({ calls });
    },
  });
  await cache.get('cam-a:1', '/ranges?camera=a');
  await cache.get('cam-b:1', '/ranges?camera=b');
  assert.equal(calls, 2);
}

{
  let calls = 0;
  const cache = createJsonRequestCache({
    fetchImpl: async () => {
      calls++;
      return jsonResponse({ error: true }, { status: 502 });
    },
  });
  await assert.rejects(() => cache.get('models:grey', '/models'));
  await assert.rejects(() => cache.get('models:grey', '/models'));
  assert.equal(calls, 2);
}

{
  let aborted = false;
  const cache = createJsonRequestCache({
    fetchImpl: async (_url, opts) => {
      return new Promise((_resolve, reject) => {
        opts.signal.addEventListener('abort', () => {
          aborted = true;
          reject(Object.assign(new Error('aborted'), { name: 'AbortError' }));
        }, { once: true });
      });
    },
  });
  const p = cache.get('ranges:grey:1', '/slow-ranges');
  cache.abort('ranges:grey:');
  await assert.rejects(() => p);
  assert.equal(aborted, true);
  assert.equal(cache._inflight.size, 0);
}

console.log('requestCache.test.mjs: all assertions passed');
