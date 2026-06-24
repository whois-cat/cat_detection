export function rangeCacheKey(cameraId, fromMs, toMs) {
  return `${cameraId}:${Math.round(fromMs)}:${Math.round(toMs)}`;
}

export async function fetchJsonWithTimeout(url, {
  fetchImpl = globalThis.fetch,
  timeoutMs = 1500,
  headers = { Accept: 'application/json' },
} = {}) {
  if (typeof fetchImpl !== 'function') throw new Error('fetch is unavailable');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const r = await fetchImpl(url, { headers, signal: controller.signal });
    const contentType = r.headers?.get?.('content-type') || '';
    if (!r.ok) {
      let body = '';
      try { body = await r.text(); } catch {}
      throw new Error(`HTTP ${r.status} ${contentType || 'without content-type'} ${body.slice(0, 80)}`);
    }
    if (contentType && !contentType.includes('application/json')) {
      let body = '';
      try { body = await r.text(); } catch {}
      throw new Error(`expected JSON, got ${contentType}; first bytes: ${body.slice(0, 80)}`);
    }
    return await r.json();
  } finally {
    clearTimeout(timer);
  }
}

export function createJsonRequestCache({
  ttlMs = 10_000,
  timeoutMs = 1500,
  now = () => Date.now(),
  fetchImpl = globalThis.fetch,
  onTiming = () => {},
} = {}) {
  const cache = new Map();
  const inflight = new Map();

  async function get(key, url, opts = {}) {
    const t = now();
    const hit = cache.get(key);
    if (hit && hit.expiresAt > t) {
      onTiming({ key, url, source: 'cache', elapsedMs: 0 });
      return hit.value;
    }
    if (inflight.has(key)) {
      onTiming({ key, url, source: 'inflight', elapsedMs: 0 });
      return inflight.get(key);
    }

    const started = now();
    const promise = fetchJsonWithTimeout(url, {
      fetchImpl,
      timeoutMs: opts.timeoutMs ?? timeoutMs,
      headers: opts.headers,
    }).then((value) => {
      cache.set(key, { value, expiresAt: now() + (opts.ttlMs ?? ttlMs) });
      onTiming({ key, url, source: 'network', elapsedMs: now() - started });
      return value;
    }).finally(() => {
      inflight.delete(key);
    });
    inflight.set(key, promise);
    return promise;
  }

  function clear(prefix = '') {
    for (const key of cache.keys()) {
      if (!prefix || key.startsWith(prefix)) cache.delete(key);
    }
    for (const key of inflight.keys()) {
      if (!prefix || key.startsWith(prefix)) inflight.delete(key);
    }
  }

  return { get, clear, _cache: cache, _inflight: inflight };
}
