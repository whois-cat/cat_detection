export function normalizeEpochMs(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) throw new Error(`invalid epoch ms: ${value}`);
  return Math.max(0, Math.round(n));
}

export function rangeCacheKey(cameraId, fromMs, toMs) {
  return `${cameraId}:${normalizeEpochMs(fromMs)}:${normalizeEpochMs(toMs)}`;
}

export function normalizeAvailabilityRanges(ranges) {
  if (!Array.isArray(ranges)) return [];
  return ranges.map((range) => {
    if (Array.isArray(range)) {
      return {
        from_ms: normalizeEpochMs(range[0]),
        to_ms: normalizeEpochMs(range[1]),
      };
    }
    return {
      from_ms: normalizeEpochMs(range?.from_ms),
      to_ms: normalizeEpochMs(range?.to_ms),
    };
  }).filter((range) => range.to_ms > range.from_ms);
}

export async function fetchJsonWithTimeout(url, {
  fetchImpl = globalThis.fetch,
  timeoutMs = 1500,
  headers = { Accept: 'application/json' },
  signal = null,
} = {}) {
  if (typeof fetchImpl !== 'function') throw new Error('fetch is unavailable');
  const controller = new AbortController();
  const abort = () => controller.abort(signal?.reason);
  if (signal?.aborted) abort();
  else signal?.addEventListener?.('abort', abort, { once: true });
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
    signal?.removeEventListener?.('abort', abort);
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
      return inflight.get(key).promise;
    }

    const started = now();
    const controller = new AbortController();
    const externalSignal = opts.signal;
    const abortFromExternal = () => controller.abort(externalSignal?.reason);
    if (externalSignal?.aborted) abortFromExternal();
    else externalSignal?.addEventListener?.('abort', abortFromExternal, { once: true });
    let entry;
    const promise = fetchJsonWithTimeout(url, {
      fetchImpl,
      timeoutMs: opts.timeoutMs ?? timeoutMs,
      headers: opts.headers,
      signal: controller.signal,
    }).then((value) => {
      cache.set(key, { value, expiresAt: now() + (opts.ttlMs ?? ttlMs) });
      onTiming({ key, url, source: 'network', elapsedMs: now() - started });
      return value;
    }).finally(() => {
      externalSignal?.removeEventListener?.('abort', abortFromExternal);
      if (inflight.get(key) === entry) inflight.delete(key);
    });
    entry = { promise, controller };
    inflight.set(key, entry);
    return promise;
  }

  function clear(prefix = '') {
    for (const key of cache.keys()) {
      if (!prefix || key.startsWith(prefix)) cache.delete(key);
    }
    for (const [key, entry] of inflight.entries()) {
      if (!prefix || key.startsWith(prefix)) {
        entry.controller.abort();
        inflight.delete(key);
      }
    }
  }

  function abort(prefix = '') {
    for (const [key, entry] of inflight.entries()) {
      if (!prefix || key.startsWith(prefix)) {
        entry.controller.abort();
        inflight.delete(key);
      }
    }
  }

  return { get, clear, abort, _cache: cache, _inflight: inflight };
}
