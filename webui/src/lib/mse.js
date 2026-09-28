// MseSink feeds streamhub's fMP4 pieces into a <video> through Media Source
// Extensions. Both live (WebSocket) and history (segment files) use it.
//
// Media timestamps are wall-clock PTS (~1.8e9 s); the sink rebases them with
// SourceBuffer.timestampOffset so video.currentTime stays small:
// wall seconds = base + currentTime.

import { isInit, codecFromInit, fragmentStart, findBox, CLOCK_RATE } from './mp4.js';

const MediaSourceImpl = globalThis.ManagedMediaSource || globalThis.MediaSource;

export class MseSink {
  constructor(video) {
    this.video = video;
    this.ms = new MediaSourceImpl();
    this.sb = null;
    this.codec = null;
    this.base = null; // seconds
    this.ops = Promise.resolve();
    this.destroyed = false;
    video.disableRemotePlayback = true; // required by ManagedMediaSource
    this.url = URL.createObjectURL(this.ms);
    video.src = this.url;
    this.opened = new Promise(resolve => this.ms.addEventListener('sourceopen', resolve, { once: true }));
  }

  // append queues fMP4 bytes: an init segment, a fragment, or a whole
  // segment file (init + fragments), which is split so the base can be set.
  append(u8) {
    if (isInit(u8)) {
      const moov = findBox(u8, ['moov']);
      const initEnd = moov[0] + moov[1];
      const p = this.enqueue(() => this.appendInit(u8.subarray(0, initEnd)));
      return initEnd < u8.length ? this.append(u8.subarray(initEnd)) : p;
    }
    return this.enqueue(() => this.appendMedia(u8));
  }

  // remove drops buffered media in [fromMs, toMs) wall-clock time.
  remove(fromMs, toMs) {
    return this.enqueue(async () => {
      if (!this.sb || this.base === null) return;
      const from = Math.max(0, fromMs / 1000 - this.base);
      const to = toMs / 1000 - this.base;
      if (to > from) await this.run(() => this.sb.remove(from, to));
    });
  }

  // wallMs is the wall-clock time of the displayed frame, or null.
  wallMs() {
    return this.base === null ? null : (this.base + this.video.currentTime) * 1000;
  }

  // mediaTime converts wall-clock ms to video.currentTime.
  mediaTime(wallMs) {
    return wallMs / 1000 - this.base;
  }

  // buffered returns buffered spans as [startMs, endMs] wall-clock pairs.
  buffered() {
    const b = this.video.buffered;
    const out = [];
    if (this.base === null) return out;
    for (let i = 0; i < b.length; i++) out.push([(this.base + b.start(i)) * 1000, (this.base + b.end(i)) * 1000]);
    return out;
  }

  destroy() {
    this.destroyed = true;
    this.video.removeAttribute('src');
    this.video.load();
    URL.revokeObjectURL(this.url);
  }

  enqueue(fn) {
    const p = this.ops.then(() => (this.destroyed ? undefined : fn()));
    this.ops = p.catch(() => {}); // keep the queue going after a failure
    return p;
  }

  async appendInit(u8) {
    await this.opened;
    const codec = codecFromInit(u8);
    const type = `video/mp4; codecs="${codec}"`;
    if (!this.sb) {
      this.sb = this.ms.addSourceBuffer(type);
      this.sb.mode = 'segments';
    } else if (codec !== this.codec) {
      this.sb.changeType(type);
    }
    this.codec = codec;
    await this.run(() => this.sb.appendBuffer(u8));
  }

  async appendMedia(u8) {
    if (!this.sb) throw new Error('media before init segment');
    if (this.base === null) {
      this.base = Math.floor(fragmentStart(u8) / CLOCK_RATE);
      this.sb.timestampOffset = -this.base;
    }
    try {
      await this.run(() => this.sb.appendBuffer(u8));
    } catch (e) {
      if (e.name !== 'QuotaExceededError') throw e;
      // Buffer full: drop everything well behind the playhead and retry once.
      const keepFrom = this.video.currentTime - 10;
      if (keepFrom > 0) await this.run(() => this.sb.remove(0, keepFrom));
      await this.run(() => this.sb.appendBuffer(u8));
    }
  }

  // run performs one SourceBuffer operation and waits for it to finish.
  run(op) {
    return new Promise((resolve, reject) => {
      const done = () => { cleanup(); resolve(); };
      const fail = () => { cleanup(); reject(new Error('SourceBuffer error')); };
      const cleanup = () => {
        this.sb.removeEventListener('updateend', done);
        this.sb.removeEventListener('error', fail);
      };
      this.sb.addEventListener('updateend', done);
      this.sb.addEventListener('error', fail);
      try { op(); } catch (e) { cleanup(); reject(e); }
    });
  }
}
