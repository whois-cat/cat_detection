// Minimal parsing of the fMP4 pieces streamhub produces (single H.264 track).

function type(u8, off) {
  return String.fromCharCode(u8[off + 4], u8[off + 5], u8[off + 6], u8[off + 7]);
}

function size(u8, off) {
  return new DataView(u8.buffer, u8.byteOffset + off, 4).getUint32(0);
}

// Header bytes to skip inside a container before its child boxes start.
const CHILD_OFFSET = {
  moov: 8, trak: 8, mdia: 8, minf: 8, stbl: 8, moof: 8, traf: 8,
  stsd: 16,  // full box header + entry count
  avc1: 86,  // box header + visual sample entry
};

// findBox returns [offset, size] of the box at the given path, or null.
export function findBox(u8, path, start = 0, end = u8.length) {
  let off = start;
  while (off + 8 <= end) {
    const sz = size(u8, off);
    if (sz < 8 || off + sz > end) return null;
    if (type(u8, off) === path[0]) {
      if (path.length === 1) return [off, sz];
      const skip = CHILD_OFFSET[path[0]];
      if (skip === undefined) return null;
      return findBox(u8, path.slice(1), off + skip, off + sz);
    }
    off += sz;
  }
  return null;
}

// isInit reports whether u8 is an init segment (starts with ftyp).
export function isInit(u8) {
  return u8.length >= 8 && type(u8, 0) === 'ftyp';
}

// codecFromInit returns the RFC 6381 codec string, e.g. "avc1.640032".
export function codecFromInit(u8) {
  const avcC = findBox(u8, ['moov', 'trak', 'mdia', 'minf', 'stbl', 'stsd', 'avc1', 'avcC']);
  if (!avcC) throw new Error('init segment has no avcC');
  const hex = b => b.toString(16).padStart(2, '0');
  const p = avcC[0] + 8;
  return `avc1.${hex(u8[p + 1])}${hex(u8[p + 2])}${hex(u8[p + 3])}`;
}

// fragmentStart returns the first sample's decode time (90 kHz ticks) of a
// fragment, or null if u8 isn't one.
export function fragmentStart(u8) {
  const tfdt = findBox(u8, ['moof', 'traf', 'tfdt']);
  if (!tfdt) return null;
  const dv = new DataView(u8.buffer, u8.byteOffset + tfdt[0] + 8);
  const version = dv.getUint8(0);
  return version === 1 ? Number(dv.getBigUint64(4)) : dv.getUint32(4);
}

export const CLOCK_RATE = 90000;
