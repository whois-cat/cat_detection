// Which cameras are shown. null = all (so newly added cameras show too),
// otherwise a list of ids in configuration order.

// select returns the new selection after clicking camera `id`. A plain click
// shows only that camera (or all again if it already was the only one);
// toggle (ctrl/cmd+click) adds or removes it, keeping at least one.
export function select(selected, allIds, id, toggle) {
  const cur = selected ?? allIds;
  if (!toggle) return cur.length === 1 && cur[0] === id ? null : [id];
  const next = cur.includes(id) ? cur.filter(x => x !== id) : [...cur, id];
  if (!next.length) return selected;
  return next.length === allIds.length ? null : allIds.filter(x => next.includes(x));
}

// encode/decode for the URL hash: "all" or "grey,black".
export const encode = selected => (selected ? selected.join(',') : 'all');

export function decode(s, allIds) {
  if (!s || s === 'all') return null;
  const ids = allIds.filter(id => s.split(',').includes(id));
  return ids.length && ids.length < allIds.length ? ids : null;
}
