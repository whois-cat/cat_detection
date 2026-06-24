export function eventKey(e) {
  return `${e.wall_ms}:${e.pts ?? ''}:${e.model ?? ''}:${e.track_id ?? ''}`;
}

export function appendBoundedEvent(events, event, maxEvents) {
  events.push(event);
  if (events.length > maxEvents) {
    events.splice(0, events.length - maxEvents);
  }
  return events;
}

export function mergeBoundedEvents(existing, incoming, maxEvents) {
  const seen = new Set(existing.map(eventKey));
  const merged = existing.slice();
  for (const event of incoming) {
    const key = eventKey(event);
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(event);
  }
  merged.sort((a, b) => (a.wall_ms ?? 0) - (b.wall_ms ?? 0));
  return merged.length > maxEvents ? merged.slice(merged.length - maxEvents) : merged;
}
