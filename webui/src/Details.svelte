<script>
  // Everything behind the shown frame: the CV result it was labeled with
  // (per box: detector score and the identity classifier's probability per
  // cat) and each feeder's decider state.
  import { topCat } from './lib/labels.js';
  import { catColor } from './lib/colors.js';
  import { fmtTimeOfDay } from './lib/time.js';

  // result: LabelStore item or null; age: shown frame − result frame (ms);
  // decisions: latest decider record per feeder.
  let { result, age, decisions } = $props();

  const fmtMs = ms => `${fmtTimeOfDay(ms, true)}.${String(Math.floor(ms % 1000)).padStart(3, '0')}`;
  const pct = p => `${(p * 100).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1)}%`;
  const sortedCats = d => Object.entries(d.cats || {}).sort((a, b) => b[1] - a[1]);
</script>

<div class="details">
  <section>
    <h4>CV{#if result} · {result.model}{/if}</h4>
    {#if !result}
      <div class="dim">no result for this frame</div>
    {:else}
      <div class="dim">
        frame {fmtMs(result.ms)} · {result.infer_ms != null ? `${Math.round(result.infer_ms)} ms` : ''}
        {#if age > 0} · {Math.round(age)} ms older than shown frame{/if}
        {#if result.worker} · {result.worker}{/if}
      </div>
      {#if !result.dets.length}
        <div class="dim">no detections</div>
      {/if}
      {#each result.dets as d, i}
        <div class="det">
          <div>
            <b style="color: {catColor(topCat(d))}">#{i + 1}</b>
            yolo cat score <b>{d.score.toFixed(2)}</b>
            <span class="dim">· box {d.box.map(v => v.toFixed(2)).join(' ')}</span>
          </div>
          {#if d.cats}
            {#each sortedCats(d) as [cat, p] (cat)}
              <div class="cat">
                <span class="name">{cat}</span>
                <span class="bar"><span style="width: {p * 100}%; background: {catColor(cat)}"></span></span>
                <span class="p">{pct(p)}</span>
              </div>
            {/each}
          {:else}
            <div class="dim">no identity classifier</div>
          {/if}
        </div>
      {/each}
    {/if}
  </section>

  <section>
    <h4>decider</h4>
    {#if !decisions.length}
      <div class="dim">no feeder on this camera, or no decision yet</div>
    {/if}
    {#each decisions as d (d.feeder)}
      <div class="decision" class:open={d.state === 'open'}>
        <div><b>{d.feeder}</b> · {d.state} · door {d.door}{#if d.display} · display <b>{d.display}</b>{/if}</div>
        <div class="dim">
          {d.present ? 'cat present' : 'no cat'}{#if d.n_cats > 1} · {d.n_cats} cats{/if}
          {#if d.identity} · {d.identity}{#if d.conf != null} {pct(d.conf)}{/if}{/if}
        </div>
        <div class="dim">{d.action === 'open' ? 'wants open' : `stays closed: ${d.reason}`}{#if d.event} · {d.event}{/if}</div>
      </div>
    {/each}
  </section>
</div>

<style>
  .details {
    font: 0.75rem/1.35 ui-monospace, monospace;
    color: #ddd;
    background: #181818;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 6px 8px;
    overflow-y: auto;
    min-width: 0;
  }
  section + section { margin-top: 8px; border-top: 1px solid #333; padding-top: 6px; }
  h4 { margin: 0 0 3px; font-size: 0.8rem; color: #cfe2ff; overflow-wrap: anywhere; }
  .dim { color: #888; }
  .det { margin-top: 5px; }
  .cat { display: grid; grid-template-columns: 5.5em 1fr 3.5em; gap: 6px; align-items: center; padding-left: 1.5em; }
  .name { overflow: hidden; text-overflow: ellipsis; }
  .bar { height: 7px; background: #2a2a2a; border-radius: 2px; overflow: hidden; }
  .bar span { display: block; height: 100%; }
  .p { text-align: right; }
  .decision { margin-top: 4px; }
  .decision.open b:first-child { color: #6f6; }
</style>
