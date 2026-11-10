// The survey guide: a visual journey through our survey, section by section.
// index.html (map, no ?sec) · guide.html?sec=<id> (one chapter).
// Chapter text: guide/<id>/text.md (notes.md syntax + :::viz slots + :::read boxes + [[term]] links);
// visuals: guide/<id>/viz.js (new visuals, crash-course visuals and paper tabs re-exported by name);
// terms: course/glossary.json + guide/glossary.json; order: guide/guide.json.
import { renderNotesArticle, inline } from '../paper/notes.js';
import { Widget } from '../learn/widget.js';
// Two guides: survey v3 (default, guide/) and the earlier survey v2 (guide.html?v=2, guide-v2/).
const V2 = new URLSearchParams(location.search).get('v') === '2';
const BASE = V2 ? 'guide-v2' : 'guide';
const PDF = V2 ? 'public/survey/NonConvex_Survey_v2_Saddles_to_EoS.pdf' : 'public/survey3/Optimization_Methods_Survey_with_Figures.pdf';
const MAP = V2 ? 'guide.html?v=2' : './';
const OTHER = V2 ? { href: './', label: 'Current survey guide' } : { href: 'guide.html?v=2', label: 'Previous survey (v2) guide' };

const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const SEC = new URLSearchParams(location.search).get('sec');
const LOCAL = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
if (LOCAL) {   // test hook: collect errors so scripts can audit every visual
  window.__atlasErrors = [];
  addEventListener('error', e => window.__atlasErrors.push(String(e.message)));
  const ce = console.error.bind(console); console.error = (...a) => { window.__atlasErrors.push(a.map(String).join(' ')); ce(...a); };
}
const katexReady = async () => { for (let i = 0; i < 100 && !window.katex; i++) await new Promise(r => setTimeout(r, 50)); };
const chapUrl = id => `guide.html?${V2 ? 'v=2&' : ''}sec=${id}`;

async function main() {
  const [guide, g1, g2, course] = await Promise.all([
    fetch(`${BASE}/guide.json`).then(r => r.json()),
    fetch('course/glossary.json').then(r => r.json()),
    fetch(`${BASE}/glossary.json`).then(r => r.json()).catch(() => ({})),
    fetch('course/course.json').then(r => r.json()),
  ]);
  const gloss = Object.assign({}, g1, g2);
  const chs = guide.chapters;
  $('#foot').innerHTML = `<span>Survey Visual Atlas · a visual guide to our survey · ${chs.filter(c => c.ready).length} of ${chs.length} chapters</span><span>Every surface is a real function; every ball runs a real optimizer. <a href="https://github.com/farisnasser/survey-visual-atlas/tree/main/scripts/check-math">Math checks</a></span>`;
  const where = termWhere(chs, course.chapters);
  if (!SEC) return renderMap(guide, gloss, where);
  const i = chs.findIndex(c => c.id === SEC);
  if (i < 0 || !chs[i].ready) { $('#main').innerHTML = `<div class="lc-head"><h1>Not written yet</h1><p class="lede">This chapter of the guide is still being written. <a href="./">Back to the map</a>.</p></div>`; return; }
  await renderChapter(guide, i, gloss, where);
}

/** Where a glossary term is built: a guide chapter or a crash-course stop. */
function termWhere(chs, stops) {
  const byG = Object.fromEntries(chs.map(c => [c.id, c])), byC = Object.fromEntries(stops.map(c => [c.id, c]));
  return (g, id) => {
    const gc = byG[g.ch];
    if (gc) return { href: chapUrl(gc.id) + (g.sec ? '#' + g.sec : ''), label: `Built in chapter ${gc.n}: ${gc.title}` };
    const cc = byC[g.ch];
    if (cc) return { href: `learn.html?ch=${cc.id}${g.sec ? '#' + g.sec : ''}`, label: `Built in crash-course stop ${cc.n}: ${cc.title}` };
    return { href: `learn.html?ch=glossary#t-${id}`, label: 'Glossary' };
  };
}

async function renderMap(guide, gloss, where) {
  document.title = 'Survey Visual Atlas · a visual guide to our survey';
  await katexReady();
  const chs = guide.chapters;
  $('#pn').innerHTML = `<a href="${PDF}" target="_blank" rel="noopener">Survey PDF</a><a href="learn.html">Crash course</a><a href="library.html">Paper library</a><a href="${OTHER.href}">${OTHER.label}</a>`;
  const parts = guide.parts.map(p => ({ ...p, chs: chs.filter(c => c.sec === p.sec) })).filter(p => p.chs.length);
  $('#main').innerHTML = `
    <header class="lc-head lc-hero">
      <div class="kicker mono">${esc(guide.kicker)}</div>
      <h1>${esc(guide.title)}</h1>
      <p class="lede">${inline(guide.lede)}</p>
      <div class="row">${chs[0].ready ? `<a class="act" href="${chapUrl(chs[0].id)}">Start the journey →</a>` : ''}<a class="btnlink" href="${PDF}" target="_blank" rel="noopener">Read the survey (PDF)</a><a class="btnlink" href="learn.html">Crash course (basics)</a><a class="btnlink" href="library.html">Paper library</a><a class="btnlink" href="${OTHER.href}">${OTHER.label}</a></div>
    </header>
    <div class="gd-parts">${parts.map(p => `
      <section class="gd-part">
        <div class="gd-part-h"><span class="mono">Section ${esc(p.sec)}</span><h2>${esc(p.title)}</h2>${p.blurb ? `<p>${inline(p.blurb)}</p>` : ''}</div>
        <ol class="journey">${p.chs.map(c => `
          <li class="stop ${c.ready ? 'ready' : 'soon'}">
            <span class="stop-n mono">${c.n}</span>
            <div class="stop-b">
              ${c.ready ? `<a class="stop-t" href="${chapUrl(c.id)}">${esc(c.title)}</a>` : `<span class="stop-t">${esc(c.title)}</span>`}
              <p class="stop-q">${inline(c.question)}</p>
              <div class="stop-terms">${(c.terms || []).map(t => gloss[t] ? `<a class="term chipterm" href="${where(gloss[t], t).href}" data-term="${t}">${inline(gloss[t].term)}</a>` : '').join('')}</div>
              ${c.ready ? '' : '<span class="chip">being written</span>'}
            </div>
          </li>`).join('')}
        </ol>
      </section>`).join('')}</div>`;
  wireTerms(gloss, where);
}

async function renderChapter(guide, i, gloss, where) {
  const chs = guide.chapters, c = chs[i], prev = chs[i - 1], next = chs[i + 1];
  const part = guide.parts.find(p => p.sec === c.sec);
  document.title = `${c.n}. ${c.title} · Survey guide`;
  $('#pn').innerHTML = `<a href="${MAP}">Map</a><a href="${PDF}" target="_blank" rel="noopener"><span class="pn-t">Survey </span>PDF</a>${prev ? `<a href="${chapUrl(prev.id)}" title="${esc(prev.title)}">← ${prev.n}<span class="pn-t">. ${esc(prev.title)}</span></a>` : ''}${next?.ready ? `<a href="${chapUrl(next.id)}" title="${esc(next.title)}">${next.n}<span class="pn-t">. ${esc(next.title)}</span> →</a>` : ''}`;
  const [md, mod] = await Promise.all([
    fetch(`${BASE}/${c.id}/text.md`).then(r => { if (!r.ok) throw new Error(`text.md ${r.status}`); return r.text(); }),
    import(`../../${BASE}/${c.id}/viz.js`).catch(err => { console.error(err); return { default: {} }; }),
  ]);
  const article = await renderNotesArticle(md);
  $('#main').innerHTML = `
    <header class="lc-head">
      <div class="kicker mono">Survey section ${esc(c.sec)}${part ? ` · ${esc(part.title)}` : ''} · chapter ${c.n} of ${chs.length}</div>
      <h1>${esc(c.title)}</h1>
      <p class="lede">${inline(c.question)}</p>
      <div class="lc-progress" aria-hidden="true">${chs.map(x => `<i class="${x.n < c.n ? 'past' : x.n === c.n ? 'now' : ''}"></i>`).join('')}</div>
    </header>
    <section class="notes as-page dense lc-text" id="course">${article}</section>
    ${next ? `<a class="lc-next ${next.ready ? '' : 'soon'}" ${next.ready ? `href="${chapUrl(next.id)}"` : ''}>
      <span class="mono">Next · chapter ${next.n} · survey section ${esc(next.sec)}${next.ready ? '' : ' · being written'}</span><b>${esc(next.title)}</b><span>${inline(next.question)}</span></a>` : ''}`;
  mountVisuals(mod.default || {}, c.id);
  wireTerms(gloss, where);
  if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
}

function mountVisuals(viz, chId) {
  const widgets = {};
  const mount = slot => {
    const name = slot.dataset.viz; io.unobserve(slot);
    if (widgets[name]) return;
    if (!viz[name]) { slot.innerHTML = `<div class="lv-missing">Visual “${esc(name)}” is missing.</div>`; console.error(`[guide] ${chId}: no visual "${name}"`); return; }
    try { widgets[name] = new Widget(slot, viz[name], name); } catch (err) { console.error(`[guide] ${chId}/${name}`, err); }
  };
  const io = new IntersectionObserver(es => { for (const e of es) if (e.isIntersecting) mount(e.target); }, { rootMargin: '500px 0px' });
  document.querySelectorAll('.lv[data-viz]').forEach(s => io.observe(s));
  // test hook: mountAll() for audits in a hidden browser, where intersection callbacks never run
  if (LOCAL) window.__course = { widgets, tick: (name, n) => widgets[name]?.tick(n), mountAll: () => document.querySelectorAll('.lv[data-viz]').forEach(mount) };
}

// ---------- term popovers: hover on a pointer device, first tap on touch ----------
function wireTerms(gloss, where) {
  const pop = $('#termPop');
  let cur = null, hideT = 0;
  const show = a => {
    const id = a.dataset.term, g = gloss[id];
    if (!g) { if (LOCAL) console.error(`[guide] unknown term "${id}"`); return false; }
    const w = where(g, id);
    let texHtml = '';
    if (g.tex && window.katex) texHtml = `<div class="tp-tex">${window.katex.renderToString(g.tex, { displayMode: true, throwOnError: false })}</div>`;
    pop.innerHTML = `<b>${inline(g.term)}</b><p>${inline(g.def)}</p>${texHtml}<a href="${w.href}">${esc(w.label)} →</a>`;
    pop.hidden = false;
    const r = a.getBoundingClientRect(), pw = Math.min(360, innerWidth - 24);
    pop.style.width = pw + 'px';
    const left = Math.min(Math.max(12, r.left + r.width / 2 - pw / 2), innerWidth - pw - 12);
    const below = r.bottom + 8 + pop.offsetHeight < innerHeight || r.top < pop.offsetHeight + 16;
    pop.style.left = left + scrollX + 'px';
    pop.style.top = (below ? r.bottom + 8 : r.top - pop.offsetHeight - 8) + scrollY + 'px';
    cur = a; return true;
  };
  const hide = () => { pop.hidden = true; cur = null; };
  document.querySelectorAll('a.term').forEach(a => {
    // [[term]] links from notes.js point at the crash-course glossary; send them to where the term is built
    const g = gloss[a.dataset.term]; if (g) a.href = where(g, a.dataset.term).href;
    a.addEventListener('pointerenter', e => { if (e.pointerType === 'mouse') { clearTimeout(hideT); show(a); } });
    a.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') hideT = setTimeout(hide, 220); });
    a.addEventListener('click', e => { if (cur !== a && matchMedia('(hover: none)').matches) { e.preventDefault(); show(a); } });
    a.addEventListener('focus', () => show(a));
    a.addEventListener('blur', () => { hideT = setTimeout(hide, 220); });
  });
  pop.addEventListener('pointerenter', () => clearTimeout(hideT));
  pop.addEventListener('pointerleave', () => { hideT = setTimeout(hide, 220); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hide(); });
  document.addEventListener('click', e => { if (cur && !e.target.closest('a.term') && !e.target.closest('#termPop')) hide(); });
}

main().catch(err => { console.error(err); $('#main').innerHTML = `<div class="lc-head"><h1>Something went wrong</h1><p class="lede">${esc(err.message)}</p></div>`; });
