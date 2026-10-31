// Crash course: learn.html (journey map) · learn.html?ch=<id> (one stop) · learn.html?ch=glossary (every term).
// Chapter text: course/<id>/text.md (notes.md syntax + :::viz slots + [[term]] links); visuals: course/<id>/viz.js;
// terms: course/glossary.json; order and status: course/course.json.
import { renderNotesArticle, inline } from '../paper/notes.js';
import { Widget } from './widget.js';

const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const CH = new URLSearchParams(location.search).get('ch');
const LOCAL = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
if (LOCAL) {   // test hook: collect errors so scripts can audit every visual
  window.__atlasErrors = [];
  addEventListener('error', e => window.__atlasErrors.push(String(e.message)));
  const ce = console.error.bind(console); console.error = (...a) => { window.__atlasErrors.push(a.map(String).join(' ')); ce(...a); };
}
const katexReady = async () => { for (let i = 0; i < 100 && !window.katex; i++) await new Promise(r => setTimeout(r, 50)); };

async function main() {
  const [course, gloss] = await Promise.all([fetch('course/course.json').then(r => r.json()), fetch('course/glossary.json').then(r => r.json())]);
  const chs = course.chapters;
  $('#foot').innerHTML = `<span>Survey Visual Atlas · crash course · ${chs.filter(c => c.ready).length} of ${chs.length} stops written</span><span>Every surface is a real function; every ball runs a real optimizer. <a href="https://github.com/farisnasser/survey-visual-atlas/tree/main/scripts/check-math">Math checks</a></span>`;
  if (!CH) return renderMap(course, gloss);
  if (CH === 'glossary') return renderGlossary(course, gloss);
  const i = chs.findIndex(c => c.id === CH);
  if (i < 0 || !chs[i].ready) { $('#main').innerHTML = `<div class="lc-head"><h1>Not written yet</h1><p class="lede">This stop of the crash course is still being written. <a href="learn.html">Back to the map</a>.</p></div>`; return; }
  await renderChapter(chs, i, gloss);
}

async function renderMap(course, gloss) {
  document.title = 'Crash course · Survey Visual Atlas';
  await katexReady();
  const chs = course.chapters;
  $('#pn').innerHTML = `<a href="learn.html?ch=glossary">Glossary · ${Object.keys(gloss).length} terms</a>`;
  $('#main').innerHTML = `
    <header class="lc-head lc-hero">
      <div class="kicker mono">Crash course · ${chs.length} stops</div>
      <h1>${esc(course.title)}</h1>
      <p class="lede">${inline(course.lede)}</p>
      <div class="row">${chs[0].ready ? `<a class="act" href="learn.html?ch=${chs[0].id}">Start the journey →</a>` : ''}<a class="btnlink" href="learn.html?ch=glossary">Glossary</a><a class="btnlink" href="./">The papers</a></div>
    </header>
    <ol class="journey">${chs.map(c => `
      <li class="stop ${c.ready ? 'ready' : 'soon'}">
        <span class="stop-n mono">${c.n}</span>
        <div class="stop-b">
          ${c.ready ? `<a class="stop-t" href="learn.html?ch=${c.id}">${esc(c.title)}</a>` : `<span class="stop-t">${esc(c.title)}</span>`}
          <p class="stop-q">${inline(c.question)}</p>
          <div class="stop-terms">${(c.terms || []).map(t => gloss[t] ? `<a class="term chipterm" href="learn.html?ch=glossary#t-${t}" data-term="${t}">${inline(gloss[t].term)}</a>` : '').join('')}</div>
          ${c.ready ? '' : '<span class="chip">being written</span>'}
        </div>
      </li>`).join('')}
    </ol>`;
  wireTerms(gloss, course);
}

async function renderGlossary(course, gloss) {
  document.title = 'Glossary · Crash course · Survey Visual Atlas';
  await katexReady();
  const byCh = Object.fromEntries(course.chapters.map(c => [c.id, c]));
  const ids = Object.keys(gloss).sort((a, b) => gloss[a].term.replace(/\$|\\/g, '').localeCompare(gloss[b].term.replace(/\$|\\/g, ''), 'en', { sensitivity: 'base' }));
  $('#pn').innerHTML = `<a href="learn.html">← The map</a>`;
  $('#main').innerHTML = `
    <header class="lc-head"><div class="kicker mono">Crash course · glossary</div><h1>Every term, in plain words</h1>
      <p class="lede">Each term links to the stop of the journey where it is built from scratch. In the chapters, hover or tap a dotted term to see its entry.</p>
      <label class="search gl-search"><span class="sr-only">Filter terms</span><input id="gq" type="search" placeholder="Filter terms…" autocomplete="off"></label></header>
    <dl class="gloss">${ids.map(id => { const g = gloss[id], c = byCh[g.ch]; return `
      <div class="g-row" id="t-${id}" data-k="${esc((g.term + ' ' + (g.aka || []).join(' ') + ' ' + g.def).toLowerCase())}">
        <dt>${inline(g.term)}${g.aka?.length ? `<span class="aka">${g.aka.map(inline).join(' · ')}</span>` : ''}</dt>
        <dd><p>${inline(g.def)}</p>${g.tex ? `<div class="g-tex">${window.katex.renderToString(g.tex, { displayMode: true, throwOnError: false })}</div>` : ''}
          ${c ? `<a class="g-ch" href="learn.html?ch=${c.id}${g.sec ? '#' + g.sec : ''}">Stop ${c.n} · ${esc(c.title)}${c.ready ? '' : ' (being written)'} →</a>` : ''}</dd>
      </div>`; }).join('')}</dl>`;
  $('#gq').addEventListener('input', e => { const q = e.target.value.trim().toLowerCase(); document.querySelectorAll('.g-row').forEach(r => { r.hidden = q && !r.dataset.k.includes(q); }); });
  if (location.hash) { const el = document.getElementById(location.hash.slice(1)); if (el) { el.classList.add('hl'); el.scrollIntoView({ block: 'center' }); } }
}

async function renderChapter(chs, i, gloss) {
  const c = chs[i], prev = chs[i - 1], next = chs[i + 1];
  document.title = `${c.n}. ${c.title} · Crash course`;
  $('#pn').innerHTML = `<a href="learn.html">Map</a>${prev ? `<a href="learn.html?ch=${prev.id}" title="${esc(prev.title)}">← ${prev.n}. ${esc(prev.title)}</a>` : ''}${next?.ready ? `<a href="learn.html?ch=${next.id}" title="${esc(next.title)}">${next.n}. ${esc(next.title)} →</a>` : ''}`;
  const [md, mod] = await Promise.all([
    fetch(`course/${c.id}/text.md`).then(r => { if (!r.ok) throw new Error(`text.md ${r.status}`); return r.text(); }),
    import(`../../course/${c.id}/viz.js`).catch(err => { console.error(err); return { default: {} }; }),
  ]);
  const article = await renderNotesArticle(md);
  $('#main').innerHTML = `
    <header class="lc-head">
      <div class="kicker mono">Crash course · stop ${c.n} of ${chs.length}</div>
      <h1>${esc(c.title)}</h1>
      <p class="lede">${inline(c.question)}</p>
      <div class="lc-progress" aria-hidden="true">${chs.map(x => `<i class="${x.n < c.n ? 'past' : x.n === c.n ? 'now' : ''}"></i>`).join('')}</div>
    </header>
    <section class="notes as-page dense lc-text" id="course">${article}</section>
    ${next ? `<a class="lc-next ${next.ready ? '' : 'soon'}" ${next.ready ? `href="learn.html?ch=${next.id}"` : ''}>
      <span class="mono">Next stop · ${next.n}${next.ready ? '' : ' · being written'}</span><b>${esc(next.title)}</b><span>${inline(next.question)}</span></a>` : ''}`;
  mountVisuals(mod.default || {}, c.id);
  wireTerms(gloss, { chapters: chs });
  if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
}

function mountVisuals(viz, chId) {
  const widgets = {};
  const mount = slot => {
    const name = slot.dataset.viz; io.unobserve(slot);
    if (widgets[name]) return;
    if (!viz[name]) { slot.innerHTML = `<div class="lv-missing">Visual “${esc(name)}” is missing.</div>`; console.error(`[course] ${chId}: no visual "${name}"`); return; }
    try { widgets[name] = new Widget(slot, viz[name], name); } catch (err) { console.error(`[course] ${chId}/${name}`, err); }
  };
  const io = new IntersectionObserver(es => { for (const e of es) if (e.isIntersecting) mount(e.target); }, { rootMargin: '500px 0px' });
  document.querySelectorAll('.lv[data-viz]').forEach(s => io.observe(s));
  // test hook: mountAll() for audits in a hidden browser, where intersection callbacks never run
  if (LOCAL) window.__course = { widgets, tick: (name, n) => widgets[name]?.tick(n), mountAll: () => document.querySelectorAll('.lv[data-viz]').forEach(mount) };
}

// ---------- term popovers: hover on a pointer device, first tap on touch ----------
function wireTerms(gloss, course) {
  const pop = $('#termPop'), byCh = Object.fromEntries((course.chapters || []).map(c => [c.id, c]));
  let cur = null, hideT = 0;
  const show = a => {
    const id = a.dataset.term, g = gloss[id];
    if (!g) { if (LOCAL) console.error(`[course] unknown term "${id}"`); return false; }
    const c = byCh[g.ch];
    let texHtml = '';
    if (g.tex && window.katex) texHtml = `<div class="tp-tex">${window.katex.renderToString(g.tex, { displayMode: true, throwOnError: false })}</div>`;
    pop.innerHTML = `<b>${inline(g.term)}</b><p>${inline(g.def)}</p>${texHtml}<a href="${c ? `learn.html?ch=${c.id}${g.sec ? '#' + g.sec : ''}` : `learn.html?ch=glossary#t-${id}`}">${c ? `Built in stop ${c.n}: ${esc(c.title)}` : 'Glossary'} →</a>`;
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
