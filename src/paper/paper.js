// Paper page shell. Renders the header from data/papers.json, then the paper's visual (papers/<key>/visual.js),
// its math panel (papers/<key>/math.md) and the study notes (papers/<key>/notes.md in the page, else the PDF public/notes/<key>.pdf).
import { Stage } from '../lib/atlas3d.js';
import { Plot } from '../lib/charts.js';
import { checkGrad2 } from '../lib/math.js';
import { renderNotesArticle } from './notes.js';

const $ = s => document.querySelector(s);
const qs = new URLSearchParams(location.search);
const KEY = qs.get('id');
const EMBED = qs.has('embed') || qs.has('thumb');
const THUMB = qs.has('thumb');
const LOCAL = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
if (LOCAL) {   // test hook: collect errors so scripts can audit every tab
  window.__atlasErrors = [];
  addEventListener('error', e => window.__atlasErrors.push(String(e.message)));
  const ce = console.error.bind(console); console.error = (...a) => { window.__atlasErrors.push(a.map(String).join(' ')); ce(...a); };
}
if (EMBED) document.body.classList.add('embed');
if (THUMB) document.body.classList.add('thumb');

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export const statusOf = p => p.visual && p.notes ? ['complete', 'Complete'] : p.visual ? ['visual', 'Visual · no notes yet'] : p.notes ? ['notes', 'Notes only'] : ['queued', 'Not started'];
const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };

async function main() {
  const db = await fetch('data/papers.json').then(r => r.json());
  const P = db.papers.find(p => p.key === KEY);
  if (!P) { $('#ph').innerHTML = `<h1>Paper not found</h1><p class="meta">No paper with key “${esc(KEY)}”. <a href="./">Back to the Atlas</a></p>`; $('#viz').hidden = true; return; }
  document.title = `${P.short || P.title} · Survey Visual Atlas`;
  let meta = {};
  if (P.visual) meta = await fetch(`papers/${KEY}/meta.json`).then(r => r.ok ? r.json() : {}).catch(() => ({}));
  renderHeader(P, meta, db);
  if (!EMBED) { renderNotes(P); renderTracker(P); renderNav(P, db); }
  if (!P.visual) { showEmpty(P); return; }
  const [mod, mathText, data] = await Promise.all([
    import(`../../papers/${KEY}/visual.js`),
    fetch(`papers/${KEY}/math.md`).then(r => r.ok ? r.text() : '').catch(() => ''),
    meta.data ? fetch(`papers/${KEY}/data.json`).then(r => r.json()) : Promise.resolve(null),
  ]);
  const visual = mod.default;
  new Shell(visual, parseMath(mathText), data, P, meta);
}

const ROMAN = { I: 1, II: 2, III: 3, IV: 4, V: 5, VI: 6, VII: 7, VIII: 8, IX: 9, X: 10 };
const secShort = s => s.replace(/^([IVX]+)\. (.*)$/, (_, r, t) => `${r}. ${t.charAt(0) + t.slice(1).toLowerCase()}`);

function renderHeader(P, meta, db) {
  const [st, stLabel] = statusOf(P);
  const tags = (meta.tags || []).map(t => `<a class="tag" href="./?concept=${encodeURIComponent(t)}">${esc(t)}</a>`).join('');
  const br = (P.branches || []).map(([id, leaf]) => `<a class="chip" href="./#b-${id}">${id} · ${esc(leaf)}</a>`).join('');
  const cited = (P.cited || []).map(s => s.split('.')[0]).filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => (ROMAN[a] || 0) - (ROMAN[b] || 0));
  $('#ph').innerHTML = `
    <div class="row"><span class="chip st-${st}"><i></i>${stLabel}</span><span class="chip mono">survey [${P.n}]</span>${br || '<span class="chip">Background</span>'}</div>
    <h1>${esc(P.title)}</h1>
    <div class="meta">${esc(P.authors)} · ${esc(P.venue)}${P.year ? ' ' + P.year : ''}${cited.length ? ` · <span title="${esc((P.cited || []).map(secShort).join(' · '))}">cited in Sections ${cited.join(', ')}</span>` : ''}</div>
    ${meta.result ? `<p class="result">${meta.result}</p>` : ''}
    <div class="row">
      ${P.link ? `<a class="btnlink" href="${esc(P.link)}" target="_blank" rel="noopener">Paper page ↗</a>` : ''}
      ${P.pdf ? `<a class="btnlink" href="${esc(P.pdf)}" target="_blank" rel="noopener">Paper PDF ↗</a>` : ''}
      ${!P.link && !P.pdf && P.scholar ? `<a class="btnlink" href="${esc(P.scholar)}" target="_blank" rel="noopener">Find the paper ↗</a>` : ''}
      ${P.notes ? `<a class="btnlink" href="#notes">Study notes</a>` : ''}
      ${tags}
    </div>`;
}

function showEmpty(P) {
  const e = $('#emptyViz'); e.hidden = false;
  e.innerHTML = `<div><b>${P.notes ? 'Visual not built yet' : 'Not started'}</b>${P.notes ? 'The notes are below. The interactive visual comes next.' : 'This reference is in the survey but has no notes or visual yet.'}</div>`;
  ['#tabs', '#mathBtn', '#hud'].forEach(s => $(s).hidden = true);
}

function renderTracker(P) {
  if (!P.summary && !P.assumptions) return;
  $('#tracker').innerHTML = `<details><summary>Tracker summary <span>from the old Papers Master sheet · AI-drafted, not verified</span></summary>
    <div class="txt">${P.summary ? `<h3>Summary</h3><div>${esc(P.summary)}</div>` : ''}${P.assumptions ? `<h3>Assumptions &amp; limits</h3><div>${esc(P.assumptions)}</div>` : ''}</div></details>`;
}

function renderNav(P, db) {
  const list = db.papers.filter(p => p.visual || p.notes).sort((a, b) => a.n - b.n);
  const i = list.findIndex(p => p.key === P.key);
  const prev = i > 0 ? list[i - 1] : null, next = i >= 0 && i < list.length - 1 ? list[i + 1] : null;
  $('#pn').innerHTML = `${prev ? `<a href="paper.html?id=${prev.key}" title="${esc(prev.title)}">← ${esc(prev.short)}</a>` : ''}${next ? `<a href="paper.html?id=${next.key}" title="${esc(next.title)}">${esc(next.short)} →</a>` : ''}`;
  $('#foot').innerHTML = `<span>Survey Visual Atlas · MLR570 · survey reference [${P.n}] · key <b>${esc(P.key)}</b></span><span>Every surface is a real function; every ball runs a real optimizer. <a href="https://github.com/farisnasser/survey-visual-atlas/tree/main/scripts/check-math">Math checks</a></span>`;
}

async function renderNotes(P) {
  const sec = $('#notes');
  if (P.notes_page) {
    sec.classList.add('as-page', 'dense');
    const dl = P.notes ? `<a class="nt-dl" href="${esc(P.notes)}" download="${esc(P.key)}-study-notes.pdf">Download PDF notes</a>` : '';
    sec.innerHTML = `<div class="sec-h"><h2>Study notes</h2><span class="sub">Our study notes, with a first-principles primer</span>${dl}</div><div class="loading">Loading notes…</div>`;
    sec.addEventListener('click', e => {   // contents links open a folded section before jumping to it
      const a = e.target.closest('a[href^="#n-"]'); if (!a) return;
      const el = document.getElementById(a.getAttribute('href').slice(1)); if (!el) return;
      const f = el.closest('details'); if (f) f.open = true;
    });
    try {
      const md = await fetch(`papers/${P.key}/notes.md`).then(r => { if (!r.ok) throw new Error(`notes.md ${r.status}`); return r.text(); });
      sec.querySelector('.loading').outerHTML = await renderNotesArticle(md);
      if (location.hash.startsWith('#n-')) { const el = document.getElementById(location.hash.slice(1)); if (el) { const f = el.closest('details'); if (f) f.open = true; el.scrollIntoView(); } }
    } catch (err) { sec.querySelector('.loading').textContent = 'Could not load the notes.'; console.error(err); }
    return;
  }
  if (!P.notes) { sec.innerHTML = `<div class="sec-h"><h2>Study notes</h2></div><div class="nonotes">No notes PDF for this paper yet. When one lands in the Drive notes folder, the next sync adds it here.</div>`; return; }
  sec.innerHTML = `<div class="sec-h"><h2>Study notes</h2><span class="sub">${esc(P.notes_title || '')} · <a href="${esc(P.notes)}" download>Download PDF</a> · <a href="${esc(P.notes)}" target="_blank" rel="noopener">Open in new tab</a></span></div><div class="pdfview" id="pdfview"><div class="loading">Loading notes…</div></div>`;
  const io = new IntersectionObserver(async es => {
    if (!es[0].isIntersecting) return; io.disconnect();
    try { await renderPdf(P.notes, $('#pdfview')); } catch (err) { $('#pdfview').innerHTML = `<div class="loading">Could not render the PDF here. <a href="${esc(P.notes)}">Open it directly</a>.</div>`; console.error(err); }
  }, { rootMargin: '600px' });
  io.observe(sec);
}

function loadScript(src) { return new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s); }); }
async function renderPdf(url, box) {
  const V = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/';
  if (!window.pdfjsLib) await loadScript(V + 'pdf.min.js');
  pdfjsLib.GlobalWorkerOptions.workerSrc = V + 'pdf.worker.min.js';
  const doc = await pdfjsLib.getDocument(url).promise;
  box.innerHTML = '';
  const dpr = Math.min(2, devicePixelRatio);
  for (let n = 1; n <= doc.numPages; n++) {
    const page = await doc.getPage(n);
    const w = Math.min(900, box.clientWidth - 16), vp0 = page.getViewport({ scale: 1 }), vp = page.getViewport({ scale: w / vp0.width * dpr });
    const c = document.createElement('canvas'); c.width = vp.width; c.height = vp.height; c.setAttribute('aria-label', `Notes page ${n}`);
    box.appendChild(c);
    await page.render({ canvasContext: c.getContext('2d'), viewport: vp }).promise;
  }
}

// ---------- math.md: "## tabId" sections; $$..$$ display, $..$ inline, ### heading, - bullets, > source line ----------
function parseMath(text) {
  const out = {}; let cur = null;
  for (const line of text.split(/\r?\n/)) {
    const m = line.match(/^##\s+([\w-]+)\s*(?:\|\s*(.*))?$/);
    if (m) { cur = out[m[1]] = { title: m[2] || '', lines: [] }; continue; }
    if (cur) cur.lines.push(line);
  }
  return out;
}
const inline = s => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/\$([^$]+?)\$/g, (_, t) => tex(t.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), false));
function tex(t, display) { try { return window.katex.renderToString(t, { displayMode: display, throwOnError: false, strict: false }); } catch { return esc(t); } }
async function renderMath(sec) {
  for (let i = 0; i < 100 && !window.katex; i++) await new Promise(r => setTimeout(r, 50));
  if (!sec) return '<p>No math panel for this tab.</p>';
  let html = '', para = [], list = [], disp = null;
  const flushP = () => { if (para.length) { html += `<p>${inline(para.join(' '))}</p>`; para = []; } };
  const flushL = () => { if (list.length) { html += `<ul>${list.map(x => `<li>${inline(x)}</li>`).join('')}</ul>`; list = []; } };
  for (const raw of sec.lines) {
    const line = raw.trim();
    if (disp !== null) { if (line.endsWith('$$')) { disp.push(line.slice(0, -2)); html += tex(disp.join('\n'), true); disp = null; } else disp.push(line); continue; }
    if (line.startsWith('$$')) { flushP(); flushL(); const rest = line.slice(2); if (rest.endsWith('$$') && rest.length > 2) html += tex(rest.slice(0, -2), true); else disp = [rest]; continue; }
    if (!line) { flushP(); flushL(); continue; }
    if (line.startsWith('### ')) { flushP(); flushL(); html += `<h4>${inline(line.slice(4))}</h4>`; continue; }
    if (line.startsWith('- ')) { flushP(); list.push(line.slice(2)); continue; }
    if (line.startsWith('> ')) { flushP(); flushL(); html += `<p class="src">${inline(line.slice(2))}</p>`; continue; }
    para.push(line);
  }
  flushP(); flushL();
  return html;
}

class Shell {
  constructor(visual, math, data, P, meta) {
    this.v = visual; this.math = math; this.P = P; this.meta = meta;
    this.stage = new Stage($('#gl'), $('#labels'));
    this.plot = new Plot($('#plot2d'));
    this.ctx = { data, stage: this.stage, plot: this.plot, charts: {}, vals: {}, base: `papers/${P.key}/`, shell: this };
    this.listeners = [];
    const tabs = $('#tabs');
    tabs.innerHTML = visual.tabs.map(t => `<button role="tab" type="button" data-tab="${t.id}" aria-selected="false">${esc(t.name)}</button>`).join('');
    tabs.querySelectorAll('button').forEach(b => b.addEventListener('click', () => this.show(b.dataset.tab)));
    const btn = $('#mathBtn'), dr = $('#mathDrawer');
    const setMath = open => { dr.hidden = !open; btn.setAttribute('aria-expanded', open); store.set('atlas.math', open ? '1' : '0'); };
    btn.addEventListener('click', () => setMath(dr.hidden));
    $('#mdClose').addEventListener('click', () => setMath(false));
    if (store.get('atlas.math') === '1' && innerWidth > 900 && !EMBED) setMath(true);
    this.stage.start();
    if (LOCAL) window.__atlas = this;   // test hook (see tick)
    const want = (location.hash || '').slice(1);
    this.show(visual.tabs.some(t => t.id === want) ? want : (THUMB && this.meta.thumb) || visual.tabs[0].id, true);
    addEventListener('hashchange', () => { const h = location.hash.slice(1); if (h !== this.tab?.id && visual.tabs.some(t => t.id === h)) this.show(h); });
    if (THUMB) this.thumbnail();
  }
  listen(el, type, fn, opts) { el.addEventListener(type, fn, opts); this.listeners.push([el, type, fn, opts]); }
  async show(id, first = false) {
    const ctx = this.ctx, prev = this.tab;
    if (prev?.leave) prev.leave(ctx);
    this.listeners.forEach(([el, t, f, o]) => el.removeEventListener(t, f, o)); this.listeners = [];
    cancelAnimationFrame(this.raf2d);
    this.stage.clear(); this.stage.frameFns = [];
    Object.values(ctx.charts).forEach(p => p.ro.disconnect()); ctx.charts = {};
    const t = this.tab = this.v.tabs.find(x => x.id === id);
    document.querySelectorAll('#tabs button').forEach(b => b.setAttribute('aria-selected', b.dataset.tab === id));
    if (!first || location.hash) history.replaceState(null, '', `#${id}`);
    $('#capT').textContent = t.title || t.name;
    $('#capP').textContent = t.caption || '';
    $('#capNote').textContent = t.note || '';
    $('#capStat').textContent = '';
    this.legend(t.legend || []);
    $('#hint').textContent = t.mode === '2d' ? (t.hint || '') : 'drag to orbit · pinch / scroll to zoom';
    this.buildControls(t);
    this.buildSide(t);
    const is2d = t.mode === '2d';
    $('#stageBox').classList.toggle('mode2d', is2d);
    $('#gl').style.visibility = is2d ? 'hidden' : 'visible';
    $('#plot2d').hidden = !is2d;
    this.stage.paused = is2d;
    this.plot.clear().set(Object.assign({ xlabel: '', ylabel: '', title: '', logx: false, logy: false, xlim: null, ylim: null, pad: [56, 18, 40, 56], legend: true }, t.plot || {}));
    if (!is2d) { this.stage.view(t.cam || [0, 4.4, 6.2], t.tgt || [0, .3, 0]); this.stage.resize(); }
    try { t.build?.(ctx); } catch (err) { console.error(`[atlas] build failed in tab ${id}`, err); }
    if (!is2d && t.frame) this.stage.onFrame(dt => t.frame(ctx, dt));
    if (is2d) {
      this.plot.draw();
      if (t.frame) { let last = performance.now(); const loop = now => { this.raf2d = requestAnimationFrame(loop); if (document.hidden) return; const dt = Math.min(.05, (now - last) / 1000); last = now; t.frame(ctx, dt); }; this.raf2d = requestAnimationFrame(loop); }
    }
    Object.values(ctx.charts).forEach(p => p.draw());
    $('#mdTitle').textContent = this.math[id]?.title || `Math · ${t.name}`;
    $('#mdBody').innerHTML = await renderMath(this.math[id]);
    if (LOCAL) this.selfCheck(t);
  }
  /** Advance n frames without requestAnimationFrame (automated audits, background tabs). */
  tick(n = 60) {
    const t = this.tab;
    for (let i = 0; i < n; i++) { if (t.mode === '2d') t.frame?.(this.ctx, 1 / 60); else { this.stage.stepBalls(); this.stage.frameFns.forEach(f => f(1 / 60)); } }
    if (t.mode !== '2d') this.stage.renderOnce();
    return document.querySelector('#capStat').textContent;
  }
  selfCheck(t) {
    for (const d of this.stage.defs || []) if (d._analytic && !d._checked) { d._checked = true; checkGrad2(`${this.P.key}/${t.id}`, d.L, d.g, d.checkDom || d.dom); }
  }
  legend(items) { $('#legend').innerHTML = items.map(([c, s]) => `<span><i style="background:${c}"></i>${esc(s)}</span>`).join(''); }
  buildControls(t) {
    const box = $('#sliders'), acts = $('#acts'), ctx = this.ctx;
    box.innerHTML = ''; acts.innerHTML = ''; ctx.vals = {};
    for (const c of t.controls || []) {
      ctx.vals[c.id] = c.value;
      const row = document.createElement('div');
      if (c.type === 'seg') {
        row.className = 'sl segrow';
        row.innerHTML = `<span>${esc(c.label)}</span><span class="seg">${c.options.map(([v, l]) => `<button type="button" data-v="${esc(v)}" aria-pressed="${String(v) === String(c.value)}">${esc(l)}</button>`).join('')}</span>`;
        row.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
          row.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b));
          const v = c.options.find(o => String(o[0]) === b.dataset.v)[0]; ctx.vals[c.id] = v; this.changed(c, v);
        }));
      } else {
        row.className = 'sl';
        const f = c.fmt || (v => (+v).toFixed(c.digits ?? 2));
        row.innerHTML = `<label for="c_${c.id}">${esc(c.label)}</label><input id="c_${c.id}" type="range" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.value}"><output>${f(c.value)}</output>`;
        const inp = row.querySelector('input'), out = row.querySelector('output');
        inp.addEventListener('input', () => { const v = +inp.value; ctx.vals[c.id] = v; out.textContent = f(v); if (c.live !== false) this.changed(c, v); });
        inp.addEventListener('change', () => { if (c.live === false) this.changed(c, +inp.value); });
      }
      box.appendChild(row);
    }
    ctx.set = (id, v) => {   // programmatic update: moves the slider, does not call onControl
      ctx.vals[id] = v; const inp = $(`#c_${id}`); if (!inp) return;
      inp.value = v; const c = (t.controls || []).find(x => x.id === id), f = c.fmt || (x => (+x).toFixed(c.digits ?? 2));
      inp.nextElementSibling.textContent = f(v);
    };
    const actions = t.actions || (t.action ? [{ label: t.action, fn: c => t.drop?.(c) }] : []);
    actions.forEach((a, i) => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'act' + (i ? ' ghost' : ''); b.textContent = a.label;
      b.addEventListener('click', () => a.fn(ctx)); acts.appendChild(b);
    });
    ctx.stat = html => { $('#capStat').innerHTML = html; };
    ctx.caption = s => { $('#capP').textContent = s; };
    ctx.legend = items => this.legend(items);
    ctx.listen = (el, type, fn, o) => this.listen(el, type, fn, o);
  }
  changed(c, v) { const t = this.tab; if (t.onControl) t.onControl(this.ctx, c.id, v); else t.drop?.(this.ctx); }
  buildSide(t) {
    const side = $('#side'), viz = $('#viz');
    side.innerHTML = '';
    const charts = t.charts || [];
    side.hidden = !charts.length; viz.classList.toggle('has-side', !!charts.length);
    for (const ch of charts) {
      const box = document.createElement('div'); box.className = 'chart';
      const c = document.createElement('canvas'); c.style.height = (ch.h || 220) + 'px'; c.setAttribute('role', 'img'); c.setAttribute('aria-label', ch.title || ch.id);
      box.appendChild(c); side.appendChild(box);
      this.ctx.charts[ch.id] = new Plot(c, Object.assign({ pad: [30, 12, 34, 50] }, ch));
    }
  }
  async thumbnail() {
    await new Promise(r => setTimeout(r, 1200));
    this.tick(this.meta.thumbTicks ?? 240);
    const st = this.stage, r = st.renderer;
    r.setPixelRatio(1); r.setSize(720, 450, false); st.camera.aspect = 1.6; st.camera.fov = 42; st.camera.updateProjectionMatrix();
    st.renderOnce();
    const url = r.domElement.toDataURL('image/webp', .86);
    if (qs.has('save')) await fetch(`/__thumb?id=${encodeURIComponent(this.P.key)}`, { method: 'POST', body: url }).catch(() => { });
    if (window.parent !== window) window.parent.postMessage({ atlasThumb: this.P.key, url }, '*');
    window.__thumbDone = true;
  }
}

const _surface = Stage.prototype.surface;
Stage.prototype.surface = function (d, o) { (this.defs ||= []).push(d); return _surface.call(this, d, o); };
const _clear = Stage.prototype.clear;
Stage.prototype.clear = function () { this.defs = []; return _clear.call(this); };

main().catch(err => { console.error(err); $('#ph').innerHTML = `<h1>Something went wrong</h1><p class="meta">${esc(err.message)}</p>`; });
