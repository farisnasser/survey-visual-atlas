// Hub: every paper from data/papers.json (generated from the Papers Master sheet), grouped by theme, year or concept.
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const statusOf = p => p.visual && p.notes ? ['complete', 'Complete'] : p.visual ? ['visual', 'Visual · no notes'] : p.notes ? ['notes', 'Notes only'] : ['queued', 'Queued'];
const SECTIONS = [
  ['Geometry', 'RQ1 · What minima, saddles and flat regions do neural-network losses have?'],
  ['Dynamics', 'RQ2 · How do GD and SGD actually move near critical points?'],
  ['Escape methods', 'RQ3 · What helps an optimizer leave saddles and flat regions, and at what cost?'],
  ['Theory vs practice', 'RQ4 · When do the theorems’ assumptions hold for real networks?'],
  ['Preliminaries', 'Background results the rest of the survey builds on.'],
  ['Introduction', ''],
];
const STATUS = [['complete', 'Complete'], ['notes', 'Notes only'], ['visual', 'Visual only'], ['queued', 'Queued']];
const qs = new URLSearchParams(location.search);
const state = { q: '', status: new Set(), phase: new Set(), section: new Set(), view: qs.get('concept') ? 'concepts' : (qs.get('view') || 'themes'), concept: qs.get('concept') || '' };
let DB, PAPERS, picks = [];

const art = key => {
  let h = 0; for (const c of key) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  const a = 8 + (h % 14), b = 22 + ((h >> 5) % 18), c = 30 + ((h >> 10) % 20);
  return `<svg viewBox="0 0 200 120" aria-hidden="true"><g fill="none" stroke="#2a9c96" stroke-width="1.4">
    ${[0, 1, 2, 3, 4].map(i => `<path d="M0 ${70 + i * 9} C ${40 + a} ${b + i * 7}, ${100 + c} ${110 - i * 3}, 200 ${40 + i * 10}" opacity="${.9 - i * .15}"/>`).join('')}
  </g></svg>`;
};

function card(p) {
  const [st, label] = statusOf(p);
  const href = `paper.html?id=${encodeURIComponent(p.key)}`;
  const th = p.thumb ? `<img src="${esc(p.thumb)}" alt="" loading="lazy">` : `<div class="ph-art">${art(p.key)}<span class="k">${esc(p.key)}</span></div>`;
  return `<a class="card" href="${href}" data-key="${esc(p.key)}" data-visual="${p.visual ? 1 : 0}">
    <div class="th">${th}<span class="chip st-${st}"><i></i>${label}</span><span class="pickmark"></span></div>
    <div class="body"><h3>${esc(p.title)}</h3><div class="by">${esc(p.short)} · ${esc(p.venue)} · ${esc(p.phase.replace(/^\d\s*/, ''))}</div>
    ${p.result ? `<div class="res">${esc(p.result)}</div>` : ''}
    ${p.tags?.length ? `<div class="tags">${p.tags.slice(0, 4).map(t => `<span>${esc(t)}</span>`).join('')}</div>` : ''}</div></a>`;
}

function filtered() {
  const q = state.q.trim().toLowerCase();
  return PAPERS.filter(p => {
    if (state.status.size && !state.status.has(statusOf(p)[0])) return false;
    if (state.phase.size && !state.phase.has(p.phase)) return false;
    if (state.section.size && !state.section.has(p.section)) return false;
    if (!q) return true;
    return [p.title, p.authors, p.key, p.venue, String(p.year), p.section, ...(p.tags || []), p.result].join(' ').toLowerCase().includes(q);
  });
}

function renderStats() {
  const n = PAPERS.length, notes = PAPERS.filter(p => p.notes).length, vis = PAPERS.filter(p => p.visual).length, done = PAPERS.filter(p => p.notes && p.visual).length;
  const tile = (v, s, col, frac) => `<div class="stat"><b>${v}</b><span>${s}</span><div class="bar"><i style="width:${(frac * 100).toFixed(0)}%;background:${col}"></i></div></div>`;
  $('#stats').innerHTML = tile(n, 'papers in the tracker', 'var(--muted)', 1) + tile(notes, 'with study notes', 'var(--pos)', notes / n) + tile(vis, 'with interactive visuals', 'var(--flat)', vis / n) + tile(done, 'complete (notes + visual)', 'var(--good)', done / n);
}

function renderFilters() {
  const phases = [...new Set(PAPERS.map(p => p.phase))].sort();
  const sections = SECTIONS.map(s => s[0]).filter(s => PAPERS.some(p => p.section === s));
  const grp = (name, key, items) => `<div class="fgroup"><span>${name}</span>${items.map(([v, l]) => `<button type="button" class="fchip" data-f="${key}" data-v="${esc(v)}" aria-pressed="${state[key].has(v)}">${esc(l)}</button>`).join('')}</div>`;
  $('#filters').innerHTML = grp('status', 'status', STATUS.filter(([v]) => PAPERS.some(p => statusOf(p)[0] === v))) + grp('phase', 'phase', phases.map(p => [p, p.replace(/^(\d)\s*/, '$1 · ')])) + grp('section', 'section', sections.map(s => [s, s]));
  $('#filters').querySelectorAll('.fchip').forEach(b => b.addEventListener('click', () => {
    const set = state[b.dataset.f]; set.has(b.dataset.v) ? set.delete(b.dataset.v) : set.add(b.dataset.v);
    b.setAttribute('aria-pressed', set.has(b.dataset.v)); render();
  }));
}

function render() {
  const list = filtered(), v = $('#view');
  document.querySelectorAll('#views button').forEach(b => b.setAttribute('aria-pressed', b.dataset.view === state.view));
  if (!list.length) { v.innerHTML = `<div class="empty">No papers match. Clear a filter or the search.</div>`; return; }
  if (state.view === 'themes') {
    v.innerHTML = SECTIONS.map(([s, sub]) => {
      const ps = list.filter(p => p.section === s); if (!ps.length) return '';
      ps.sort((a, b) => (statusOf(a)[0] === 'complete' ? 0 : 1) - (statusOf(b)[0] === 'complete' ? 0 : 1) || a.year - b.year);
      return `<section class="group"><h2>${esc(s)} <span>${ps.length}</span></h2>${sub ? `<p class="gsub">${esc(sub)}</p>` : ''}<div class="grid">${ps.map(card).join('')}</div></section>`;
    }).join('');
  } else if (state.view === 'timeline') {
    const years = [...new Set(list.map(p => p.year))].sort((a, b) => a - b);
    v.innerHTML = years.map(y => { const ps = list.filter(p => p.year === y); return `<div class="year"><div class="y">${y}<small>${ps.length} paper${ps.length > 1 ? 's' : ''}</small></div><div class="grid">${ps.map(card).join('')}</div></div>`; }).join('');
  } else {
    const counts = {};
    list.forEach(p => (p.tags || []).forEach(t => { counts[t] = (counts[t] || 0) + 1; }));
    const tags = Object.keys(counts).sort((a, b) => counts[b] - counts[a] || a.localeCompare(b));
    if (!state.concept || !counts[state.concept]) state.concept = tags[0] || '';
    const ps = list.filter(p => (p.tags || []).includes(state.concept));
    v.innerHTML = `<div class="concepts">${tags.map(t => `<button type="button" data-c="${esc(t)}" aria-pressed="${t === state.concept}">${esc(t)}<b>${counts[t]}</b></button>`).join('')}</div>
      <section class="group"><h2>${esc(state.concept)} <span>${ps.length} paper${ps.length > 1 ? 's' : ''}</span></h2><div class="cpapers">${ps.map(p => {
        const tab = p.tag_tabs?.[state.concept];
        return `<a class="crow" href="paper.html?id=${p.key}${tab ? '#' + tab : ''}"><div><b>${esc(p.title)}</b><br><span>${esc(p.short)} · ${esc(p.venue)}</span></div>${tab ? `<em>open “${esc((p.tabs || []).find(x => x[0] === tab)?.[1] || tab)}” tab →</em>` : ''}</a>`;
      }).join('')}</div></section>`;
    v.querySelectorAll('.concepts button').forEach(b => b.addEventListener('click', () => { state.concept = b.dataset.c; history.replaceState(null, '', `?concept=${encodeURIComponent(state.concept)}`); render(); }));
  }
  if (document.body.classList.contains('comparing')) wirePicks();
}

function wirePicks() {
  document.querySelectorAll('.card').forEach(c => {
    c.classList.toggle('pick', picks.includes(c.dataset.key));
    const i = picks.indexOf(c.dataset.key); c.querySelector('.pickmark').textContent = i >= 0 ? (i ? 'B' : 'A') : '';
    c.onclick = e => {
      if (!document.body.classList.contains('comparing')) return;
      e.preventDefault(); if (c.dataset.visual !== '1') return;
      const k = c.dataset.key; picks = picks.includes(k) ? picks.filter(x => x !== k) : [...picks, k].slice(-2);
      updateCompare(); wirePicks();
    };
  });
}
function updateCompare() {
  const go = $('#cmpGo');
  $('#cmpMsg').textContent = picks.length === 2 ? `${PAPERS.find(p => p.key === picks[0]).short} vs ${PAPERS.find(p => p.key === picks[1]).short}` : `Pick ${2 - picks.length} more paper${picks.length ? '' : 's'} with a visual`;
  go.setAttribute('aria-disabled', picks.length !== 2);
  go.href = picks.length === 2 ? `compare.html?a=${picks[0]}&b=${picks[1]}` : '#';
}

async function main() {
  DB = await fetch('data/papers.json').then(r => r.json());
  PAPERS = DB.papers;
  renderStats(); renderFilters(); render();
  $('#q').addEventListener('input', e => { state.q = e.target.value; render(); });
  document.querySelectorAll('#views button').forEach(b => b.addEventListener('click', () => { state.view = b.dataset.view; history.replaceState(null, '', state.view === 'themes' ? './' : `?view=${state.view}`); render(); }));
  $('#cmpBtn').addEventListener('click', () => {
    const on = !document.body.classList.contains('comparing');
    document.body.classList.toggle('comparing', on); $('#cmpBtn').setAttribute('aria-pressed', on); $('#cmpBar').hidden = !on;
    if (!on) picks = []; updateCompare(); render();
  });
  $('#cmpX').addEventListener('click', () => $('#cmpBtn').click());
  $('#foot').innerHTML = `<span>Synced from the Papers Master sheet on ${esc(DB.synced)} · ${PAPERS.length} papers</span><span>Faris Nasser &amp; Salah Moussa · MLR570, American University of Sharjah · <a href="https://github.com/farisnasser/survey-visual-atlas">source</a></span>`;
}
main().catch(err => { $('#view').innerHTML = `<div class="empty">Could not load the paper list (${esc(err.message)}).</div>`; });
