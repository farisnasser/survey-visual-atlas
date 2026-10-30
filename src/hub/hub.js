// Hub: every reference of the survey (data/papers.json, built from the survey PDF), organized by the survey's Fig. 5
// taxonomy, as a numbered reference list, by year, or by concept. Papers with notes or a visual get full cards.
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const statusOf = p => p.visual && p.notes ? ['complete', 'Complete'] : p.visual ? ['visual', 'Visual · no notes'] : p.notes ? ['notes', 'Notes only'] : ['queued', 'Not started'];
const STATUS = [['complete', 'Complete'], ['notes', 'Notes only'], ['visual', 'Visual only'], ['queued', 'Not started']];
const RQ = { III: 'RQ1 · What minima, saddles and flat regions do neural-network losses have?', IV: 'RQ2 · How do GD and SGD actually move near critical points?',
  V: 'RQ3 · What helps an optimizer leave saddles and flat regions, and at what cost?', VI: 'How overparameterization changes the landscape and the dynamics.', VII: 'RQ4 · What happens in practical training: large steps, sharpness, modern optimizers.' };
const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'];
const secTitle = s => s.replace(/^([IVX]+)\. (.)(.*)$/, (_, r, a, b) => `${r}. ${a}${b.toLowerCase()}`).replace(/\bgd\b/gi, 'GD').replace(/\bsgd\b/gi, 'SGD');
const qs = new URLSearchParams(location.search);
const state = { q: '', status: new Set(), branch: new Set(), view: qs.get('concept') ? 'concepts' : (qs.get('view') || 'map'), concept: qs.get('concept') || '' };
let DB, PAPERS, TAX, picks = [];

const hasPage = p => p.visual || p.notes;
const branchOf = p => p.branches?.length ? p.branches.map(b => b[0]) : ['BG'];
const art = key => {
  let h = 0; for (const c of key) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  const a = 8 + (h % 14), b = 22 + ((h >> 5) % 18), c = 30 + ((h >> 10) % 20);
  return `<svg viewBox="0 0 200 120" aria-hidden="true"><g fill="none" stroke="#2a9c96" stroke-width="1.4">
    ${[0, 1, 2, 3, 4].map(i => `<path d="M0 ${70 + i * 9} C ${40 + a} ${b + i * 7}, ${100 + c} ${110 - i * 3}, 200 ${40 + i * 10}" opacity="${.9 - i * .15}"/>`).join('')}
  </g></svg>`;
};

function card(p) {
  const [st, label] = statusOf(p);
  const th = p.thumb ? `<img src="${esc(p.thumb)}" alt="" loading="lazy">` : `<div class="ph-art">${art(p.key)}<span class="k">${esc(p.key)}</span></div>`;
  return `<a class="card" href="paper.html?id=${encodeURIComponent(p.key)}" data-key="${esc(p.key)}" data-visual="${p.visual ? 1 : 0}">
    <div class="th">${th}<span class="chip st-${st}"><i></i>${label}</span><span class="rn">[${p.n}]</span><span class="pickmark"></span></div>
    <div class="body"><h3>${esc(p.title)}</h3><div class="by">${esc(p.short)} · ${esc(shortVenue(p.venue))}</div>
    ${p.result ? `<div class="res">${esc(p.result)}</div>` : ''}
    ${p.tags?.length ? `<div class="tags">${p.tags.slice(0, 4).map(t => `<span>${esc(t)}</span>`).join('')}</div>` : ''}</div></a>`;
}
function row(p) {
  const [st, label] = statusOf(p);
  return `<a class="refrow" href="paper.html?id=${encodeURIComponent(p.key)}" title="${esc(p.ref)}"><span class="rn">[${p.n}]</span><span class="rt"><b>${esc(p.title)}</b><span>${esc(p.short)} · ${esc(shortVenue(p.venue))}</span></span><span class="chip st-${st}"><i></i>${label}</span></a>`;
}
const shortVenue = v => { const m = String(v || '').match(/\(([A-Z][A-Za-z]+)\)/); return m ? m[1] : String(v || '').replace(/, vol\..*$/, '').replace(/^arXiv:.*/, 'arXiv'); };
function block(ps) {   // full cards for papers with pages, compact rows for the rest
  const pages = ps.filter(hasPage), rest = ps.filter(p => !hasPage(p));
  return `${pages.length ? `<div class="grid">${pages.map(card).join('')}</div>` : ''}${rest.length ? `<div class="rows">${rest.map(row).join('')}</div>` : ''}`;
}

function filtered() {
  const q = state.q.trim().toLowerCase();
  return PAPERS.filter(p => {
    if (state.status.size && !state.status.has(statusOf(p)[0])) return false;
    if (state.branch.size && !branchOf(p).some(b => state.branch.has(b))) return false;
    if (!q) return true;
    return [p.title, p.authors, p.key, p.venue, String(p.year), `[${p.n}]`, ...(p.tags || []), p.result, ...(p.branches || []).map(b => b[1])].join(' ').toLowerCase().includes(q);
  });
}

function renderStats() {
  const n = PAPERS.length, notes = PAPERS.filter(p => p.notes).length, vis = PAPERS.filter(p => p.visual).length, done = PAPERS.filter(p => p.notes && p.visual).length;
  const tile = (v, s, col, frac) => `<div class="stat"><b>${v}</b><span>${s}</span><div class="bar"><i style="width:${(frac * 100).toFixed(0)}%;background:${col}"></i></div></div>`;
  $('#stats').innerHTML = tile(n, 'references in the survey', 'var(--muted)', 1) + tile(notes, 'with study notes', 'var(--pos)', notes / n) + tile(vis, 'with interactive visuals', 'var(--flat)', vis / n) + tile(done, 'complete (notes + visual)', 'var(--good)', done / n);
}

function renderFilters() {
  const grp = (name, key, items) => `<div class="fgroup"><span>${name}</span>${items.map(([v, l]) => `<button type="button" class="fchip" data-f="${key}" data-v="${esc(v)}" aria-pressed="${state[key].has(v)}">${esc(l)}</button>`).join('')}</div>`;
  $('#filters').innerHTML = grp('status', 'status', STATUS.filter(([v]) => PAPERS.some(p => statusOf(p)[0] === v)))
    + grp('branch', 'branch', [...TAX.map(b => [b.id, `${b.id} · ${b.name.split(' ').slice(0, 2).join(' ')}`]), ['BG', 'Background']]);
  $('#filters').querySelectorAll('.fchip').forEach(b => b.addEventListener('click', () => {
    const set = state[b.dataset.f]; set.has(b.dataset.v) ? set.delete(b.dataset.v) : set.add(b.dataset.v);
    b.setAttribute('aria-pressed', set.has(b.dataset.v)); render();
  }));
}

function renderMap(list) {
  const ids = new Set(list.map(p => p.n));
  let html = TAX.map(br => {
    const leaves = br.leaves.map(leaf => {
      const ps = PAPERS.filter(p => ids.has(p.n) && p.branches?.some(b => b[0] === br.id && b[1] === leaf)).sort((a, b) => a.n - b.n);
      if (!ps.length) return '';
      const done = ps.filter(hasPage).length;
      return `<div class="leaf"><h3>${esc(leaf)} <span>${ps.map(p => `[${p.n}]`).join(' ')} · ${done}/${ps.length} with pages</span></h3>${block(ps)}</div>`;
    }).join('');
    if (!leaves) return '';
    const all = PAPERS.filter(p => p.branches?.some(b => b[0] === br.id)), pg = all.filter(hasPage).length;
    return `<section class="group branch" id="b-${br.id}"><h2><em>${br.id}</em>${esc(br.name)} <span>${pg} of ${all.length} references with pages</span></h2><p class="gsub">${esc(RQ[br.id] || '')}</p>${leaves}</section>`;
  }).join('');
  const bg = list.filter(p => !p.branches?.length);
  if (bg.length) {
    const bySec = {};
    bg.forEach(p => { const s = p.cited[0] || 'Uncited'; (bySec[s] = bySec[s] || []).push(p); });
    const order = Object.keys(bySec).sort((a, b) => ROMAN.indexOf(a.split('.')[0]) - ROMAN.indexOf(b.split('.')[0]));
    html += `<section class="group branch" id="b-BG"><h2><em>·</em>Background <span>${bg.filter(hasPage).length} of ${bg.length} with pages</span></h2><p class="gsub">References outside the Fig. 5 taxonomy, grouped by the section that first cites them.</p>
      ${order.map(s => `<div class="leaf"><h3>${esc(secTitle(s))} <span>${bySec[s].length}</span></h3>${block(bySec[s].sort((a, b) => a.n - b.n))}</div>`).join('')}</section>`;
  }
  return html;
}

function render() {
  const list = filtered(), v = $('#view');
  document.querySelectorAll('#views button').forEach(b => b.setAttribute('aria-pressed', b.dataset.view === state.view));
  if (!list.length) { v.innerHTML = `<div class="empty">No references match. Clear a filter or the search.</div>`; return; }
  if (state.view === 'map') v.innerHTML = renderMap(list);
  else if (state.view === 'refs') {
    v.innerHTML = `<section class="group"><h2>All references <span>${list.length} of ${PAPERS.length}, numbered as in the survey</span></h2><div class="rows">${list.sort((a, b) => a.n - b.n).map(row).join('')}</div></section>`;
  } else if (state.view === 'timeline') {
    const years = [...new Set(list.map(p => p.year || 0))].sort((a, b) => a - b);
    v.innerHTML = years.map(y => { const ps = list.filter(p => (p.year || 0) === y).sort((a, b) => a.n - b.n); return `<div class="year"><div class="y">${y || '—'}<small>${ps.length} ref${ps.length > 1 ? 's' : ''}</small></div><div>${block(ps)}</div></div>`; }).join('');
  } else {
    const counts = {};
    list.forEach(p => (p.tags || []).forEach(t => { counts[t] = (counts[t] || 0) + 1; }));
    const tags = Object.keys(counts).sort((a, b) => counts[b] - counts[a] || a.localeCompare(b));
    if (!tags.length) { v.innerHTML = `<div class="empty">Concepts come from papers with visuals; none match the current filter.</div>`; return; }
    if (!state.concept || !counts[state.concept]) state.concept = tags[0];
    const ps = list.filter(p => (p.tags || []).includes(state.concept)).sort((a, b) => a.n - b.n);
    v.innerHTML = `<div class="concepts">${tags.map(t => `<button type="button" data-c="${esc(t)}" aria-pressed="${t === state.concept}">${esc(t)}<b>${counts[t]}</b></button>`).join('')}</div>
      <section class="group"><h2>${esc(state.concept)} <span>${ps.length} paper${ps.length > 1 ? 's' : ''}</span></h2><div class="cpapers">${ps.map(p => {
        const tab = p.tag_tabs?.[state.concept];
        return `<a class="crow" href="paper.html?id=${p.key}${tab ? '#' + tab : ''}"><div><b>[${p.n}] ${esc(p.title)}</b><br><span>${esc(p.short)} · ${esc(shortVenue(p.venue))}</span></div>${tab ? `<em>open “${esc((p.tabs || []).find(x => x[0] === tab)?.[1] || tab)}” tab →</em>` : ''}</a>`;
      }).join('')}</div></section>`;
    v.querySelectorAll('.concepts button').forEach(b => b.addEventListener('click', () => { state.concept = b.dataset.c; history.replaceState(null, '', `?concept=${encodeURIComponent(state.concept)}`); render(); }));
  }
  if (document.body.classList.contains('comparing')) wirePicks();
  if (location.hash && state.view === 'map') document.querySelector(location.hash)?.scrollIntoView();
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
  PAPERS = DB.papers; TAX = DB.taxonomy;
  renderStats(); renderFilters(); render();
  $('#q').addEventListener('input', e => { state.q = e.target.value; render(); });
  document.querySelectorAll('#views button').forEach(b => b.addEventListener('click', () => { state.view = b.dataset.view; history.replaceState(null, '', state.view === 'map' ? './' : `?view=${state.view}`); render(); }));
  $('#cmpBtn').addEventListener('click', () => {
    const on = !document.body.classList.contains('comparing');
    document.body.classList.toggle('comparing', on); $('#cmpBtn').setAttribute('aria-pressed', on); $('#cmpBar').hidden = !on;
    if (!on) picks = []; updateCompare(); render();
  });
  $('#cmpX').addEventListener('click', () => $('#cmpBtn').click());
  $('#foot').innerHTML = `<span>Built from the survey’s reference list (${PAPERS.length} references, Fig. 5 taxonomy) · synced ${esc(DB.synced)}</span><span>Faris Nasser &amp; Salah Moussa · MLR570, American University of Sharjah · <a href="https://github.com/farisnasser/survey-visual-atlas">source</a></span>`;
}
main().catch(err => { $('#view').innerHTML = `<div class="empty">Could not load the reference list (${esc(err.message)}).</div>`; });
