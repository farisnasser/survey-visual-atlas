// Rendering audit (localhost only). In the browser: await import('/scripts/audit_pages.js'); await __audit([urls], width)
window.__audit = async function (urls, w = 1280) {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const out = [];
  for (const u of urls) {
    const f = document.createElement('iframe');
    f.style.cssText = `position:fixed;left:0;top:0;width:${w}px;height:900px;opacity:0.01;z-index:-1`;
    const errs = [];
    f.src = u; document.body.append(f);
    await new Promise(r => f.onload = r);
    const W = f.contentWindow, D = f.contentDocument;
    const oe = W.console.error.bind(W.console);
    W.console.error = (...a) => { errs.push(a.map(x => x?.message || String(x)).join(' ').slice(0, 200)); oe(...a); };
    W.addEventListener('error', e => errs.push('uncaught: ' + e.message));
    W.addEventListener('unhandledrejection', e => errs.push('rejection: ' + (e.reason?.message || e.reason)));
    for (let i = 0; i < 40 && !D.querySelector('h1'); i++) await sleep(150);
    await sleep(800);
    try { W.__course?.mountAll?.(); } catch (e) { errs.push('mountAll: ' + e.message); }
    await sleep(1500);
    const wid = W.__course?.widgets || {};
    for (const k in wid) { try { wid[k].tick?.(120); } catch (e) { errs.push(`tick ${k}: ${e.message}`); } }
    await sleep(400);
    const r = { u, errs: [...errs, ...(W.__atlasErrors || [])] };
    r.h1 = D.querySelector('h1')?.textContent?.slice(0, 50) || '(no h1)';
    r.katexErr = [...D.querySelectorAll('.katex-error')].map(e => e.textContent.slice(0, 60));
    // raw markup left in visible text
    const tw = D.createTreeWalker(D.body, NodeFilter.SHOW_TEXT, { acceptNode: n => n.parentElement.closest('.katex, script, style, code, pre, textarea') ? 2 : 1 });
    const raw = []; let n;
    while ((n = tw.nextNode())) { const s = n.textContent; if (/\$\$|\$[^\s$][^$]*\$|\*\*|:::|\[\[|\]\]|\\(frac|lambda|eta|nabla|theta|mathbb|tfrac)/.test(s)) raw.push(s.trim().slice(0, 90)); }
    r.raw = raw.slice(0, 6);
    r.badImg = [...D.querySelectorAll('img')].filter(i => i.complete && !i.naturalWidth).map(i => i.getAttribute('src'));
    const slots = [...D.querySelectorAll('.lv[data-viz]')];
    r.viz = slots.length;
    r.unmounted = slots.filter(s => !s.querySelector('.lv-box')).map(s => s.dataset.viz);
    r.noStat = slots.filter(s => s.querySelector('.lv-box') && !s.querySelector('.stat')?.textContent.trim()).map(s => s.dataset.viz);
    r.blank2d = slots.filter(s => { const c = s.querySelector('canvas.plot2d'); if (!c || !c.width) return false; const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let k = 0; for (let i = 3; i < d.length; i += 4 * 97) if (d[i]) k++; return k < 5; }).map(s => s.dataset.viz);
    r.no3d = slots.filter(s => s.querySelector('.gl') && !s.querySelector('.gl canvas')).map(s => s.dataset.viz);
    r.overflow = D.documentElement.scrollWidth > D.documentElement.clientWidth + 1 ? D.documentElement.scrollWidth : 0;
    r.emptyMain = (D.querySelector('#main')?.textContent.trim().length || 0) < 200;
    f.remove();
    const bad = r.errs.length || r.katexErr.length || r.raw.length || r.badImg.length || r.unmounted.length || r.noStat.length || r.blank2d.length || r.no3d.length || r.overflow || r.emptyMain;
    out.push(bad ? r : { u, ok: true, viz: r.viz });
  }
  return out;
};

// paper pages: every tab of every paper (paper.html?id=<key>), math drawer and notes
window.__auditPapers = async function (keys, w = 1280) {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const out = [];
  for (const k of keys) {
    const f = document.createElement('iframe');
    f.style.cssText = `position:fixed;left:0;top:0;width:${w}px;height:900px;opacity:0.01;z-index:-1`;
    f.src = `/paper.html?id=${k}`; document.body.append(f);
    await new Promise(r => f.onload = r);
    const W = f.contentWindow, D = f.contentDocument, r = { k, tabs: {}, errs: [] };
    for (let i = 0; i < 60 && !W.__atlas; i++) await sleep(150);
    if (!W.__atlas) { r.errs.push('shell never started'); r.errs.push(...(W.__atlasErrors || [])); out.push(r); f.remove(); continue; }
    await sleep(500);
    for (const t of W.__atlas.v.tabs) {
      const e0 = W.__atlasErrors.length;
      try { await W.__atlas.show(t.id); W.__atlas.tick(60); } catch (e) { W.__atlasErrors.push(`${t.id}: ${e.message}`); }
      await sleep(120);
      const issues = [];
      const e = W.__atlasErrors.slice(e0); if (e.length) issues.push(...e.map(x => x.slice(0, 160)));
      if (t.mode === '2d') { const c = D.querySelector('#plot2d'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4 * 97) if (d[i]) n++; if (n < 5) issues.push('blank 2d plot'); }
      const ke = [...D.querySelectorAll('#mdBody .katex-error')].map(x => 'math drawer katex: ' + x.textContent.slice(0, 50)); issues.push(...ke);
      if (issues.length) r.tabs[t.id] = issues;
    }
    await sleep(800);
    r.notesKatex = [...D.querySelectorAll('.notes .katex-error, #notes .katex-error')].map(x => x.textContent.slice(0, 60)).slice(0, 5);
    const tw = D.createTreeWalker(D.querySelector('.notes, #notes') || D.body, NodeFilter.SHOW_TEXT, { acceptNode: n => n.parentElement.closest('.katex, script, style, code, pre, #mdBody, .lv-hud, #capStat, figcaption') ? 2 : 1 });
    const raw = []; let n; while ((n = tw.nextNode())) { const s = n.textContent; if (/\$\$|\$[^\s$\d][^$]{0,80}\$|\*\*|:::|\\(frac|lambda|eta|nabla|theta|mathbb|tfrac)/.test(s)) raw.push(s.trim().slice(0, 90)); }
    r.raw = raw.slice(0, 4);
    r.badImg = [...D.querySelectorAll('img')].filter(i => i.complete && !i.naturalWidth).map(i => i.getAttribute('src'));
    f.remove();
    if (Object.keys(r.tabs).length || r.errs.length || r.notesKatex.length || r.raw.length || r.badImg.length) out.push(r);
  }
  return out;
};
