// One live visual inside a crash-course chapter. Same tab API as the paper visuals (see the paper-visuals skill):
// { title, caption, note, legend, mode: '3d' | '2d', cam, tgt, plot, controls, actions | action+drop, charts,
//   build(ctx), frame(ctx, dt), onControl(ctx, id, v) }. Mounted lazily when its slot nears the viewport.
import { Stage } from '../lib/atlas3d.js';
import { Plot } from '../lib/charts.js';
import { checkGrad2 } from '../lib/math.js';

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const LOCAL = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);

export class Widget {
  constructor(slot, t, name) {
    this.slot = slot; this.t = t; this.name = name; this.visible = false;
    const is2d = t.mode === '2d';
    slot.innerHTML = `<figure class="lv-box ${is2d ? 'lv-2d' : 'lv-3d'}">
      <div class="lv-stage">${is2d ? '<canvas class="plot2d"></canvas>' : '<div class="gl"></div>'}<div class="labels"></div>${is2d ? '' : '<span class="lv-hint">drag to orbit · scroll to zoom</span>'}</div>
      <div class="lv-charts"></div>
      <figcaption class="lv-hud">
        <div class="lv-cap"><b>${esc(t.title)}</b>${t.caption ? `<p>${esc(t.caption)}</p>` : ''}<p class="note">${esc(t.note || '')}</p><div class="stat"></div><div class="legend"></div></div>
        <div class="lv-ctrl"><div class="sliders"></div><div class="acts"></div></div>
      </figcaption></figure>`;
    const $ = s => slot.querySelector(s);
    this.$ = $;
    this.listeners = [];
    const ctx = this.ctx = { charts: {}, vals: {}, widget: this };
    if (is2d) { ctx.plot = new Plot($('.plot2d')); ctx.plot.set(Object.assign({ pad: [34, 16, 40, 54], legend: true }, t.plot || {})); }
    else {
      ctx.stage = new Stage($('.gl'), $('.labels'));
      ctx.stage.view(t.cam || [0, 4.4, 6.2], t.tgt || [0, .3, 0]);
    }
    ctx.stat = html => { $('.stat').innerHTML = html; };
    ctx.caption = s => { const p = $('.lv-cap p'); if (p) p.textContent = s; };
    ctx.legend = items => { $('.legend').innerHTML = items.map(([c, s]) => `<span><i style="background:${c}"></i>${esc(s)}</span>`).join(''); };
    ctx.listen = (el, type, fn, o) => { el.addEventListener(type, fn, o); this.listeners.push([el, type, fn, o]); };
    ctx.legend(t.legend || []);
    this.buildControls();
    this.buildCharts();
    new IntersectionObserver(es => { this.visible = es[0].isIntersecting; }).observe(slot);
    try { t.build?.(ctx); } catch (err) { console.error(`[course] build failed in ${name}`, err); }
    if (is2d) {
      ctx.plot.draw();
      if (t.frame) {
        let last = performance.now();
        const loop = now => { requestAnimationFrame(loop); const dt = Math.min(.05, (now - last) / 1000); last = now; if (!this.visible || document.hidden) return; t.frame(ctx, dt); };
        requestAnimationFrame(loop);
      }
    } else {
      if (t.frame) ctx.stage.onFrame(dt => t.frame(ctx, dt));
      ctx.stage.start();
      if (LOCAL) for (const d of ctx.stage.defs || []) if (d._analytic) checkGrad2(`course/${name}`, d.L, d.g, d.checkDom || d.dom);
    }
    Object.values(ctx.charts).forEach(p => p.draw());
  }
  buildControls() {
    const t = this.t, ctx = this.ctx, box = this.$('.sliders'), acts = this.$('.acts');
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
        const f = c.fmt || (v => (+v).toFixed(c.digits ?? 2)), id = `lv_${this.name}_${c.id}`;
        row.innerHTML = `<label for="${id}">${esc(c.label)}</label><input id="${id}" type="range" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.value}"><output>${f(c.value)}</output>`;
        const inp = row.querySelector('input'), out = row.querySelector('output');
        inp.addEventListener('input', () => { const v = +inp.value; ctx.vals[c.id] = v; out.textContent = f(v); if (c.live !== false) this.changed(c, v); });
        inp.addEventListener('change', () => { if (c.live === false) this.changed(c, +inp.value); });
        row.dataset.id = c.id;
      }
      box.appendChild(row);
    }
    ctx.set = (id, v) => {   // programmatic update: moves the slider, does not call onControl
      ctx.vals[id] = v; const row = box.querySelector(`[data-id="${id}"]`); if (!row) return;
      const c = t.controls.find(x => x.id === id), f = c.fmt || (x => (+x).toFixed(c.digits ?? 2));
      row.querySelector('input').value = v; row.querySelector('output').textContent = f(v);
    };
    const actions = t.actions || (t.action ? [{ label: t.action, fn: c => t.drop?.(c) }] : []);
    actions.forEach((a, i) => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'act' + (i ? ' ghost' : ''); b.textContent = a.label;
      b.addEventListener('click', () => a.fn(ctx)); acts.appendChild(b);
    });
  }
  buildCharts() {
    const box = this.$('.lv-charts'), charts = this.t.charts || [];
    box.hidden = !charts.length;
    for (const ch of charts) {
      const wrap = document.createElement('div'); wrap.className = 'chart';
      const c = document.createElement('canvas'); c.style.height = (ch.h || 200) + 'px'; c.setAttribute('role', 'img'); c.setAttribute('aria-label', ch.title || ch.id);
      wrap.appendChild(c); box.appendChild(wrap);
      this.ctx.charts[ch.id] = new Plot(c, Object.assign({ pad: [30, 12, 34, 50] }, ch));
    }
  }
  changed(c, v) { const t = this.t; if (t.onControl) t.onControl(this.ctx, c.id, v); else t.drop?.(this.ctx); }
  /** Advance n frames without requestAnimationFrame (audits). */
  tick(n = 60) {
    const t = this.t, ctx = this.ctx;
    for (let i = 0; i < n; i++) { if (t.mode === '2d') t.frame?.(ctx, 1 / 60); else { ctx.stage.stepBalls(); ctx.stage.frameFns.forEach(f => f(1 / 60)); } }
    if (t.mode !== '2d') ctx.stage.renderOnce();
    return this.$('.stat').textContent;
  }
}

// keep track of the surfaces each stage draws (gradient self-check on localhost)
const _surface = Stage.prototype.surface;
if (!Stage.prototype._lvWrapped) {
  Stage.prototype._lvWrapped = true;
  Stage.prototype.surface = function (d, o) { (this.defs ||= []).push(d); return _surface.call(this, d, o); };
}
