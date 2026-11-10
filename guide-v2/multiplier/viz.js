// Chapter 5 (survey Section II-F). New visual 'multiplier' (numbers checked in scripts/check-math/guide_multiplier.py);
// reused crash-course visuals 'zigzag' (course_curvature.py) and 'escape' (course_escape.py).
import { HEX } from '../../src/lib/atlas3d.js';
import curv from '../../course/curvature/viz.js';
import esc from '../../course/escape/viz.js';

const T = 30;
const regime = el => el < 0 ? ['grows monotonically', 'negative curvature: GD leaves the critical point (saddle escape, Section III)', HEX.neg]
  : el === 0 ? ['stays put', 'flat direction: no first- or second-order force', HEX.flat]
  : el < 1 ? ['shrinks, same sign every step', 'stable: GD creeps in', HEX.good]
  : el === 1 ? ['vanishes in one step', 'stable: one step lands exactly', HEX.good]
  : el < 2 ? ['shrinks, flipping sign every step', 'stable but bouncing', HEX.good]
  : el === 2 ? ['keeps its size, flipping sign', 'the edge: neither shrinks nor grows', HEX.flat]
  : ['grows, flipping sign every step', 'excess curvature: GD bounces out (edge of stability, Section IV)', HEX.pos];

const multiplier = {
  title: 'One multiplier, three regimes',
  caption: 'Near a critical point, each GD step multiplies the distance along a Hessian eigenvector by m = 1 − ηλ. Pick ηλ and watch the distance u_t over 30 steps. The chart shows where ηλ sits on the line m = 1 − ηλ.',
  note: 'Exact: u_{t+1} = (1 − ηλ) u_t from u₀ = 1, the linearized GD map along one eigen-direction. Points beyond ±4 are clipped in the top plot; the log chart shows their true size.',
  legend: [[HEX.ball, 'u_t'], [HEX.neg, 'λ < 0: saddle escape'], [HEX.good, '0 < ηλ < 2: stable'], [HEX.pos, 'ηλ > 2: too sharp']],
  mode: '2d',
  plot: { xlabel: 'step t', ylabel: 'u_t (distance along the eigenvector)', xlim: [0, T], ylim: [-4, 4], legend: false },
  controls: [
    { id: 'el', label: 'ηλ (step × curvature)', min: -.6, max: 2.6, step: .01, value: -.1, digits: 2 },
    { id: 'pre', type: 'seg', label: 'presets', value: '', options: [['-0.1', 'saddle −0.1'], ['0.5', 'stable 0.5'], ['1.5', 'bouncing 1.5'], ['2', 'edge 2'], ['2.2', 'too sharp 2.2']] },
  ],
  actions: [{ label: 'Replay', fn: ctx => { ctx.k = 0; } }],
  charts: [
    { id: 'm', title: 'the multiplier m = 1 − ηλ', xlabel: 'ηλ', ylabel: 'm', xlim: [-.6, 2.6], ylim: [-1.8, 1.8], h: 200, legend: false },
    { id: 'a', title: '|u_t| on a log scale', xlabel: 'step t', ylabel: '|u_t|', logy: true, xlim: [0, T], ylim: [1e-10, 1e4], h: 200, legend: false },
  ],
  build(ctx) { ctx.k = 0; this.draw(ctx); },
  onControl(ctx, id, v) {
    if (id === 'pre') { ctx.set('el', +v); ctx.vals.el = +v; }
    ctx.k = 0; this.draw(ctx);
  },
  frame(ctx, dt) {
    if (ctx.k >= T) return;
    ctx.acc = (ctx.acc || 0) + dt;
    if (ctx.acc < .06) return;
    ctx.acc = 0; ctx.k++; this.draw(ctx);
  },
  draw(ctx) {
    const el = Math.round(ctx.vals.el * 100) / 100, m = 1 - el, k = ctx.k;
    const u = Array.from({ length: T + 1 }, (_, t) => m ** t);
    const [what, why, col] = regime(el);
    const P = ctx.plot.clear();
    P.hline(0, { color: 'rgba(255,255,255,.35)' });
    P.hline(1, { color: 'rgba(255,255,255,.18)', dash: [3, 4] }); P.hline(-1, { color: 'rgba(255,255,255,.18)', dash: [3, 4] });
    const shown = u.slice(0, k + 1).map((v, t) => [t, Math.max(-4, Math.min(4, v))]);
    P.custom((g, p) => { g.strokeStyle = 'rgba(255,122,89,.35)'; g.lineWidth = 1; for (const [t, v] of shown) { g.beginPath(); g.moveTo(p.sx(t), p.sy(0)); g.lineTo(p.sx(t), p.sy(v)); g.stroke(); } });
    P.line(shown, { color: HEX.ball, width: 1.4, dots: 3 });
    P.draw();
    const M = ctx.charts.m.clear();
    M.band(-.6, 0, { color: 'rgba(255,93,115,.12)' }); M.band(0, 2, { color: 'rgba(95,224,176,.08)' }); M.band(2, 2.6, { color: 'rgba(124,196,255,.12)' });
    M.hline(1, { color: 'rgba(255,255,255,.3)', dash: [3, 4], text: '+1' }); M.hline(-1, { color: 'rgba(255,255,255,.3)', dash: [3, 4], text: '−1' }); M.hline(0, { color: 'rgba(255,255,255,.2)' });
    M.fn(x => 1 - x, { color: '#e8ecf8', width: 2 });
    M.scatter([[el, m]], { color: col, r: 6 });
    M.text(-.55, -1.45, 'saddle escape', { color: HEX.neg }); M.text(.55, 1.5, 'stable band', { color: HEX.good }); M.text(2.05, 1.5, 'too sharp', { color: HEX.pos });
    M.draw();
    const A = ctx.charts.a.clear();
    A.hline(1, { color: 'rgba(255,255,255,.3)', dash: [3, 4] });
    A.line(u.slice(0, k + 1).map((v, t) => [t, Math.abs(v)]), { color: HEX.ball, width: 1.6, dots: 2 });
    A.draw();
    const last = Math.abs(u[T]);
    ctx.stat(`ηλ = ${el.toFixed(2)} → m = 1 − ηλ = ${m.toFixed(2)}, |m| = ${Math.abs(m).toFixed(2)} · the distance ${what} · after ${T} steps |u| = ${last === 0 ? '0' : last.toPrecision(3)} · ${why}`);
  },
};

export default { multiplier, zigzag: curv.zigzag, escape: esc.escape };
