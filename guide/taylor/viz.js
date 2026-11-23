// Chapter 2 (survey Sections II-A, II-B). New visuals 'uvloss' and 'steepest' (numbers in
// scripts/check-math/guide3_notes.py); reused: crash-course stops 2 and 4 (course_slope.py, course_curvature.py).
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { arrow } from '../../src/learn/plot2d.js';
import slope from '../../course/slope/viz.js';
import curv from '../../course/curvature/viz.js';

// ---------- 'uvloss': a two-weight linear "network" u*v fit to target 1 is already non-convex ----------
const fuv = (u, v) => (u * v - 1) ** 2, guv = (u, v) => [2 * (u * v - 1) * v, 2 * (u * v - 1) * u];
const hyper = (s) => { const a = []; for (let u = .38; u <= 2.6; u += .02) a.push([s * u, s / u]); return a; };
const uvloss = {
  title: 'Two linear layers, one target: L(u, v) = (uv − 1)² is already non-convex',
  caption: 'Prediction uv for input 1, target 1. Every point on the curve uv = 1 (green) has zero loss, so the minimum is not unique; the origin has zero gradient but loss 1, a saddle. Roll gradient descent from different starts and it settles at different points of the curve.',
  note: 'Exact gradient 2(uv − 1)(v, u), η = 0.05, up to 600 steps. Height drawn as √L. Hessian at the origin [[0, −2], [−2, 0]]: eigenvalues ±2.',
  legend: [[HEX.ball, 'gradient descent'], [HEX.good, 'uv = 1 (zero loss)'], [HEX.neg, 'saddle at the origin']],
  mode: '3d', cam: [2.4, 3.6, 3.6], tgt: [0, .1, 0],
  controls: [{ id: 'ang', label: 'start direction (degrees)', min: 0, max: 359, step: 1, value: 200, digits: 0 }],
  action: 'Roll',
  build(ctx) {
    ctx.d = def({ L: fuv, g: guv, dom: [-2.6, 2.6, -2.6, 2.6], h: v => Math.sqrt(v), fit: 1.5, hClip: [0, 4.5] });
    ctx.d.hess = (u, v) => [2 * v * v, 2 * (2 * u * v - 1), 2 * u * u];
    ctx.stage.surface(ctx.d);
    for (const s of [1, -1]) ctx.stage.line(ctx.d, hyper(s), C.good, .95, .03);
    ctx.stage.marker(ctx.d, 0, 0, C.neg, .07);
    this.drop(ctx);
  },
  onControl(ctx) { this.drop(ctx); },
  drop(ctx) {
    const a = ctx.vals.ang * Math.PI / 180, st = ctx.stage; st.clearBalls();
    ctx.start = [1.6 * Math.cos(a), 1.6 * Math.sin(a)];
    ctx.ball = st.ball(ctx.d, { start: ctx.start, step: opt.gd(.05), speed: 1.5, maxSteps: 600, stop: b => fuv(...b.p) < 1e-10 });
  },
  frame(ctx) {
    const b = ctx.ball; if (!b) return;
    const [u, v] = b.p;
    ctx.stat(`start (${ctx.start.map(c => c.toFixed(2)).join(', ')}) · step ${b.steps} · (u, v) = (${u.toFixed(3)}, ${v.toFixed(3)}) · uv = ${(u * v).toFixed(4)} · loss ${fuv(u, v).toExponential(2)}${fuv(u, v) < 1e-8 ? ' · reached a zero-loss point' : ''}`);
  },
};

// ---------- 'steepest': among steps of length r, -g gives the most negative g^T Delta ----------
const G = [3, 4], NG = 5;
const steepest = {
  title: 'Steepest descent: every step on the circle, scored by g·Δ',
  caption: 'All steps of the same length r form a circle. The first-order model predicts the change g·Δ = ‖g‖ r cos φ, where φ is the angle between the step and the gradient. It is most negative when the step points straight against g.',
  note: 'g = (3, 4), so ‖g‖ = 5. With r = 0.5 the best step is Δ = (−0.3, −0.4) and g·Δ = −2.5.',
  legend: [[HEX.pos, 'gradient g'], [HEX.ball, 'step Δ'], ['#e8ecf8', 'predicted change g·Δ']],
  mode: '2d',
  plot: { xlabel: 'Δ₁', ylabel: 'Δ₂', xlim: [-1.2, 1.2], ylim: [-1.2, 1.2], equal: true, legend: false },
  charts: [{ id: 'c', title: 'predicted change g·Δ against the angle φ', xlabel: 'angle between Δ and g (degrees)', ylabel: 'g·Δ', xlim: [0, 360], ylim: [-5.5, 5.5], h: 190, legend: false }],
  controls: [
    { id: 'phi', label: 'angle φ between Δ and g (degrees)', min: 0, max: 360, step: 1, value: 120, digits: 0 },
    { id: 'r', label: 'step length r', min: .1, max: 1, step: .05, value: .5 },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { phi, r } = ctx.vals, a0 = Math.atan2(G[1], G[0]), a = a0 + phi * Math.PI / 180;
    const D = [r * Math.cos(a), r * Math.sin(a)], val = G[0] * D[0] + G[1] * D[1];
    const P = ctx.plot.clear();
    P.line(Array.from({ length: 121 }, (_, i) => [r * Math.cos(2 * Math.PI * i / 120), r * Math.sin(2 * Math.PI * i / 120)]), { color: 'rgba(232,236,248,.35)', width: 1.2 });
    arrow(P, 0, 0, G[0] / NG, G[1] / NG, { color: HEX.pos, width: 2 });
    arrow(P, 0, 0, -r * G[0] / NG, -r * G[1] / NG, { color: HEX.good, width: 1.2, head: 7 });
    arrow(P, 0, 0, D[0], D[1], { color: HEX.ball, width: 2.6 });
    P.text(G[0] / NG * 1.08, G[1] / NG * 1.08, 'g (drawn at unit length)', { color: HEX.pos });
    P.draw();
    const C2 = ctx.charts.c.clear();
    C2.fn(t => NG * r * Math.cos(t * Math.PI / 180), { color: '#e8ecf8', width: 1.8 });
    C2.hline(0, { color: 'rgba(232,236,248,.25)' });
    C2.vline(180, { color: HEX.good, dash: [4, 4], text: 'opposite g' });
    C2.scatter([[phi, val]], { color: HEX.ball, r: 5 });
    C2.draw();
    ctx.stat(`Δ = (${D[0].toFixed(3)}, ${D[1].toFixed(3)}), ‖Δ‖ = ${r.toFixed(2)} · φ = ${phi}° · g·Δ = ${val.toFixed(3)} · best possible at this length: −‖g‖r = ${(-NG * r).toFixed(3)} at Δ = −r g/‖g‖ = (${(-r * .6).toFixed(2)}, ${(-r * .8).toFixed(2)})`);
  },
};

export default { uvloss, steepest, plane: slope.plane, quad: curv.quad };
