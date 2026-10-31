// Stop 1 visuals. Numbers checked in scripts/check-math/course_goal.py.
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';

// eight data points for the line fit (fixed; the same in the check script)
const X = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], Y = [-0.95, -0.42, -0.18, 0.41, 0.62, 1.18, 1.31, 1.86], N = X.length;
const loss = (w, b) => { let s = 0; for (let i = 0; i < N; i++) { const r = w * X[i] + b - Y[i]; s += r * r; } return s / N; };
const grad = (w, b) => { let gw = 0, gb = 0; for (let i = 0; i < N; i++) { const r = w * X[i] + b - Y[i]; gw += 2 * r * X[i] / N; gb += 2 * r / N; } return [gw, gb]; };
const xm = X.reduce((a, v) => a + v, 0) / N, ym = Y.reduce((a, v) => a + v, 0) / N;
const W_STAR = X.reduce((a, v, i) => a + (v - xm) * (Y[i] - ym), 0) / X.reduce((a, v) => a + (v - xm) ** 2, 0), B_STAR = ym - W_STAR * xm;
const f4 = v => v.toFixed(4), f3 = v => v.toFixed(3);

const fit = {
  title: 'The loss landscape of a line fit',
  caption: 'Each point of the surface is one line (w, b); its height is that line\'s loss on the eight data points below. Move the sliders, or let the ball roll downhill.',
  note: 'Surface height is drawn on a log scale so the bottom is visible; the readout is the true loss.',
  legend: [[HEX.ball, 'current line (w, b)'], [HEX.good, 'best line (w*, b*)'], [HEX.neg, 'errors ŷᵢ − yᵢ']],
  mode: '3d', cam: [3.6, 3.4, 4.6], tgt: [0, .5, 0],
  controls: [
    { id: 'w', label: 'slope w', min: -.6, max: 2.2, step: .01, value: 0 },
    { id: 'b', label: 'intercept b', min: -1.1, max: 1.7, step: .01, value: 0 },
  ],
  actions: [
    { label: 'Roll downhill', fn: ctx => fit.roll(ctx) },
    { label: 'Reset', fn: ctx => { ctx.set('w', 0); ctx.set('b', 0); fit.place(ctx); } },
  ],
  charts: [{ id: 'data', title: 'the data and the current line', xlabel: 'x', ylabel: 'y', xlim: [-2, 2.5], ylim: [-1.6, 2.4], h: 210, legend: false }],
  build(ctx) {
    const d = ctx.d = def({ L: loss, g: grad, dom: [-.6, 2.2, -1.1, 1.7] });
    ctx.stage.surface(d);
    ctx.stage.marker(d, W_STAR, B_STAR, C.good, .12);
    ctx.stage.label('slope w →', d.W(1.6, -1.1, .15), '', HEX.muted);
    ctx.stage.label('intercept b →', d.W(-.6, 1.2, .15), '', HEX.muted);
    ctx.dot = ctx.stage.dot(d.W(0, 0, .07), C.ball, .075);
    this.place(ctx);
  },
  place(ctx) {
    if (ctx.ball) { ctx.stage.removeBall(ctx.ball); ctx.ball = null; }
    this.show(ctx, ctx.vals.w, ctx.vals.b);
  },
  show(ctx, w, b) {
    ctx.dot.visible = !ctx.ball;
    ctx.dot.position.copy(ctx.d.W(w, b, .07));
    const P = ctx.charts.data, res = X.map((x, i) => [[x, Y[i]], [x, w * x + b]]);
    P.clear();
    res.forEach(s => P.line(s, { color: HEX.neg, width: 1.4, alpha: .8 }));
    P.fn(x => W_STAR * x + B_STAR, { color: HEX.good, width: 1.2, dash: [5, 4], alpha: .8 });
    P.fn(x => w * x + b, { color: HEX.ball, width: 2.2 });
    P.scatter(X.map((x, i) => [x, Y[i]]), { color: '#e8ecf8', r: 3.6 });
    P.draw();
    ctx.stat(`w = ${w.toFixed(3)}, b = ${b.toFixed(3)} · loss 𝓛(w, b) = ${f4(loss(w, b))} · best possible 𝓛* = ${f4(loss(W_STAR, B_STAR))}`);
  },
  roll(ctx) {
    if (ctx.ball) ctx.stage.removeBall(ctx.ball);
    ctx.ball = ctx.stage.ball(ctx.d, { start: [ctx.vals.w, ctx.vals.b], step: opt.gd(.15), speed: .35, stop: b => Math.hypot(...grad(...b.p)) < 1e-4 });
  },
  frame(ctx) {
    const b = ctx.ball; if (!b) return;
    ctx.set('w', b.p[0]); ctx.set('b', b.p[1]);
    if (b.steps !== ctx.lastSteps) { ctx.lastSteps = b.steps; this.show(ctx, b.p[0], b.p[1]); }
  },
  onControl(ctx) { this.place(ctx); },
};

// one-parameter loss with three valleys
const F = t => .1 * t * t + .5 * Math.sin(3 * t), dF = t => .2 * t + 1.5 * Math.cos(3 * t);
const MINIMA = [-2.5045, -0.5013, 1.50352];   // Newton-refined in the check script
const valleys = {
  title: 'Which valley do you end in?',
  caption: 'Pick a starting point θ₀ and let the ball roll downhill (gradient descent with a small step). It stops in the valley below its start.',
  note: '𝓛(θ) = 0.1θ² + 0.5 sin 3θ: three local minima, one global (green).',
  legend: [[HEX.good, 'global minimum'], ['#e8ecf8', 'local minimum'], [HEX.ball, 'the path']],
  mode: '2d',
  plot: { xlabel: 'θ', ylabel: '𝓛(θ)', xlim: [-3, 3], ylim: [-.65, 1.25], legend: false },
  controls: [{ id: 't0', label: 'start θ₀', min: -3, max: 3, step: .01, value: 2.6 }],
  actions: [{ label: 'Roll downhill', fn: ctx => { ctx.path = [ctx.vals.t0]; ctx.run = true; } }],
  build(ctx) { ctx.path = [ctx.vals.t0]; ctx.run = false; this.draw(ctx); },
  draw(ctx) {
    const P = ctx.plot.clear();
    P.fn(F, { color: '#8ea0d0', width: 2 });
    MINIMA.forEach((m, i) => P.scatter([[m, F(m)]], { color: i === 1 ? HEX.good : '#e8ecf8', r: 6 }));
    P.line(ctx.path.map(t => [t, F(t)]), { color: HEX.ball, width: 1.4, alpha: .55, dots: 2 });
    const t = ctx.path[ctx.path.length - 1];
    P.scatter([[t, F(t)]], { color: HEX.ball, r: 7 });
    P.draw();
    const end = MINIMA.reduce((a, m) => Math.abs(m - t) < Math.abs(a - t) ? m : a);
    const settled = Math.abs(dF(t)) < 1e-4;
    ctx.stat(settled
      ? `stopped at θ ≈ ${end.toFixed(2)} with loss ${f3(F(end))} after ${ctx.path.length - 1} steps · ${end === MINIMA[1] ? 'the global minimum' : `a local minimum (the global one has loss ${f3(F(MINIMA[1]))})`}`
      : `θ = ${t.toFixed(3)} · loss ${f3(F(t))} · slope ${dF(t).toFixed(3)}`);
  },
  frame(ctx) {
    if (!ctx.run) return;
    for (let k = 0; k < 2; k++) {
      const t = ctx.path[ctx.path.length - 1];
      if (Math.abs(dF(t)) < 1e-4 || ctx.path.length > 4000) { ctx.run = false; break; }
      ctx.path.push(t - .05 * dF(t));
    }
    this.draw(ctx);
  },
  onControl(ctx) { ctx.path = [ctx.vals.t0]; ctx.run = false; this.draw(ctx); },
};

export default { fit, valleys };
