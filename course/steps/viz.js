// Stop 3 visuals. Numbers checked in scripts/check-math/course_steps.py.
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';

const verdict = k => k <= 0 ? 'no progress' : k < 1 ? 'slides down one side' : Math.abs(k - 1) < 1e-9 ? 'lands on the minimum in one step' : k < 2 ? 'jumps across, but converges' : Math.abs(k - 2) < 1e-9 ? 'bounces forever' : 'diverges';

// ---------- GD on a parabola ----------
const parabola = {
  title: 'Step size on a parabola: slow, perfect, bouncing, exploding',
  caption: 'Gradient descent on f(x) = (a/2)x² from x₀ = 2. Every step multiplies x by 1 − ηa. Slide η past 1/a and then past 2/a.',
  note: 'The dots are the iterates x₀, x₁, x₂, … (up to 25); the lines join consecutive steps.',
  legend: [[HEX.ball, 'GD iterates']],
  mode: '2d',
  plot: { xlabel: 'x', ylabel: 'f(x)', xlim: [-3, 3], ylim: [-.3, 9], legend: false },
  controls: [
    { id: 'eta', label: 'step size η', min: .01, max: .8, step: .005, value: .1, digits: 3 },
    { id: 'a', label: 'curvature a', min: .5, max: 6, step: .1, value: 4, digits: 1 },
  ],
  actions: [{ label: 'Replay', fn: ctx => { ctx.k = 0; } }],
  build(ctx) { ctx.k = 0; this.draw(ctx); },
  onControl(ctx) { ctx.k = 0; this.draw(ctx); },
  frame(ctx, dt) { if (ctx.k < 25) { ctx.acc = (ctx.acc || 0) + dt * 8; if (ctx.acc >= 1) { ctx.acc = 0; ctx.k++; this.draw(ctx); } } },
  draw(ctx) {
    const { eta, a } = ctx.vals, f = x => a / 2 * x * x, P = ctx.plot.clear(), xs = [2];
    for (let t = 0; t < 25; t++) xs.push((1 - eta * a) * xs[t]);
    P.fn(f, { color: '#8ea0d0', width: 2 });
    const shown = xs.slice(0, ctx.k + 1).map(x => [x, f(x)]);
    P.line(shown, { color: HEX.ball, width: 1.4, alpha: .7 });
    P.scatter(shown, { color: HEX.ball, r: 3.5 });
    P.scatter([[0, 0]], { color: HEX.good, r: 5 });
    P.draw();
    const k = eta * a;
    ctx.stat(`ηa = ${k.toFixed(3)} · factor 1 − ηa = ${(1 - k).toFixed(3)} · 1/a = ${(1 / a).toFixed(3)}, 2/a = ${(2 / a).toFixed(3)} · |x₂₅| = ${Math.abs(xs[25]).toPrecision(3)} → ${verdict(k)}`);
  },
};

// ---------- GD on an elongated bowl: one step size, two curvatures ----------
const bowlL = (x, y) => .5 * (x * x + 10 * y * y), bowlG = (x, y) => [x, 10 * y];
const bowl = {
  title: 'One step size, two curvatures',
  caption: 'f(x, y) = ½(x² + 10y²) bends 10 times more sharply along y than along x. Find a step that is fast along x without blowing up along y.',
  note: 'Curvatures 1 (x) and 10 (y), so L = 10: stable only for η < 2/10 = 0.2. The y-axis is drawn stretched so the valley is visible.',
  legend: [[HEX.ball, 'gradient descent'], [HEX.good, 'minimum']],
  mode: '3d', cam: [3.7, 4.0, 4.3], tgt: [0, .55, 0],
  controls: [{ id: 'eta', label: 'step size η', min: .01, max: .22, step: .001, value: .17, digits: 3 }],
  action: 'Drop the ball',
  build(ctx) {
    ctx.d = def({ L: bowlL, g: bowlG, dom: [-1.6, 1.6, -.7, .7], square: true });
    ctx.stage.surface(ctx.d);
    ctx.stage.marker(ctx.d, 0, 0, C.good, .1);
    this.drop(ctx);
  },
  drop(ctx) {
    ctx.stage.clearBalls();
    const eta = ctx.vals.eta;
    ctx.ball = ctx.stage.ball(ctx.d, { start: [-1.45, .5], step: opt.gd(eta), speed: .25, maxSteps: 400,
      stop: b => bowlL(...b.p) < 1e-8 || Math.abs(b.p[1]) >= .7 - 1e-9 || Math.abs(b.p[0]) >= 1.6 - 1e-9 });
  },
  onControl(ctx) { this.drop(ctx); },
  frame(ctx) {
    const b = ctx.ball, eta = ctx.vals.eta; if (!b) return;
    const out = Math.abs(b.p[1]) >= .7 - 1e-9 || Math.abs(b.p[0]) >= 1.6 - 1e-9;
    ctx.stat(`η = ${eta.toFixed(3)} · factor along x: 1 − η = ${(1 - eta).toFixed(3)}, along y: 1 − 10η = ${(1 - 10 * eta).toFixed(3)} · step ${b.steps} · ${out ? 'left the picture: diverged' : `f = ${bowlL(...b.p).toExponential(2)}`}`);
  },
};

// ---------- Lipschitz: the graph stays inside a double cone ----------
const FL = x => .8 * Math.sin(2 * x) + .3 * x, dFL = x => 1.6 * Math.cos(2 * x) + .3;
const lipschitz = {
  title: 'Lipschitz: never steeper than K',
  caption: 'Place the cone at any point of the graph. f is K-Lipschitz exactly when the graph never enters the red regions (slopes steeper than ±K) wherever the cone is placed.',
  note: 'f(x) = 0.8 sin 2x + 0.3x; its steepest slope is max |f′| = 1.9, so it is K-Lipschitz exactly for K ≥ 1.9.',
  legend: [['#8ea0d0', 'f'], [HEX.neg, 'forbidden for slope K'], [HEX.ball, 'where the graph breaks the rule']],
  mode: '2d',
  plot: { xlabel: 'x', ylabel: 'f(x)', xlim: [-3, 3], ylim: [-2.2, 2.2], legend: false },
  controls: [
    { id: 'x0', label: 'cone at x₀', min: -2.8, max: 2.8, step: .01, value: -.3 },
    { id: 'K', label: 'constant K', min: .5, max: 3, step: .01, value: 1.2 },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { x0, K } = ctx.vals, y0 = FL(x0), P = ctx.plot.clear();
    P.custom((g, p) => {   // forbidden region: |y - y0| > K |x - x0|  (upper and lower wedges)
      g.fillStyle = 'rgba(255,93,115,.14)';
      for (const s of [1, -1]) {
        g.beginPath(); g.moveTo(p.sx(x0), p.sy(y0));
        g.lineTo(p.sx(x0 - 10), p.sy(y0 + s * 10 * K)); g.lineTo(p.sx(x0 - 10), p.sy(y0 + s * 50)); g.lineTo(p.sx(x0 + 10), p.sy(y0 + s * 50)); g.lineTo(p.sx(x0 + 10), p.sy(y0 + s * 10 * K));
        g.closePath(); g.fill();
      }
    });
    P.fn(x => y0 + K * (x - x0), { color: 'rgba(255,93,115,.7)', width: 1.2 });
    P.fn(x => y0 - K * (x - x0), { color: 'rgba(255,93,115,.7)', width: 1.2 });
    P.fn(FL, { color: '#8ea0d0', width: 2.2 });
    const bad = []; let worst = 0;
    for (let x = -3; x <= 3; x += .005) { const r = Math.abs(x - x0) > 1e-9 ? Math.abs(FL(x) - y0) / Math.abs(x - x0) : Math.abs(dFL(x0)); worst = Math.max(worst, r); bad.push([x, r > K + 1e-9 ? FL(x) : NaN]); }
    P.line(bad, { color: HEX.ball, width: 3.2 });
    P.scatter([[x0, y0]], { color: '#ffffff', r: 5 });
    P.draw();
    ctx.stat(`steepest secant from x₀ = ${worst.toFixed(3)} · K = ${K.toFixed(2)} → ${worst > K + 1e-9 ? 'rule broken at this point' : 'rule holds at this point'} · globally K-Lipschitz: ${K >= 1.9 ? 'yes' : 'no'} (needs K ≥ 1.9)`);
  },
};

// ---------- the quadratic upper bound and the step 1/L ----------
const FS = x => .4 * Math.cos(2 * x) + .1 * x * x, dFS = x => -.8 * Math.sin(2 * x) + .2 * x;
const upper = {
  title: 'The quadratic upper bound and the step η = 1/L',
  caption: 'The violet parabola f(x) + f′(x)(y − x) + (L/2)(y − x)² touches f at x. If L is at least the true smoothness constant (1.8) it stays above f everywhere, and the GD step η = 1/L jumps to its lowest point.',
  note: 'f(x) = 0.4 cos 2x + 0.1x², f″ between −1.4 and 1.8, so L = 1.8. Try L below 1.8 near x = 1.57, where f bends most sharply.',
  legend: [['#8ea0d0', 'f'], [HEX.opt2, 'upper bound'], [HEX.neg, 'bound violated'], [HEX.ball, 'GD step with η = 1/L']],
  mode: '2d',
  plot: { xlabel: 'y', ylabel: 'f', xlim: [-3, 3], ylim: [-.5, 1.5], legend: false },
  controls: [
    { id: 'x', label: 'point x', min: -2.6, max: 2.6, step: .01, value: .35 },
    { id: 'L', label: 'L', min: .6, max: 3, step: .01, value: 1.8 },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { x, L } = ctx.vals, fx = FS(x), gx = dFS(x), q = y => fx + gx * (y - x) + L / 2 * (y - x) ** 2, P = ctx.plot.clear();
    P.fn(FS, { color: '#8ea0d0', width: 2.2 });
    P.fn(q, { color: HEX.opt2, width: 1.8 });
    const viol = []; for (let y = -3; y <= 3; y += .005) viol.push([y, FS(y) > q(y) + 1e-12 ? FS(y) : NaN]);
    P.line(viol, { color: HEX.neg, width: 3.4 });
    const xn = x - gx / L;
    P.line([[x, fx], [xn, q(xn)]], { color: HEX.ball, width: 1.4, dash: [4, 3] });
    P.scatter([[xn, q(xn)]], { color: HEX.opt2, r: 4.5 });
    P.scatter([[xn, FS(xn)]], { color: HEX.ball, r: 5.5 });
    P.scatter([[x, fx]], { color: '#ffffff', r: 5 });
    P.draw();
    const guar = gx * gx / (2 * L), act = fx - FS(xn);
    ctx.stat(`step to x − f′(x)/L = ${xn.toFixed(3)} · guaranteed decrease f′²/(2L) = ${guar.toFixed(4)} · actual decrease ${act.toFixed(4)} → ${act >= guar - 1e-12 ? 'guarantee holds' : 'guarantee FAILS (L too small)'}`);
  },
};

// ---------- the O(1/sqrt(T)) guarantee vs what GD actually does ----------
const FSTAR = -0.180931, LS = 1.8;
const rate = {
  title: 'The worst-case bound vs the real run',
  caption: 'GD with η = 1/L on the same f. Coral: the smallest gradient seen in the first T steps. Violet: the guarantee √(2L(f(x₀) − f*)/T). The real run is always below, often far below.',
  note: 'Start near x₀ = 0, a local maximum where the gradient is tiny, to see a slow start. Both axes logarithmic.',
  legend: [[HEX.ball, 'min over t < T of |f′(x_t)|'], [HEX.opt2, 'bound √(2L(f(x₀) − f*)/T)']],
  mode: '2d',
  plot: { xlabel: 'T (steps)', ylabel: 'gradient size', logx: true, logy: true, xlim: [1, 1000], ylim: [1e-9, 10] },
  controls: [{ id: 'x0', label: 'start x₀', min: -2.8, max: 2.8, step: .001, value: .002, digits: 3 }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const x0 = ctx.vals.x0, P = ctx.plot.clear(), run = [], bound = [];
    let x = x0, m = Infinity, hit = null;
    for (let T = 1; T <= 1000; T++) { m = Math.min(m, Math.abs(dFS(x))); run.push([T, Math.max(m, 1e-12)]); bound.push([T, Math.sqrt(2 * LS * (FS(x0) - FSTAR) / T)]); if (hit === null && m <= 1e-3) hit = T; x = x - dFS(x) / LS; }
    P.line(bound, { color: HEX.opt2, width: 2, label: 'bound' });
    P.line(run, { color: HEX.ball, width: 2.2, label: 'GD' });
    P.hline(1e-3, { color: 'rgba(255,200,87,.6)', dash: [4, 4], text: 'ε = 10⁻³' });
    P.draw();
    const need = 2 * LS * (FS(x0) - FSTAR) / 1e-6;
    ctx.stat(`f(x₀) − f* = ${(FS(x0) - FSTAR).toFixed(4)} · bound promises |f′| ≤ 10⁻³ by T = ${Math.ceil(need).toLocaleString('en')} · GD got there at T = ${hit ?? '> 1000'}`);
  },
};

export default { parabola, bowl, lipschitz, upper, rate };
