// Stop 8 visuals. Numbers checked in scripts/check-math/course_noise.py, which ports the same RNG (mulberry32), so the
// random runs below are reproduced there number for number.
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { contours, drawContours, arrow, draggable } from '../../src/learn/plot2d.js';
import { rng } from '../../src/lib/math.js';

// ---------- the dataset: a line fit with n = 50 points ----------
const N = 50, X = Array.from({ length: N }, (_, i) => 4 * i / (N - 1)), Y = X.map((x, i) => .8 * x - .5 + .6 * Math.sin(2.7 * i + 1));
const loss = (w, b) => { let s = 0; for (let i = 0; i < N; i++) { const r = w * X[i] + b - Y[i]; s += r * r; } return s / N; };
const gradI = (w, b, i) => { const r = w * X[i] + b - Y[i]; return [2 * r * X[i], 2 * r]; };
const fullGrad = (w, b) => { let a = 0, c = 0; for (let i = 0; i < N; i++) { const g = gradI(w, b, i); a += g[0]; c += g[1]; } return [a / N, c / N]; };
const batchGrad = (w, b, B, r) => { let a = 0, c = 0; for (let k = 0; k < B; k++) { const g = gradI(w, b, Math.floor(r() * N)); a += g[0]; c += g[1]; } return [a / B, c / B]; };
const xm = X.reduce((a, v) => a + v, 0) / N, ym = Y.reduce((a, v) => a + v, 0) / N;
const W_STAR = X.reduce((a, v, i) => a + (v - xm) * (Y[i] - ym), 0) / X.reduce((a, v) => a + (v - xm) ** 2, 0), B_STAR = ym - W_STAR * xm, F_STAR = loss(W_STAR, B_STAR);
const SIG2 = (() => { let s = 0; for (let i = 0; i < N; i++) { const g = gradI(W_STAR, B_STAR, i); s += g[0] ** 2 + g[1] ** 2; } return s / N; })();   // 4.81
const L_MAX = 12.3251;   // largest Hessian eigenvalue (check script)
const P0 = [1.8, -1.2], DOM = [-.2, 2.0, -2.6, 1.6], CL = [.2, 2.2, -2.2, .6];
const fmt = v => Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4 ? v.toExponential(1) : v.toFixed(4);

// ---------- minibatch gradients: a cloud around the true gradient ----------
const cloud = {
  title: 'A minibatch gradient is a random arrow around the true one',
  caption: 'Drag the white point. The white arrow is one full gradient step (all 50 examples). Each coral dot is where a step with a random minibatch of B examples would land instead. Raise B and the cloud shrinks.',
  note: 'f(w, b) = average squared error of the line w x + b on 50 points; one step of size η = 0.08; 400 minibatches drawn with replacement.',
  legend: [['#e8ecf8', 'full-gradient step'], [HEX.ball, 'minibatch steps'], [HEX.opt2, 'their average'], [HEX.good, 'minimum']],
  mode: '2d',
  plot: { xlabel: 'slope w', ylabel: 'intercept b', xlim: [CL[0], CL[1]], ylim: [CL[2], CL[3]], equal: true, legend: false, grid: false },
  controls: [{ id: 'B', label: 'batch size B', min: 1, max: 50, step: 1, value: 5, digits: 0 }],
  build(ctx) {
    ctx.p = P0.slice();
    ctx.sets = contours(loss, [-1.5, 3.3, -3.4, 2.4], [.2, .3, .5, 1, 2, 4, 8, 16], 120);
    draggable(ctx, q => { ctx.p = [Math.max(CL[0], Math.min(CL[1], q[0])), Math.max(CL[2], Math.min(CL[3], q[1]))]; this.draw(ctx); });
    this.draw(ctx);
  },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const [w, b] = ctx.p, B = ctx.vals.B, eta = .08, g = fullGrad(w, b), r = rng(1), P = ctx.plot.clear(), pts = [];
    let mw = 0, mb = 0, v = 0;
    for (let k = 0; k < 400; k++) { const s = batchGrad(w, b, B, r); mw += s[0] / 400; mb += s[1] / 400; v += ((s[0] - g[0]) ** 2 + (s[1] - g[1]) ** 2) / 400; pts.push([w - eta * s[0], b - eta * s[1]]); }
    let s2 = 0; for (let i = 0; i < N; i++) { const gi = gradI(w, b, i); s2 += ((gi[0] - g[0]) ** 2 + (gi[1] - g[1]) ** 2) / N; }
    drawContours(P, ctx.sets);
    P.scatter(pts, { color: 'rgba(255,122,89,.45)', r: 2.2 });
    arrow(P, w, b, -eta * g[0], -eta * g[1], { color: '#e8ecf8', width: 2.2 });
    P.scatter([[w - eta * mw, b - eta * mb]], { color: HEX.opt2, r: 5 });
    P.scatter([[W_STAR, B_STAR]], { color: HEX.good, r: 5 });
    P.scatter([[w, b]], { color: '#ffffff', r: 5 });
    P.draw();
    ctx.stat(`∇f = (${g[0].toFixed(2)}, ${g[1].toFixed(2)}) · average of the 400 minibatch gradients (${mw.toFixed(2)}, ${mb.toFixed(2)}) · spread E‖g − ∇f‖²: measured ${v.toFixed(2)}, exact σ²/B = ${(s2 / B).toFixed(2)} (σ² = ${s2.toFixed(2)})`);
  },
};

// ---------- SGD's noise floor (3D) ----------
function sgdPath(eta, B, seed, T, decay) { const r = rng(seed); let [w, b] = P0; const p = [[w, b]]; for (let t = 0; t < T; t++) { const g = batchGrad(w, b, B, r), e = decay ? eta / (1 + t / 200) : eta; w -= e * g[0]; b -= e * g[1]; p.push([w, b]); } return p; }
function gdPath(eta, T) { let [w, b] = P0; const p = [[w, b]]; for (let t = 0; t < T; t++) { const g = fullGrad(w, b); w -= eta * g[0]; b -= eta * g[1]; p.push([w, b]); } return p; }
const T_FLOOR = 4000;
const floor = {
  title: 'Constant steps leave SGD rattling around the minimum',
  caption: 'SGD with a constant step (coral) gets close fast, then bounces in a cloud whose size is set by η and B: the noise floor. Shrinking the step over time (violet, ηₜ = η/(1 + t/200)) lets it settle. Full-gradient descent (white) has no noise and no floor.',
  note: 'Same 50-point line fit, start (1.8, −1.2), 4000 steps. Height on a log scale. Averages are over the last 1000 steps of one seeded run.',
  legend: [[HEX.ball, 'SGD, constant η'], [HEX.opt2, 'SGD, decaying η'], [HEX.opt3, 'GD, same η'], [HEX.good, 'minimum']],
  mode: '3d', cam: [2.8, 2.9, 2.5], tgt: [0, .2, 0],
  controls: [
    { id: 'eta', label: 'step size η', min: .002, max: .04, step: .001, value: .02, digits: 3 },
    { id: 'B', type: 'seg', label: 'batch size B', value: 1, options: [[1, '1'], [5, '5'], [25, '25']] },
  ],
  action: 'Run again',
  charts: [{ id: 'f', title: 'f(x_t) − f*, log scale', xlabel: 'step t', ylabel: 'f − f*', logy: true, xlim: [0, T_FLOOR], ylim: [1e-6, 10], h: 210, legend: false }],
  build(ctx) {
    ctx.d = def({ L: loss, g: fullGrad, dom: DOM, square: true });
    ctx.stage.surface(ctx.d);
    ctx.stage.marker(ctx.d, W_STAR, B_STAR, C.good, .09);
    this.drop(ctx);
  },
  onControl(ctx) { this.drop(ctx); },
  drop(ctx) {
    const { eta } = ctx.vals, B = +ctx.vals.B, s = ctx.stage;
    ctx.paths = { c: sgdPath(eta, B, 11, T_FLOOR, false), d: sgdPath(eta, B, 12, T_FLOOR, true), g: gdPath(eta, T_FLOOR) };
    s.clearBalls();
    ctx.balls = [['g', C.opt3], ['d', C.opt2], ['c', C.ball]].map(([k, col]) => s.ball(ctx.d, { start: P0, color: col, speed: 6, maxSteps: T_FLOOR, step: (p, ball) => ctx.paths[k][ball.steps + 1] || p }));
    const av = p => { let a = 0; for (let t = T_FLOOR - 999; t <= T_FLOOR; t++) a += loss(...p[t]) - F_STAR; return a / 1000; };
    ctx.sum = `averages over the last 1000 steps: constant η ${fmt(av(ctx.paths.c))} (small-step theory ησ²/(4B) = ${fmt(eta * SIG2 / (4 * B))}) · decaying η ${fmt(av(ctx.paths.d))} · GD ${av(ctx.paths.g) < 1e-12 ? 'below 10⁻¹²' : fmt(av(ctx.paths.g))}`;
  },
  frame(ctx) {
    if (!ctx.balls) return;
    const n = ctx.balls[0].steps, P = ctx.charts.f.clear(), every = 8;
    for (const [k, col] of [['c', HEX.ball], ['d', HEX.opt2], ['g', HEX.opt3]]) { const pts = []; for (let t = 0; t <= n; t += every) pts.push([t, Math.max(loss(...ctx.paths[k][t]) - F_STAR, 1e-12)]); P.line(pts, { color: col, width: k === 'c' ? 1.2 : 1.8 }); }
    P.hline(ctx.vals.eta * SIG2 / (4 * ctx.vals.B), { color: 'rgba(255,200,87,.7)', dash: [4, 4], text: 'ησ²/(4B)' });
    P.draw();
    ctx.stat(ctx.sum);
  },
};

// ---------- step-size schedules: Robbins–Monro ----------
const T_S = 10000, R_S = 100, X0 = 2;
const SCHED = [
  { key: 'const', label: 'constant η', color: HEX.ball, seed: 5 },
  { key: 'rm', label: 'ηₜ = 1/(t + 2)', color: HEX.good, seed: 6, fn: t => 1 / (t + 2) },
  { key: 'fast', label: 'ηₜ = 1/(t + 2)²', color: HEX.opt2, seed: 7, fn: t => 1 / (t + 2) ** 2 },
];
const tGrid = (() => { const s = new Set(); for (let k = 0; k <= 80; k++) s.add(Math.round(10 ** (4 * k / 80))); return [...s].sort((a, b) => a - b); })();
const schedules = {
  title: 'Shrinking the step: fast enough to stop the noise, slow enough to arrive',
  caption: 'Minimize f(x) = x²/2 when every gradient comes with standard normal noise, from x₀ = 2. Solid: the average of x² over 100 runs; dashed: its exact value. A constant step stalls at a floor; 1/t keeps improving; 1/t² stops moving too early.',
  note: 'Exact curves from E x²ₜ₊₁ = (1 − ηₜ)² E x²ₜ + ηₜ². Both axes logarithmic.',
  legend: SCHED.map(s => [s.color, s.label]),
  mode: '2d',
  plot: { xlabel: 'step t', ylabel: 'average x_t²', logx: true, logy: true, xlim: [1, T_S], ylim: [1e-5, 10], legend: false },
  controls: [{ id: 'eta', label: 'constant step η', min: .01, max: .5, step: .01, value: .1, live: false }],
  build(ctx) { this.run(ctx); },
  onControl(ctx) { this.run(ctx); },
  run(ctx) {
    const P = ctx.plot.clear(), at = {};
    for (const s of SCHED) {
      const fn = s.fn || (() => ctx.vals.eta), r = rng(s.seed), xs = new Float64Array(R_S).fill(X0), mc = [], ex = []; let m = X0 * X0, gi = 0;
      for (let t = 0; t < T_S; t++) {
        const e = fn(t);
        for (let k = 0; k < R_S; k++) xs[k] = xs[k] - e * (xs[k] + r.normal());
        m = (1 - e) ** 2 * m + e * e;
        if (t + 1 === tGrid[gi]) { let a = 0; for (let k = 0; k < R_S; k++) a += xs[k] * xs[k]; mc.push([t + 1, a / R_S]); ex.push([t + 1, m]); gi++; }
      }
      P.line(ex, { color: s.color, width: 1.2, dash: [4, 4], alpha: .8 });
      P.line(mc, { color: s.color, width: 2 });
      at[s.key] = [mc[mc.length - 1][1], ex[ex.length - 1][1]];
    }
    P.draw();
    const g = v => v.toPrecision(3);
    ctx.stat(`at t = 10⁴ (100 runs / exact): constant ${g(at.const[0])} / ${g(at.const[1])} · 1/(t + 2): ${g(at.rm[0])} / ${g(at.rm[1])} · 1/(t + 2)²: ${g(at.fast[0])} / ${g(at.fast[1])}`);
  },
};

// ---------- variance reduction: SVRG vs SGD vs GD, per epoch of gradient evaluations ----------
const EPOCHS = 60;
const vr = {
  title: 'Variance reduction: noise that shrinks as you arrive',
  caption: 'Cost on the horizontal axis: one epoch = 50 example gradients. SGD (coral) is cheap per step but stalls at its floor. GD (white, η = 1/L) pays 50 gradients per step. SVRG (violet) corrects each cheap gradient with a full gradient taken once per pass, so its noise vanishes near the minimum.',
  note: 'SVRG: m = 50 inner steps per full gradient (one outer loop = 3 epochs of gradient evaluations). Same η for SGD and SVRG.',
  legend: [[HEX.ball, 'SGD'], [HEX.opt2, 'SVRG'], [HEX.opt3, 'GD, η = 1/L']],
  mode: '2d',
  plot: { xlabel: 'epochs (gradient evaluations / n)', ylabel: 'f − f*', logy: true, xlim: [0, EPOCHS], ylim: [1e-14, 10], legend: false },
  controls: [{ id: 'eta', label: 'step size η (SGD, SVRG)', min: .005, max: .04, step: .001, value: .03, digits: 3 }],
  charts: [{ id: 'e', title: 'error of each gradient estimate, ‖g − ∇f‖²', xlabel: 'epochs', ylabel: '‖g − ∇f‖²', logy: true, xlim: [0, EPOCHS], ylim: [1e-24, 1e3], h: 190, legend: false }],
  build(ctx) { this.run(ctx); },
  onControl(ctx) { this.run(ctx); },
  run(ctx) {
    const eta = ctx.vals.eta, sg = [[0, loss(...P0) - F_STAR]], sv = [[0, loss(...P0) - F_STAR]], gd = [[0, loss(...P0) - F_STAR]], eS = [], eV = [];
    { const r = rng(41); let [w, b] = P0; for (let t = 1; t <= EPOCHS * N; t++) { const i = Math.floor(r() * N), g = gradI(w, b, i); if (t % 3 === 0) { const G = fullGrad(w, b); eS.push([t / N, (g[0] - G[0]) ** 2 + (g[1] - G[1]) ** 2]); } w -= eta * g[0]; b -= eta * g[1]; if (t % N === 0) sg.push([t / N, Math.max(loss(w, b) - F_STAR, 1e-16)]); } }
    { const r = rng(42); let [w, b] = P0, ev = 0;
      for (let o = 0; o < EPOCHS / 3; o++) {
        const sw = w, sb = b, mu = fullGrad(w, b); ev += N;
        for (let k = 0; k < N; k++) { const i = Math.floor(r() * N), g = gradI(w, b, i), g0 = gradI(sw, sb, i), est = [g[0] - g0[0] + mu[0], g[1] - g0[1] + mu[1]]; ev += 2; { const G = fullGrad(w, b), e2 = (est[0] - G[0]) ** 2 + (est[1] - G[1]) ** 2; if (e2 > 0) eV.push([ev / N, e2]); } w -= eta * est[0]; b -= eta * est[1]; }
        sv.push([ev / N, Math.max(loss(w, b) - F_STAR, 1e-16)]);
      } }
    { let [w, b] = P0; for (let t = 1; t <= EPOCHS; t++) { const g = fullGrad(w, b); w -= g[0] / L_MAX; b -= g[1] / L_MAX; gd.push([t, Math.max(loss(w, b) - F_STAR, 1e-16)]); } }
    const P = ctx.plot.clear();
    P.line(sg, { color: HEX.ball, width: 1.8 }); P.line(gd, { color: HEX.opt3, width: 1.8 }); P.line(sv, { color: HEX.opt2, width: 2.2, dots: 2.5 });
    P.draw();
    ctx.charts.e.clear().scatter(eS, { color: 'rgba(255,122,89,.55)', r: 1.3 }).scatter(eV, { color: HEX.opt2, r: 1.5 }).draw();
    const last = a => a[a.length - 1][1], show = v => v <= 1e-15 ? '< 10⁻¹⁵' : v.toExponential(1);
    ctx.stat(`after ${EPOCHS} epochs of gradients: SGD ${show(last(sg))} · GD ${show(last(gd))} · SVRG ${show(last(sv))}`);
  },
};

// ---------- noise at a saddle (3D) ----------
const fS = (x, y) => .5 * x * x + .25 * y ** 4 - .5 * y * y, gS = (x, y) => [x, y ** 3 - y];
const S_START = [1.5, 0], ETA_E = .1, T_E = 600;
function escPath(mode, s, seed) {
  const r = rng(seed); let [x, y] = S_START; const p = [[x, y]]; let out = null;
  for (let t = 0; t < T_E; t++) {
    let [gx, gy] = gS(x, y);
    if (mode !== 'gd') { const n = r() < .5 ? s : -s; if (mode === 'y') gy += n; else gx += n; }
    x -= ETA_E * gx; y -= ETA_E * gy; p.push([x, y]);
    if (out === null && Math.abs(y) > .5) out = t + 1;
  }
  return { p, out };
}
const escape = {
  title: 'Noise kicks the ball off a saddle, if it points the right way',
  caption: 'f(x, y) = x²/2 + y⁴/4 − y²/2 has a saddle at the origin. Both balls start at (1.5, 0), exactly on the line that leads into the saddle. Each SGD gradient adds a random ±s (two training examples, f ± s·y or f ± s·x). GD (white) slides into the saddle and stays.',
  note: 'Step η = 0.1, 600 steps. Red: the direction of negative curvature (y), blue: positive (x). Escaped = |y| > 0.5. "Drop again" uses a new random seed.',
  legend: [[HEX.ball, 'SGD'], [HEX.opt3, 'GD'], [HEX.neg, 'negative curvature'], [HEX.good, 'minima']],
  mode: '3d', cam: [3.0, 2.7, 2.8], tgt: [0, .25, 0],
  controls: [
    { id: 'dir', type: 'seg', label: 'noise points along', value: 'y', options: [['y', 'y (the escape direction)'], ['x', 'x only']] },
    { id: 's', label: 'noise size s', min: .05, max: 1, step: .01, value: .3 },
  ],
  actions: [
    { label: 'Drop again', fn: ctx => { ctx.seed++; escape.drop(ctx); } },
    { label: 'Reset seed', fn: ctx => { ctx.seed = 31; escape.drop(ctx); } },
  ],
  build(ctx) {
    ctx.seed = 31;
    ctx.d = def({ L: fS, g: gS, dom: [-1.8, 1.8, -1.6, 1.6], h: v => v, fit: 1.3 });
    ctx.stage.surface(ctx.d);
    ctx.stage.eigenLines(ctx.d, 0, 0, { len: .9 });
    ctx.stage.marker(ctx.d, 0, 1, C.good, .1); ctx.stage.marker(ctx.d, 0, -1, C.good, .1);
    this.drop(ctx);
  },
  onControl(ctx) { this.drop(ctx); },
  drop(ctx) {
    const s = ctx.stage; s.clearBalls();
    const sg = escPath(ctx.vals.dir, ctx.vals.s, ctx.seed), gd = escPath('gd', 0, 1);
    s.ball(ctx.d, { start: S_START, color: C.opt3, speed: 2, maxSteps: T_E, step: (p, b) => gd.p[b.steps + 1] || p });
    s.ball(ctx.d, { start: S_START, color: C.ball, speed: 2, maxSteps: T_E, step: (p, b) => sg.p[b.steps + 1] || p });
    const end = sg.p[T_E];
    ctx.stat(`seed ${ctx.seed} · SGD: ${sg.out ? `escaped after ${sg.out} steps, ends near (${end[0].toFixed(2)}, ${end[1].toFixed(2)})` : `after ${T_E} steps still on the line y = 0 (y = ${end[1].toFixed(2)}), at x = ${end[0].toFixed(3)}`} · GD: x = ${gd.p[T_E][0].toExponential(1)}, y = 0, stuck at the saddle`);
  },
};

export default { cloud, floor, schedules, vr, escape };
