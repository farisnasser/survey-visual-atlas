// Stop 7 visuals. Numbers checked in scripts/check-math/course_memory.py.
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { contours, drawContours } from '../../src/learn/plot2d.js';

// colours for the two curvature directions of the valley: flattest (amber) → sharpest (blue)
const mix = t => `rgb(${[[255, 124], [200, 196], [87, 255]].map(([a, b]) => Math.round(a + (b - a) * t)).join(',')})`;
const stepsTo = rho => rho < 1 ? Math.round(Math.log(1e6) / -Math.log(rho)) : Infinity;

// ---------- momentum is an average of past gradients ----------
const echo = {
  title: 'Momentum adds up the past',
  caption: 'A gradient that has a steady part (1) and a part that flips sign every step (a zig-zag of size a). Momentum adds all past gradients with weights 1, β, β², …: the steady part piles up, the flipping part cancels.',
  note: 'Plotted: the gradients g_t (dots) and the momentum sum s_t = g_t + β g_{t−1} + β² g_{t−2} + …; the step is −η s_t.',
  legend: [['#e8ecf8', 'gradient g_t'], [HEX.ball, 'momentum sum s_t'], [HEX.flat, 'steady level 1/(1 − β)']],
  mode: '2d',
  plot: { xlabel: 'step t', ylabel: 'value', xlim: [0, 60], legend: false },
  controls: [
    { id: 'beta', label: 'momentum β', min: 0, max: .95, step: .01, value: .9 },
    { id: 'a', label: 'zig-zag size a', min: 0, max: 3, step: .1, value: 1.5, digits: 1 },
  ],
  charts: [{ id: 'w', title: 'weight of the gradient from k steps ago: βᵏ', xlabel: 'k (steps ago)', ylabel: 'βᵏ', ylim: [0, 1.05], h: 190, legend: false }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { beta, a } = ctx.vals, P = ctx.plot.clear(), g = [], s = [];
    let v = 0;
    for (let t = 0; t <= 60; t++) { const gt = 1 + a * (t % 2 ? -1 : 1); v = beta * v + gt; g.push([t, gt]); s.push([t, v]); }
    P.hline(1 / (1 - beta), { color: HEX.flat, dash: [5, 4], width: 1.2 });
    P.hline(0, { color: 'rgba(232,236,248,.25)', width: 1 });
    P.line(g, { color: 'rgba(232,236,248,.35)', width: 1 });
    P.scatter(g, { color: '#e8ecf8', r: 2.6 });
    P.line(s, { color: HEX.ball, width: 2.4 });
    P.draw();
    const W = ctx.charts.w.clear(), half = Math.log(.5) / Math.log(beta || 1e-9);
    W.bars(Array.from({ length: 30 }, (_, k) => ({ y: beta ** k, color: k < half ? HEX.ball : 'rgba(255,122,89,.35)' })));
    W.draw();
    ctx.stat(`steady part multiplied by 1/(1 − β) = ${(1 / (1 - beta)).toFixed(2)} · flipping part by 1/(1 + β) = ${(1 / (1 + beta)).toFixed(3)} · memory ≈ last 1/(1 − β) = ${(1 / (1 - beta)).toFixed(1)} steps (bright bars: half the total weight)`);
  },
};

// ---------- GD vs heavy ball vs Nesterov in the valley ----------
const LC = 10, START = [-1.45, .5], DOM = [-1.6, 1.6, -.7, .7];
const polyak = mu => [4 / (Math.sqrt(LC) + Math.sqrt(mu)) ** 2, ((Math.sqrt(LC) - Math.sqrt(mu)) / (Math.sqrt(LC) + Math.sqrt(mu))) ** 2];
const nest = mu => [1 / LC, (Math.sqrt(LC / mu) - 1) / (Math.sqrt(LC / mu) + 1)];
/** pure simulation, the same updates as the check script: returns the f values and the iterates */
function simulate(kind, mu, eta, beta, T) {
  const f = (x, y) => .5 * (mu * x * x + LC * y * y);
  let z = START.slice(), zp = START.slice(); const fs = [f(...z)], ps = [z.slice()];
  for (let t = 0; t < T; t++) {
    let n;
    if (kind === 'gd') n = [z[0] - eta * mu * z[0], z[1] - eta * LC * z[1]];
    else if (kind === 'hb') n = [z[0] - eta * mu * z[0] + beta * (z[0] - zp[0]), z[1] - eta * LC * z[1] + beta * (z[1] - zp[1])];
    else { const y = [z[0] + beta * (z[0] - zp[0]), z[1] + beta * (z[1] - zp[1])]; n = [y[0] - eta * mu * y[0], y[1] - eta * LC * y[1]]; }
    zp = z; z = n; fs.push(f(...z)); ps.push(z.slice());
    if (!isFinite(fs[fs.length - 1]) || fs[fs.length - 1] > 1e6) break;
  }
  return { fs, ps, hit: fs.findIndex(v => v < 1e-8), div: fs[fs.length - 1] > 1e6 || !isFinite(fs[fs.length - 1]) };
}
const valley = {
  title: 'Memory in a narrow valley',
  caption: 'f(x, y) = ½(μx² + 10y²). Gradient descent with its best step (violet) zig-zags; the heavy ball (coral) and Nesterov (white) use the same η and β. Try both tunings, then move the sliders.',
  note: 'κ = 10/μ. All three start at (−1.45, 0.5). Heights rescaled to fit; the y-axis is stretched so the valley is visible. Counts are exact runs until f < 10⁻⁸.',
  legend: [[HEX.opt2, 'GD, step 2/(L + μ)'], [HEX.ball, 'heavy ball'], [HEX.opt3, 'Nesterov'], [HEX.good, 'minimum']],
  mode: '3d', cam: [3.5, 2.8, 1.8], tgt: [-.4, .25, 0],
  controls: [
    { id: 'k', type: 'seg', label: 'condition number κ', value: 100, options: [[10, '10'], [100, '100']] },
    { id: 'eta', label: 'step size η', min: .01, max: .4, step: .001, value: .1, digits: 3 },
    { id: 'beta', label: 'momentum β', min: 0, max: .99, step: .001, value: .818, digits: 3 },
  ],
  actions: [
    { label: 'Drop the balls', fn: ctx => valley.drop(ctx) },
    { label: 'Polyak’s tuning', fn: ctx => { const [e, b] = polyak(LC / ctx.vals.k); ctx.set('eta', e); ctx.set('beta', b); valley.drop(ctx); } },
    { label: 'Nesterov’s tuning', fn: ctx => { const [e, b] = nest(LC / ctx.vals.k); ctx.set('eta', e); ctx.set('beta', b); valley.drop(ctx); } },
  ],
  charts: [{ id: 'f', title: 'f(x_t), log scale', xlabel: 'step t', ylabel: 'f', logy: true, ylim: [1e-9, 10], xlim: [0, 500], h: 210, legend: false }],
  build(ctx) {
    const mu = LC / ctx.vals.k;
    ctx.d = def({ L: (x, y) => .5 * (mu * x * x + LC * y * y), g: (x, y) => [mu * x, LC * y], dom: DOM, square: true, h: v => v, fit: 1 });
    ctx.stage.surface(ctx.d);
    ctx.stage.marker(ctx.d, 0, 0, C.good, .1);
    this.drop(ctx);
  },
  onControl(ctx, id) { if (id === 'k') { ctx.stage.clear(); this.build(ctx); } else this.drop(ctx); },
  drop(ctx) {
    const s = ctx.stage, d = ctx.d, mu = LC / ctx.vals.k, { eta, beta } = ctx.vals, T = ctx.vals.k === 100 ? 600 : 120;
    s.clearBalls();
    ctx.sims = { gd: simulate('gd', mu, 2 / (LC + mu), 0, T), hb: simulate('hb', mu, eta, beta, T), nag: simulate('nag', mu, eta, beta, T) };
    const out = p => !d.in(p[0], p[1]);
    ctx.balls = Object.entries({ gd: C.opt2, hb: C.ball, nag: C.opt3 }).map(([k, col]) => {
      const sim = ctx.sims[k], last = sim.hit >= 0 ? sim.hit : sim.ps.length - 1;
      const b = s.ball(d, { start: START, color: col, speed: ctx.vals.k === 100 ? 1.5 : .3, maxSteps: last, step: (p, ball) => sim.ps[ball.steps + 1] || p, stop: ball => out(sim.ps[ball.steps + 1] || [0, 0]) });
      b.key = k; return b;
    });
  },
  frame(ctx) {
    if (!ctx.balls) return;
    const n = Math.max(...ctx.balls.map(b => b.steps)), S = ctx.sims, xmax = ctx.vals.k === 100 ? 500 : 60;
    const P = ctx.charts.f.clear().set({ xlim: [0, xmax] });
    for (const [k, col] of [['gd', HEX.opt2], ['hb', HEX.ball], ['nag', HEX.opt3]]) P.line(S[k].fs.slice(0, n + 1).map((v, t) => [t, Math.max(v, 1e-12)]), { color: col, width: k === 'gd' ? 1.6 : 2 });
    P.hline(1e-8, { color: 'rgba(255,200,87,.5)', dash: [4, 4], text: '10⁻⁸' });
    P.draw();
    const say = s => s.div ? 'diverges' : s.hit >= 0 ? `${s.hit}` : `> ${s.fs.length - 1}`;
    ctx.stat(`κ = ${ctx.vals.k} · η = ${ctx.vals.eta.toFixed(3)}, β = ${ctx.vals.beta.toFixed(3)} · steps to f < 10⁻⁸: GD ${say(S.gd)} · heavy ball ${say(S.hb)} · Nesterov ${say(S.nag)}`);
  },
};

// ---------- the roots of the 2×2 iteration matrix ----------
function roots(kind, lam, eta, beta) {   // roots of the characteristic polynomial along one curvature λ, as [re, im]
  if (kind === 'gd') return [[1 - eta * lam, 0]];
  const b = kind === 'hb' ? 1 + beta - eta * lam : (1 + beta) * (1 - eta * lam), c = kind === 'hb' ? beta : beta * (1 - eta * lam);
  const D = b * b - 4 * c;
  return D >= 0 ? [[(b + Math.sqrt(D)) / 2, 0], [(b - Math.sqrt(D)) / 2, 0]] : [[b / 2, Math.sqrt(-D) / 2], [b / 2, -Math.sqrt(-D) / 2]];
}
function errorPath(kind, lam, eta, beta, T) {   // |u_t| along one curvature from u_0 = u_{-1} = 1
  let u = 1, up = 1; const out = [[0, 1]];
  for (let t = 1; t <= T; t++) {
    let n;
    if (kind === 'gd') n = (1 - eta * lam) * u;
    else if (kind === 'hb') n = (1 + beta - eta * lam) * u - beta * up;
    else n = (1 + beta) * (1 - eta * lam) * u - beta * (1 - eta * lam) * up;
    up = u; u = n; out.push([t, Math.max(Math.abs(u), 1e-14)]);
    if (Math.abs(u) > 1e6) break;
  }
  return out;
}
const rootsViz = {
  title: 'Every curvature becomes a pair of points',
  caption: 'Each dot is a root r of the iteration along one curvature λ between μ (amber) and L = 10 (blue). Along that direction the error is multiplied by |r| per step, so everything must sit inside the unit circle, and the farthest dot sets the speed.',
  note: 'Heavy ball: r² − (1 + β − ηλ) r + β = 0. Complex roots lie on the dashed circle of radius √β. Side chart: the error along the flattest and the sharpest direction.',
  legend: [[HEX.flat, 'λ = μ (flattest)'], [HEX.pos, 'λ = L (sharpest)'], [HEX.flat + '99', 'radius √β'], ['#e8ecf8', 'unit circle']],
  mode: '2d',
  plot: { xlabel: 'real part of r', ylabel: 'imaginary part', xlim: [-1.15, 1.15], ylim: [-1.15, 1.15], equal: true, legend: false },
  controls: [
    { id: 'm', type: 'seg', label: 'method', value: 'hb', options: [['gd', 'GD'], ['hb', 'heavy ball'], ['nag', 'Nesterov']] },
    { id: 'k', type: 'seg', label: 'κ', value: 100, options: [[10, '10'], [100, '100']] },
    { id: 'eta', label: 'step size η', min: .005, max: .4, step: .001, value: .1, digits: 3 },
    { id: 'beta', label: 'momentum β', min: 0, max: .99, step: .001, value: .5, digits: 3 },
  ],
  actions: [
    { label: 'Polyak’s tuning', fn: ctx => { const [e, b] = polyak(LC / ctx.vals.k); ctx.set('eta', e); ctx.set('beta', b); rootsViz.draw(ctx); } },
    { label: 'Nesterov’s tuning', fn: ctx => { const [e, b] = nest(LC / ctx.vals.k); ctx.set('eta', e); ctx.set('beta', b); rootsViz.draw(ctx); } },
  ],
  charts: [{ id: 'u', title: 'error |u_t| along λ = μ and λ = L', xlabel: 'step t', ylabel: '|u_t|', logy: true, xlim: [0, 150], ylim: [1e-8, 100], h: 200, legend: false }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { m, k, eta, beta } = ctx.vals, mu = LC / k, P = ctx.plot.clear(), lams = Array.from({ length: 121 }, (_, i) => mu + (LC - mu) * i / 120);
    P.custom((g, p) => {
      const circ = (r, col, dash) => { g.strokeStyle = col; g.setLineDash(dash); g.lineWidth = 1.3; g.beginPath(); g.arc(p.sx(0), p.sy(0), Math.abs(p.sx(r) - p.sx(0)), 0, 2 * Math.PI); g.stroke(); g.setLineDash([]); };
      circ(1, 'rgba(232,236,248,.6)', []);
      if (m === 'hb') circ(Math.sqrt(beta), 'rgba(255,200,87,.6)', [5, 4]);
      g.strokeStyle = 'rgba(232,236,248,.15)'; g.beginPath(); g.moveTo(p.sx(-1.2), p.sy(0)); g.lineTo(p.sx(1.2), p.sy(0)); g.moveTo(p.sx(0), p.sy(-1.2)); g.lineTo(p.sx(0), p.sy(1.2)); g.stroke();
    });
    let worst = 0;
    lams.forEach((lam, i) => {
      const rs = roots(m, lam, eta, beta), t = i / 120, end = i === 0 || i === 120;
      rs.forEach(r => worst = Math.max(worst, Math.hypot(r[0], r[1])));
      P.scatter(rs.map(r => [r[0], r[1]]), { color: mix(t), r: end ? 5.5 : 2.2 });
    });
    P.draw();
    const U = ctx.charts.u.clear();
    U.line(errorPath(m, mu, eta, beta, 150), { color: HEX.flat, width: 2 });
    U.line(errorPath(m, LC, eta, beta, 150), { color: HEX.pos, width: 2 });
    U.draw();
    const gdBest = (k - 1) / (k + 1), n = stepsTo(worst);
    ctx.stat(`largest |r| = ${worst.toFixed(3)} → ${worst < 1 ? `the error shrinks by 10⁶ in about ${n} steps` : 'diverges along some direction'} · for comparison GD with its best step: ${gdBest.toFixed(3)} (${stepsTo(gdBest)} steps)`);
  },
};

// ---------- overshoot on the stop-1 curve ----------
const F = t => .1 * t * t + .5 * Math.sin(3 * t), dF = t => .2 * t + 1.5 * Math.cos(3 * t);
const MINIMA = [-2.5045, -0.5013, 1.50352, 3.5032];   // the last one is just outside stop 1's window
const ETA_O = .05;
function rollPath(t0, beta, T = 3000) { let x = t0, xp = t0; const p = [x]; for (let k = 0; k < T; k++) { const n = x - ETA_O * dF(x) + beta * (x - xp); xp = x; x = n; p.push(x); } return p; }
const overshoot = {
  title: 'Momentum overshoots: sometimes into a better valley, sometimes out of one',
  caption: 'The loss from stop 1, drawn a little wider. A ball with momentum β rolls from θ₀ with step η = 0.05. With β = 0 it is gradient descent and stops in the valley below its start.',
  note: '𝓛(θ) = 0.1θ² + 0.5 sin 3θ. The side chart shows that the loss is not monotone: momentum methods are not descent methods.',
  legend: [[HEX.good, 'global minimum (in this window)'], ['#e8ecf8', 'other local minima'], [HEX.ball, 'the path']],
  mode: '2d',
  plot: { xlabel: 'θ', ylabel: '𝓛(θ)', xlim: [-3.3, 3.9], ylim: [-.65, 1.9], legend: false },
  controls: [
    { id: 't0', label: 'start θ₀', min: -3.2, max: 3.8, step: .01, value: 2.4 },
    { id: 'beta', label: 'momentum β', min: 0, max: .99, step: .01, value: .95 },
  ],
  actions: [{ label: 'Roll', fn: ctx => { ctx.k = 0; ctx.run = true; } }],
  charts: [{ id: 'f', title: 'loss along the path', xlabel: 'step t', ylabel: '𝓛(θ_t)', xlim: [0, 300], ylim: [-.6, 1.6], h: 190, legend: false }],
  build(ctx) { this.reset(ctx); },
  onControl(ctx) { this.reset(ctx); },
  reset(ctx) { ctx.path = rollPath(ctx.vals.t0, ctx.vals.beta); ctx.k = 0; ctx.run = true; this.draw(ctx); },
  frame(ctx) { if (ctx.run) { ctx.k = Math.min(ctx.k + 3, 300); if (ctx.k >= 300) ctx.run = false; this.draw(ctx); } },
  draw(ctx) {
    const P = ctx.plot.clear(), k = ctx.k, path = ctx.path;
    P.fn(F, { color: '#8ea0d0', width: 2 });
    MINIMA.forEach((m, i) => P.scatter([[m, F(m)]], { color: i === 1 ? HEX.good : '#e8ecf8', r: 6 }));
    const shown = path.slice(0, k + 1).map(t => [t, F(t)]);
    P.line(shown, { color: HEX.ball, width: 1.4, alpha: .6, dots: 2.2 });
    P.scatter([shown[shown.length - 1]], { color: HEX.ball, r: 7 });
    P.draw();
    ctx.charts.f.clear().line(path.slice(0, k + 1).map((t, i) => [i, F(t)]), { color: HEX.ball, width: 1.8 }).draw();
    let ups = 0; for (let i = 1; i <= 200; i++) if (F(path[i]) > F(path[i - 1]) + 1e-12) ups++;
    const end = path[path.length - 1], m = MINIMA.reduce((a, v) => Math.abs(v - end) < Math.abs(a - end) ? v : a);
    ctx.stat(`after 3000 steps θ = ${end.toFixed(3)}: ${m === MINIMA[1] ? 'the global minimum' : 'a local minimum'}, loss ${F(m).toFixed(3)} · the loss went up on ${ups} of the first 200 steps`);
  },
};

// ---------- Lookahead wrapping the heavy ball ----------
const MU_LA = .1, ETA_LA = .1, fV = (x, y) => .5 * (MU_LA * x * x + LC * y * y);
function hbRun(beta, T) { let z = START.slice(), zp = z.slice(); const p = [z.slice()]; for (let t = 0; t < T; t++) { const n = [z[0] - ETA_LA * MU_LA * z[0] + beta * (z[0] - zp[0]), z[1] - ETA_LA * LC * z[1] + beta * (z[1] - zp[1])]; zp = z; z = n; p.push(z.slice()); } return p; }
function laRun(beta, k, alpha, T) {   // velocity kept across syncs (the paper's default)
  let phi = START.slice(), z = phi.slice(), zp = z.slice(), c = 0; const cycles = [[z.slice()]], slow = [phi.slice()];
  for (let t = 0; t < T; t++) {
    const n = [z[0] - ETA_LA * MU_LA * z[0] + beta * (z[0] - zp[0]), z[1] - ETA_LA * LC * z[1] + beta * (z[1] - zp[1])]; zp = z; z = n; c++;
    cycles[cycles.length - 1].push(z.slice());
    if (c === k) { const v = [z[0] - zp[0], z[1] - zp[1]]; phi = [phi[0] + alpha * (z[0] - phi[0]), phi[1] + alpha * (z[1] - phi[1])]; z = phi.slice(); zp = [z[0] - v[0], z[1] - v[1]]; c = 0; cycles.push([z.slice()]); }
    slow.push(phi.slice());
  }
  return { cycles, slow };
}
const lookahead = {
  title: 'Lookahead: k fast steps, then one careful step',
  caption: 'An over-eager heavy ball (coral, last 60 steps shown) swings back and forth across the valley. Lookahead runs the same heavy ball for k steps (faint white: the current inner run), then moves its slow weights (white dots) only a fraction α of the way to where the fast weights ended.',
  note: 'f = ½(0.1x² + 10y²), η = 0.1, both start at (−1.45, 0.5). Momentum is kept across syncs, as in the paper’s experiments. The slow weights are what Lookahead reports.',
  legend: [[HEX.ball, 'heavy ball alone'], ['rgba(242,244,255,.4)', 'fast weights'], [HEX.opt3, 'slow weights'], [HEX.good, 'minimum']],
  mode: '2d',
  plot: { xlabel: 'x', ylabel: 'y', xlim: [-1.6, 1.6], ylim: [-.75, .75], equal: true, legend: false, grid: false, yfmt: v => Math.abs(v) < 1e-9 ? '0' : String(+v.toFixed(2)) },
  controls: [
    { id: 'beta', label: 'momentum β', min: .5, max: .995, step: .001, value: .99, digits: 3 },
    { id: 'k', label: 'fast steps k', min: 2, max: 40, step: 1, value: 20, digits: 0 },
    { id: 'alpha', label: 'slow step α', min: .1, max: 1, step: .05, value: .5 },
  ],
  actions: [{ label: 'Replay', fn: ctx => { ctx.n = 0; } }],
  charts: [{ id: 'f', title: 'f, log scale', xlabel: 'step t', ylabel: 'f', logy: true, xlim: [0, 1500], ylim: [1e-9, 10], h: 200, legend: false }],
  build(ctx) { ctx.sets = contours(fV, [-1.9, 1.9, -.9, .9], [.002, .01, .04, .1, .25, .5, 1, 1.6, 2.4], 120); this.run(ctx); },
  onControl(ctx) { this.run(ctx); },
  run(ctx) {
    const { beta, k, alpha } = ctx.vals, T = 1500;
    ctx.hb = hbRun(beta, T); ctx.la = laRun(beta, k, alpha, T); ctx.n = 0;
    const hit = ps => { const i = ps.findIndex(p => fV(...p) < 1e-8); return i < 0 ? `> ${T}` : i; };
    // per-step contraction of each method along the valley's two curvatures (λ = 0.1 and 10)
    const rHB = Math.max(...[MU_LA, LC].map(l => Math.max(...roots('hb', l, ETA_LA, beta).map(r => Math.hypot(...r)))));
    const rLA = Math.max(...[MU_LA, LC].map(l => laRate(l, beta, k, alpha)));
    ctx.summary = `steps to f < 10⁻⁸: heavy ball ${hit(ctx.hb)} · Lookahead ${hit(ctx.la.slow)} · per-step contraction ${rHB.toFixed(4)} vs ${rLA.toFixed(4)}`;
    this.draw(ctx);
  },
  frame(ctx) { if (ctx.n < 1500) { ctx.n = Math.min(1500, ctx.n + 6); this.draw(ctx); } },
  draw(ctx) {
    const P = ctx.plot.clear(), n = ctx.n, k = ctx.vals.k;
    drawContours(P, ctx.sets);
    const cyc = Math.floor(n / k), inner = ctx.la.cycles[cyc] || [];
    P.line(ctx.hb.slice(Math.max(0, n - 60), n + 1), { color: HEX.ball, width: 1.3, alpha: .75 });
    P.line(inner.slice(0, n - cyc * k + 1), { color: 'rgba(242,244,255,.45)', width: 1.2 });
    const syncs = []; for (let t = 0; t <= n; t += k) syncs.push(ctx.la.slow[t]);
    P.line(syncs, { color: HEX.opt3, width: 1.6 });
    P.scatter(syncs, { color: HEX.opt3, r: 3.2 });
    P.scatter([[0, 0]], { color: HEX.good, r: 5 });
    P.scatter([ctx.hb[n]], { color: HEX.ball, r: 5 });
    P.draw();
    ctx.charts.f.clear()
      .line(ctx.hb.slice(0, n + 1).map((p, t) => [t, Math.max(fV(...p), 1e-12)]), { color: HEX.ball, width: 1.6 })
      .line(ctx.la.slow.slice(0, n + 1).map((p, t) => [t, Math.max(fV(...p), 1e-12)]), { color: HEX.opt3, width: 1.8 })
      .hline(1e-8, { color: 'rgba(255,200,87,.5)', dash: [4, 4] }).draw();
    ctx.stat(ctx.summary);
  },
};
function laRate(lam, beta, k, alpha) {   // spectral radius of the (φ, v) cycle map, per step
  const a = 1 + beta - ETA_LA * lam, step = ([u, w]) => [a * u - beta * w, u];
  const cols = [[1, 0], [0, 1]].map(([phi, v]) => { let s = [phi, phi - v]; for (let i = 0; i < k; i++) s = step(s); return [phi + alpha * (s[0] - phi), s[0] - s[1]]; });
  const [[p, r], [q, s]] = cols, tr = p + s, det = p * s - q * r, D = tr * tr - 4 * det;
  const rad = D >= 0 ? Math.max(Math.abs((tr + Math.sqrt(D)) / 2), Math.abs((tr - Math.sqrt(D)) / 2)) : Math.sqrt(det);
  return rad ** (1 / k);
}

export default { echo, valley, roots: rootsViz, overshoot, lookahead };
