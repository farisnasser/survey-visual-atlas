// Stop 9 visuals. Numbers checked in scripts/check-math/course_adaptive.py (which ports the RNG used by the Adam toy).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { contours, drawContours } from '../../src/learn/plot2d.js';
import { rng } from '../../src/lib/math.js';

const LC = 10;
/** Hessian of a valley with curvatures L/kappa and L, rotated by deg degrees: [a, b, c] = [[a, b], [b, c]] */
function hrot(kap, deg) {
  const t = deg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t), mu = LC / kap;
  return [mu * c * c + LC * s * s, (mu - LC) * c * s, mu * s * s + LC * c * c];
}
const quadF = H => (x, y) => .5 * (H[0] * x * x + 2 * H[1] * x * y + H[2] * y * y);
const quadG = H => (x, y) => [H[0] * x + H[1] * y, H[1] * x + H[2] * y];
const eig2s = (a, b, c) => { const m = (a + c) / 2, r = Math.hypot((a - c) / 2, b); return [m - r, m + r]; };

// ---------- preconditioning: diagonal vs Newton ----------
const P_START = [-1.3, 1.1];
function precondPath(H, P, eta, T) {   // x <- x - eta P grad f; P = [p00, p01, p11]
  const g = quadG(H), f = quadF(H); let [x, y] = P_START; const p = [[x, y]]; let hit = -1;
  for (let t = 0; t < T; t++) {
    if (f(x, y) < 1e-8) { hit = t; break; }
    const [gx, gy] = g(x, y); x -= eta * (P[0] * gx + P[1] * gy); y -= eta * (P[1] * gx + P[2] * gy); p.push([x, y]);
  }
  return { p, hit };
}
/** eigenvalues of P H for symmetric 2x2 P, H (real, positive here) */
const evPH = (P, H) => { const a = P[0] * H[0] + P[1] * H[1], b = P[0] * H[1] + P[1] * H[2], c = P[1] * H[0] + P[2] * H[1], d = P[1] * H[1] + P[2] * H[2], tr = a + d, det = a * d - b * c, D = Math.sqrt(Math.max(0, tr * tr / 4 - det)); return [tr / 2 - D, tr / 2 + D]; };
const precond = {
  title: 'A step size per coordinate fixes axis-aligned valleys, not rotated ones',
  caption: 'Rotate the valley. GD (violet) zig-zags at any angle. Dividing each coordinate by its own curvature (coral, the diagonal preconditioner) makes an axis-aligned valley perfectly round, but helps less and less as the valley turns, and not at all at 45°. Newton (white, the full inverse Hessian) jumps to the minimum in one step at every angle.',
  note: 'f = ½ xᵀHx with curvatures L/κ and L = 10 along rotated axes; start (−1.3, 1.1); each method uses its best fixed step. Paths show the first 30 steps; counts are steps to f < 10⁻⁸.',
  legend: [[HEX.opt2, 'GD'], [HEX.ball, 'diagonal preconditioner diag(H)⁻¹'], [HEX.opt3, 'Newton H⁻¹'], [HEX.good, 'minimum']],
  mode: '2d',
  plot: { xlabel: 'x₁', ylabel: 'x₂', xlim: [-1.6, 1.6], ylim: [-1.6, 1.6], equal: true, legend: false, grid: false },
  controls: [
    { id: 'deg', label: 'valley angle (degrees)', min: 0, max: 90, step: 1, value: 30, digits: 0 },
    { id: 'k', type: 'seg', label: 'κ', value: 100, options: [[10, '10'], [100, '100']] },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { deg, k } = ctx.vals, H = hrot(+k, deg), P = ctx.plot.clear();
    const lv = [.02, .08, .2, .4, .7, 1.1, 1.6, 2.3, 3.2, 4.4, 6, 8].map(v => v * (k == 100 ? 1 : .5));
    drawContours(P, contours(quadF(H), [-4, 4, -2.2, 2.2], lv, 120));
    const run = Pm => { const [a, b] = evPH(Pm, H); return precondPath(H, Pm, 2 / (a + b), 3000); };
    const gd = run([1, 0, 1]), dg = run([1 / H[0], 0, 1 / H[2]]);
    const det = H[0] * H[2] - H[1] * H[1], nt = precondPath(H, [H[2] / det, -H[1] / det, H[0] / det], 1, 5);
    P.line(gd.p.slice(0, 30), { color: HEX.opt2, width: 1, alpha: .75 });
    P.line(dg.p.slice(0, 30), { color: HEX.ball, width: 1.3, alpha: .9 });
    P.line(nt.p, { color: HEX.opt3, width: 2, dash: [5, 4] });
    P.scatter([[0, 0]], { color: HEX.good, r: 5 }); P.scatter([P_START], { color: '#ffffff', r: 5 });
    P.draw();
    const [a, b] = evPH([1 / H[0], 0, 1 / H[2]], H), say = r => r.hit < 0 ? '> 3000' : r.hit;
    ctx.stat(`κ = ${k}, angle ${deg}° · after the diagonal rescaling the valley has condition number ${(b / a).toFixed(2)} · steps to f < 10⁻⁸: GD ${say(gd)}, diagonal ${say(dg)}, Newton ${nt.hit}`);
  },
};

// ---------- Adam's bias correction ----------
const bias = {
  title: 'Why Adam divides by 1 − βᵗ',
  caption: 'Feed Adam the same gradient every step. Its two averages start at zero, so early on they are too small, and by different amounts. Without correction the step (coral) starts at 3.16α and overshoots to 6.57α for β₂ = 0.999. With bias correction (green) the step is exactly α from the first step on.',
  note: 'Constant gradient, β₁ = 0.9. Uncorrected step / α = (1 − β₁ᵗ)/√(1 − β₂ᵗ). Horizontal axis logarithmic.',
  legend: [[HEX.ball, 'without bias correction'], [HEX.good, 'with bias correction']],
  mode: '2d',
  plot: { xlabel: 'step t', ylabel: 'step size / α', logx: true, xlim: [1, 1e5], ylim: [0, 22], legend: false },
  controls: [{ id: 'b2', type: 'seg', label: 'β₂', value: .999, options: [[.99, '0.99'], [.999, '0.999'], [.9999, '0.9999']] }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const b2 = +ctx.vals.b2, P = ctx.plot.clear(), pts = []; let pk = 0, tp = 1;
    for (let t = 1; t <= 1e5; t++) { const s = (1 - .9 ** t) / Math.sqrt(1 - b2 ** t); if (s > pk) { pk = s; tp = t; } if (t < 50 || t % Math.ceil(t / 60) === 0) pts.push([t, s]); }
    P.hline(1, { color: HEX.good, width: 2.2 });
    P.line(pts, { color: HEX.ball, width: 2.2 });
    P.scatter([[tp, pk]], { color: HEX.ball, r: 5 });
    P.draw();
    ctx.stat(`β₂ = ${b2}: uncorrected first step ${(.1 / Math.sqrt(1 - b2)).toFixed(2)}α, largest ${pk.toFixed(pk > 10 ? 2 : 3)}α at t = ${tp}; it settles to α only after about 1/(1 − β₂) = ${Math.round(1 / (1 - b2)).toLocaleString('en')} steps · corrected: α at every step`);
  },
};

// ---------- GD vs AdaGrad vs Adam in the valley, aligned or rotated (3D) ----------
const R_START = [-1.3, 1.2], T_R = 400;
function racePath(kind, H, lr) {
  const g = quadG(H), [mu, L] = eig2s(...H); let [x, y] = R_START, m = [0, 0], v = [0, 0], G = [0, 0]; const p = [[x, y]];
  for (let t = 1; t <= T_R; t++) {
    const gr = g(x, y);
    if (kind === 'gd') { x -= 2 / (L + mu) * gr[0]; y -= 2 / (L + mu) * gr[1]; }
    else if (kind === 'adagrad') { G = [G[0] + gr[0] ** 2, G[1] + gr[1] ** 2]; x -= lr * gr[0] / (Math.sqrt(G[0]) + 1e-8); y -= lr * gr[1] / (Math.sqrt(G[1]) + 1e-8); }
    else {
      m = [.9 * m[0] + .1 * gr[0], .9 * m[1] + .1 * gr[1]]; v = [.999 * v[0] + .001 * gr[0] ** 2, .999 * v[1] + .001 * gr[1] ** 2];
      const c1 = 1 - .9 ** t, c2 = 1 - .999 ** t;
      x -= lr * (m[0] / c1) / (Math.sqrt(v[0] / c2) + 1e-8); y -= lr * (m[1] / c1) / (Math.sqrt(v[1] / c2) + 1e-8);
    }
    p.push([x, y]);
  }
  return p;
}
const race = {
  title: 'Adaptive steps shine when the valley lines up with the axes',
  caption: 'A valley with condition number 100. Gradient descent with its best step (violet) against AdaGrad (white) and Adam (coral), which give every coordinate its own step size. Turn the valley by 45° and their advantage disappears.',
  note: 'f = ½ xᵀHx, curvatures 0.1 and 10, start (−1.3, 1.2), 400 steps, Adam β₁ = 0.9, β₂ = 0.999. Height drawn as √f. Deterministic gradients.',
  legend: [[HEX.opt2, 'GD, best step'], [HEX.opt3, 'AdaGrad'], [HEX.ball, 'Adam'], [HEX.good, 'minimum']],
  mode: '3d', cam: [2.6, 4.6, 3.0], tgt: [0, .1, 0],
  controls: [
    { id: 'deg', type: 'seg', label: 'valley', value: 0, options: [[0, 'along an axis'], [45, 'rotated 45°']] },
    { id: 'alpha', label: 'Adam step α', min: .005, max: .2, step: .005, value: .05, digits: 3 },
    { id: 'eta', label: 'AdaGrad step η', min: .05, max: 1, step: .05, value: .5 },
  ],
  action: 'Run again',
  charts: [{ id: 'f', title: 'f(x_t), log scale', xlabel: 'step t', ylabel: 'f', logy: true, xlim: [0, T_R], ylim: [1e-20, 10], h: 200, legend: false }],
  build(ctx) {
    ctx.H = hrot(100, +ctx.vals.deg);
    ctx.d = def({ L: quadF(ctx.H), g: quadG(ctx.H), dom: [-1.6, 1.6, -1.6, 1.6], h: v => Math.sqrt(v), fit: 1.4 });
    ctx.stage.surface(ctx.d);
    ctx.stage.marker(ctx.d, 0, 0, C.good, .1);
    this.drop(ctx);
  },
  onControl(ctx, id) {
    if (id !== 'deg') return this.drop(ctx);
    ctx.stage.clear(); this.build(ctx);
    ctx.stage.view(+ctx.vals.deg ? [3.96, 4.6, .28] : [2.6, 4.6, 3.0], [0, .1, 0]);   // same view relative to the valley
  },
  drop(ctx) {
    const s = ctx.stage, H = ctx.H; s.clearBalls();
    ctx.paths = { gd: racePath('gd', H), ada: racePath('adagrad', H, ctx.vals.eta), adam: racePath('adam', H, ctx.vals.alpha) };
    ctx.balls = [['gd', C.opt2], ['ada', C.opt3], ['adam', C.ball]].map(([k, col]) => s.ball(ctx.d, { start: R_START, color: col, speed: 1, maxSteps: T_R, step: (p, b) => ctx.d.clampP(ctx.paths[k][b.steps + 1] || p) }));
  },
  frame(ctx) {
    if (!ctx.balls) return;
    const n = ctx.balls[0].steps, f = quadF(ctx.H), P = ctx.charts.f.clear();
    for (const [k, col] of [['gd', HEX.opt2], ['ada', HEX.opt3], ['adam', HEX.ball]]) P.line(ctx.paths[k].slice(0, n + 1).map((p, t) => [t, Math.max(f(...p), 1e-30)]), { color: col, width: 1.8 });
    P.draw();
    const e = k => f(...ctx.paths[k][T_R]).toExponential(1);
    ctx.stat(`f after ${T_R} steps: GD ${e('gd')} · AdaGrad ${e('ada')} · Adam ${e('adam')}`);
  },
};

// ---------- when Adam fails: the counterexample of Taniguchi et al. 2024 ----------
const T_A = 400000;
const tMarks = (() => { const s = new Set(); for (let k = 0; k <= 160; k++) s.add(Math.round(10 ** (k / 160 * Math.log10(T_A)))); return [...s].sort((a, b) => a - b); })();
const adamFail = {
  title: 'A one-dimensional problem where Adam walks the wrong way',
  caption: 'Minimize f(θ) = θ on [−1, 1]: the answer is θ = −1. Each stochastic gradient is k² with probability 1/k and −k otherwise, so it is right on average (+1). Adam (coral) is pushed to the worst point θ = +1 unless β₂ is close to 1. ADOPT (violet) reaches −1; AMSGrad (white) does for k = 10 and crawls for k = 50.',
  note: 'One seeded run of 400 000 steps, step ηₜ = 0.01/√(1 + 0.01t), β₁ = 0.9, θ kept in [−1, 1]. Horizontal axis logarithmic.',
  legend: [[HEX.ball, 'Adam'], [HEX.opt3, 'AMSGrad'], [HEX.opt2, 'ADOPT'], [HEX.good, 'θ* = −1']],
  mode: '2d',
  plot: { xlabel: 'step t', ylabel: 'θ_t', logx: true, xlim: [1, T_A], ylim: [-1.1, 1.1], legend: false },
  controls: [
    { id: 'k', type: 'seg', label: 'k (rarity of the big gradient)', value: 10, options: [[10, '10'], [50, '50']] },
    { id: 'b2', type: 'seg', label: 'β₂', value: .9, options: [[.1, '0.1'], [.5, '0.5'], [.9, '0.9'], [.99, '0.99'], [.999, '0.999']] },
  ],
  build(ctx) { this.run(ctx); },
  onControl(ctx) { this.run(ctx); },
  run(ctx) {
    const k = +ctx.vals.k, b2 = +ctx.vals.b2, b1 = .9, eps = 1e-8, r = rng(1), G = new Float64Array(T_A + 2);
    for (let i = 0; i < G.length; i++) G[i] = r() < 1 / k ? k * k : -k;
    let tha = 0, ma = 0, va = 0, ths = 0, ms = 0, vs = 0, vmax = 0, tho = 0, vo = G[0] ** 2, mo = G[1] / Math.max(Math.sqrt(vo), eps), mi = 0;
    const A = [], S = [], O = [], clip = x => Math.max(-1, Math.min(1, x));
    for (let t = 1; t <= T_A; t++) {
      const eta = .01 / Math.sqrt(1 + .01 * t), g = G[t];
      ma = b1 * ma + (1 - b1) * g; va = b2 * va + (1 - b2) * g * g;
      tha = clip(tha - eta * (ma / (1 - b1 ** t)) / (Math.sqrt(va / (1 - b2 ** t)) + eps));
      ms = b1 * ms + (1 - b1) * g; vs = b2 * vs + (1 - b2) * g * g; vmax = Math.max(vmax, vs / (1 - b2 ** t));
      ths = clip(ths - eta * (ms / (1 - b1 ** t)) / (Math.sqrt(vmax) + eps));
      tho = clip(tho - eta * mo); vo = b2 * vo + (1 - b2) * g * g; mo = b1 * mo + (1 - b1) * G[t + 1] / Math.max(Math.sqrt(vo), eps);
      if (t === tMarks[mi]) { A.push([t, tha]); S.push([t, ths]); O.push([t, tho]); mi++; }
    }
    const P = ctx.plot.clear();
    P.hline(-1, { color: HEX.good, dash: [5, 4], width: 1.4 }); P.hline(1, { color: 'rgba(255,93,115,.5)', dash: [5, 4], width: 1.2, text: 'worst point' });
    P.line(S, { color: HEX.opt3, width: 1.8 }); P.line(O, { color: HEX.opt2, width: 2 }); P.line(A, { color: HEX.ball, width: 2.2 });
    P.draw();
    ctx.stat(`k = ${k}, β₂ = ${b2} · θ after 400 000 steps: Adam ${tha.toFixed(3)} · AMSGrad ${ths.toFixed(3)} · ADOPT ${tho.toFixed(3)}`);
  },
};

// ---------- gradient clipping under (L0, L1)-smoothness (3D) ----------
const fZ = (x, y) => x * x / 2 + x ** 4 / 4 + y * y / 2, gZ = (x, y) => [x + x ** 3, y];
const BEST = { 1: [6, .9742], 2: [22, .3577], 4: [81, .1111], 8: [322, .0292] };   // best fixed step per start (check script)
function zPath(x0, step, cap = 200000, keep = 600) {
  let x = x0, y = 1; const p = [[x, y]]; let hit = null;
  for (let t = 0; t < cap; t++) {
    const g = gZ(x, y), n = Math.hypot(...g);
    if (n <= 1e-4) { hit = t; break; }
    const h = step(n); x -= h * g[0]; y -= h * g[1];
    if (!isFinite(x) || Math.abs(x) > 1e6) { hit = 'diverged'; break; }
    if (p.length < keep) p.push([x, y]);
  }
  return { p, hit };
}
const clipViz = {
  title: 'Clipping: big steps where it is flat, small steps where it is steep',
  caption: 'f(x, y) = x²/2 + x⁴/4 + y²/2 is steep far out along x and gentle near the minimum. A fixed step (violet) must be tiny to survive the steep start, then crawls. Clipped GD (coral) takes steps of length at most γ = 0.5 when the gradient is large and full steps η = 1 near the minimum.',
  note: 'Start (x₀, 1). Clipped step: x ← x − min(η, γη/‖∇f‖) ∇f with η = 1, γ = 0.5. Counts are iterations until ‖∇f‖ ≤ 10⁻⁴. Height on a log scale; the y-axis is stretched.',
  legend: [[HEX.opt2, 'GD, fixed step h'], [HEX.ball, 'clipped GD'], [HEX.good, 'minimum']],
  mode: '3d', cam: [.2, 4.0, 3.9], tgt: [0, .2, 0],
  controls: [
    { id: 'x0', type: 'seg', label: 'start x₀', value: 4, options: [[1, '1'], [2, '2'], [4, '4'], [8, '8']] },
    { id: 'lh', label: 'fixed step h', min: -3, max: 0, step: .001, value: Math.log10(.1111), fmt: v => (10 ** v).toPrecision(3) },
  ],
  actions: [
    { label: 'Best fixed step', fn: ctx => { ctx.set('lh', Math.log10(BEST[ctx.vals.x0][1])); clipViz.drop(ctx); } },
  ],
  build(ctx) {
    ctx.d = def({ L: fZ, g: gZ, dom: [-8.6, 8.6, -1.5, 1.5], square: true, h: v => Math.log1p(v), fit: 1.6 });
    ctx.stage.surface(ctx.d);
    ctx.stage.marker(ctx.d, 0, 0, C.good, .1);
    this.drop(ctx);
  },
  onControl(ctx, id) { if (id === 'x0') ctx.set('lh', Math.log10(BEST[ctx.vals.x0][1])); this.drop(ctx); },
  drop(ctx) {
    const s = ctx.stage, x0 = +ctx.vals.x0, h = 10 ** ctx.vals.lh; s.clearBalls();
    const fx = zPath(x0, () => h), cl = zPath(x0, n => Math.min(1, .5 / n));
    for (const [r, col] of [[fx, C.opt2], [cl, C.ball]]) s.ball(ctx.d, { start: [x0, 1], color: col, speed: .5, maxSteps: r.p.length - 1, step: (p, b) => ctx.d.clampP(r.p[b.steps + 1] || p) });
    const say = r => r.hit === null ? '> 200 000' : r.hit;
    ctx.stat(`x₀ = ${x0}, ‖∇f(x₀)‖ = ${Math.hypot(...gZ(x0, 1)).toFixed(1)} · iterations to ‖∇f‖ ≤ 10⁻⁴: fixed step h = ${h.toPrecision(3)}: ${say(fx)} · clipped: ${say(cl)} · best fixed step from this start: ${BEST[x0][0]} (h = ${BEST[x0][1]})`);
  },
};

export default { precond, bias, race, adamfail: adamFail, clip: clipViz };
