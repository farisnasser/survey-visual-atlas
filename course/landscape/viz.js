// Stop 11 visuals. Numbers checked in scripts/check-math/course_landscape.py (same seeded RNG as src/lib/math.js).
import { def, opt, C, HEX, segPts } from '../../src/lib/atlas3d.js';
import { rng, eigSym } from '../../src/lib/math.js';

// A ball that is never clamped to the drawn box: same function, huge domain, the surface's display scales.
const freeDef = (d, L, g) => Object.assign(def({ L, g, h: d.h, dom: [-1e3, 1e3, -1e3, 1e3] }), { cx: d.cx, cy: d.cy, sx: d.sx, sy: d.sy, hScale: d.hScale, hOff: d.hOff, hClip: d.hClip });
const gdRun = (g, p, eta, T) => { let [x, y] = p; for (let t = 0; t < T; t++) { const q = g(x, y); x -= eta * q[0]; y -= eta * q[1]; } return [x, y]; };
const near = (p, q, tol = 1e-3) => Math.hypot(p[0] - q[0], p[1] - q[1]) < tol;

// ---------- 1. symmetry: two tanh units fitting tanh(1.6x) + tanh(0.4x) ----------
const XS = Array.from({ length: 9 }, (_, i) => -2 + i / 2), YS = XS.map(x => Math.tanh(1.6 * x) + Math.tanh(.4 * x));
const mL = (a, b) => { let s = 0; for (let i = 0; i < 9; i++) { const r = Math.tanh(a * XS[i]) + Math.tanh(b * XS[i]) - YS[i]; s += r * r; } return s / 9; };
const sech2 = z => { const c = Math.cosh(z); return isFinite(c) ? 1 / (c * c) : 0; };
const mG = (a, b) => { let ga = 0, gb = 0; for (let i = 0; i < 9; i++) { const x = XS[i], r = Math.tanh(a * x) + Math.tanh(b * x) - YS[i]; ga += 2 * r * x * sech2(a * x); gb += 2 * r * x * sech2(b * x); } return [ga / 9, gb / 9]; };
const SAD = [.73627, .73627], MA = [1.6, .4], MB = [.4, 1.6];
const mirror = {
  title: 'Swap the two neurons: the same network, a second minimum',
  caption: 'Two hidden units with weights w₁, w₂ fit data made by units 1.6 and 0.4. Swapping the units changes nothing, so every minimum has a mirror image across the diagonal w₁ = w₂, and a saddle sits on the diagonal between them. Balls from random starts split between the two copies; a ball started exactly on the diagonal keeps both units identical forever and stops at the saddle.',
  note: 'Loss = mean over 9 points of (tanh(w₁x) + tanh(w₂x) − y)². Height drawn as √loss. Slide d to start the violet ball at (2 + d, 2 − d).',
  legend: [[HEX.good, 'global minima'], [HEX.neg, 'saddle: curvature −0.106'], [HEX.pos, '+0.775'], ['#ffffff', 'diagonal w₁ = w₂'], [HEX.ball, '24 random starts'], [HEX.opt2, 'start (2 + d, 2 − d)']],
  mode: '3d', cam: [-2.5, 3.2, 2.5], tgt: [0, .2, 0],
  controls: [{ id: 'd', label: 'offset d from the diagonal', min: -.1, max: .1, step: .001, value: 0, digits: 3 }],
  action: 'Drop the balls',
  build(ctx) {
    const d = ctx.d = def({ L: mL, g: mG, h: v => Math.sqrt(v), dom: [-.5, 2.5, -.5, 2.5], N: 120, fit: 1.5 });
    const st = ctx.stage; st.surface(d); ctx.fd = freeDef(d, mL, mG);
    st.line(d, segPts(-.5, -.5, 2.5, 2.5), C.white, .8, .02);
    st.marker(d, ...MA, C.good, .09); st.marker(d, ...MB, C.good, .09);
    st.eigenLines(d, ...SAD, { len: .45 });
    st.label('w₁ →', d.W(2.3, -.5, .15), '', HEX.muted); st.label('w₂ →', d.W(-.5, 2.3, .15), '', HEX.muted);
    const R = rng(3); ctx.starts = Array.from({ length: 24 }, () => [-.3 + 2.8 * R(), -.3 + 2.8 * R()]);
    ctx.ends = ctx.starts.map(s => gdRun(mG, s, .5, 3000));
    this.drop(ctx);
  },
  drop(ctx) {
    const st = ctx.stage, dd = ctx.vals.d; st.clearBalls();
    ctx.bs = ctx.starts.map(s => st.ball(ctx.fd, { start: s, step: opt.gd(.5), speed: 10, maxSteps: 3000, r: .04 }));
    ctx.main = st.ball(ctx.fd, { start: [2 + dd, 2 - dd], step: opt.gd(.5), speed: 10, maxSteps: 3000, color: C.opt2, r: .06 });
    ctx.mainEnd = gdRun(mG, [2 + dd, 2 - dd], .5, 3000);
  },
  onControl(ctx) { this.drop(ctx); },
  frame(ctx) {
    if (!ctx.main) return;
    const a = ctx.ends.filter(p => near(p, MA)).length, b = ctx.ends.filter(p => near(p, MB)).length, e = ctx.mainEnd;
    const where = near(e, SAD) ? 'the saddle (0.736, 0.736): its units stay identical' : near(e, MA) ? 'the minimum (1.6, 0.4)' : near(e, MB) ? 'the minimum (0.4, 1.6)' : 'not settled';
    ctx.stat(`step ${ctx.main.steps} of 3000 (step size 0.5) · random starts: ${a} end at (1.6, 0.4), ${b} at (0.4, 1.6) · violet ball, d = ${ctx.vals.d.toFixed(3)}: ends at ${where}`);
  },
};

// ---------- 2. benign: f(u) = 1/2 ||u u^T - diag(1, s)||^2 ----------
const bL = s => (x, y) => .5 * ((x * x - 1) ** 2 + 2 * (x * y) ** 2 + (y * y - s) ** 2);
const bG = s => (x, y) => [2 * ((x * x - 1) * x + x * y * y), 2 * (x * x * y + (y * y - s) * y)];
const benign = {
  title: 'A landscape with no bad minima',
  caption: 'Fit a 2 × 2 matrix diag(1, s) by u uᵀ, the smallest matrix-factorization problem. The only minima are ±(1, 0), both global; the origin is a maximum and ±(0, √s) are strict saddles. Every random start reaches a global minimum. Only the violet ball, started exactly on the saddle\'s stable set (the vertical axis), ends at the saddle.',
  note: 'f(u) = ½‖u uᵀ − diag(1, s)‖², u ∈ ℝ²; GD step 0.1, 2000 steps. Height drawn on a log scale.',
  legend: [[HEX.good, 'global minima'], [HEX.neg, 'negative curvature'], [HEX.pos, 'positive curvature'], [HEX.ball, '24 random starts'], [HEX.opt2, 'start (0, 1.2)']],
  mode: '3d', cam: [1.6, 4.5, 3.6], tgt: [0, .45, 0],
  controls: [{ id: 's', label: 'second eigenvalue s', min: .1, max: .9, step: .05, value: .5, digits: 2 }],
  action: 'Drop the balls',
  build(ctx) {
    const s = ctx.vals.s, st = ctx.stage, L = bL(s), g = bG(s);
    st.clear();
    const d = ctx.d = def({ L, g, dom: [-1.5, 1.5, -1.5, 1.5], N: 110, fit: 1.6 });
    st.surface(d); ctx.fd = freeDef(d, L, g);
    st.marker(d, 1, 0, C.good, .1); st.marker(d, -1, 0, C.good, .1);
    for (const sg of [1, -1]) st.eigenLines(d, 0, sg * Math.sqrt(s), { len: .4, labels: sg > 0 });
    st.label('max', d.W(0, 0, .2), '', HEX.muted);
    const R = rng(5); ctx.starts = Array.from({ length: 24 }, () => [-1.4 + 2.8 * R(), -1.4 + 2.8 * R()]);
    ctx.ends = ctx.starts.map(p => gdRun(g, p, .1, 2000)); ctx.mainEnd = gdRun(g, [0, 1.2], .1, 2000);
    this.drop(ctx);
  },
  drop(ctx) {
    const st = ctx.stage; st.clearBalls();
    ctx.bs = ctx.starts.map(p => st.ball(ctx.fd, { start: p, step: opt.gd(.1), speed: 8, maxSteps: 2000, r: .04 }));
    ctx.main = st.ball(ctx.fd, { start: [0, 1.2], step: opt.gd(.1), speed: 8, maxSteps: 2000, color: C.opt2, r: .06 });
  },
  onControl(ctx) { this.build(ctx); },
  frame(ctx) {
    if (!ctx.main) return;
    const s = ctx.vals.s, p = ctx.ends.filter(q => near(q, [1, 0])).length, m = ctx.ends.filter(q => near(q, [-1, 0])).length;
    ctx.stat(`s = ${s.toFixed(2)} · step ${ctx.main.steps} of 2000 · random starts: ${p} reach (1, 0), ${m} reach (−1, 0): ${p + m} of 24 at a global minimum · saddle curvatures ${(2 * (s - 1)).toFixed(2)} and ${(4 * s).toFixed(2)} · violet ball ends at ${near(ctx.mainEnd, [0, Math.sqrt(s)], 1e-6) ? `the saddle (0, ${Math.sqrt(s).toFixed(3)})` : 'a minimum'}`);
  },
};

// ---------- 3. one neuron, one example: F(v, w) = (1 - v phi(w))^2 ----------
const ACT = { quad: [z => -((z - 1) ** 2) / 8, z => -(z - 1) / 4], tanh: [Math.tanh, z => sech2(z)] };
const nF = k => (v, w) => (1 - v * ACT[k][0](w)) ** 2;
const nG = k => (v, w) => { const [p, dp] = ACT[k], e = 1 - v * p(w); return [-2 * e * p(w), -2 * e * v * dp(w)]; };
const neuron = {
  title: 'A bad minimum in the smallest network',
  caption: 'One neuron v·φ(w·x) fitting one example (x, y) = (1, 1). With the activation φ(z) = −(z − 1)²/8, the line w = 1 (white) is a trap: φ is at its maximum 0 there, so for v > 0 every nearby change makes the output worse, and the loss is stuck at 1 while 0 is possible (green curves). With tanh there is no such line, and every ball reaches zero loss.',
  note: 'F(v, w) = (1 − v·φ(w))², the example of Sun et al. 2020 (Fig. 2). 30 random starts, GD step 0.05, 6000 steps. Height log(1 + F).',
  legend: [[HEX.good, 'global minima, F = 0'], ['#ffffff', 'spurious minima, F = 1'], [HEX.ball, 'GD from 30 starts']],
  mode: '3d', cam: [0, 4.4, 4.4], tgt: [0, .2, 0],
  controls: [{ id: 'act', type: 'seg', label: 'activation φ', value: 'quad', options: [['quad', '−(z − 1)²/8'], ['tanh', 'tanh']] }],
  action: 'Drop the balls',
  build(ctx) {
    const k = ctx.vals.act, st = ctx.stage, L = nF(k), g = nG(k), phi = ACT[k][0];
    st.clear();
    const d = ctx.d = def({ L, g, dom: [-3, 3, -3, 3], h: v => Math.log1p(Math.min(v, 1e4)), N: 120 });
    st.surface(d); ctx.fd = freeDef(d, L, g);
    const pos = [], neg = []; for (let w = -3; w <= 3.0001; w += .01) { const v = 1 / phi(w); if (isFinite(v) && Math.abs(v) <= 3) (v > 0 ? pos : neg).push([v, w]); }
    const split = a => { const out = [[]]; a.forEach((p, i) => { if (i && Math.abs(p[1] - a[i - 1][1]) > .02) out.push([]); out[out.length - 1].push(p); }); return out; };
    [...split(pos), ...split(neg)].forEach(sg => sg.length > 1 && st.tube(d, sg, C.good, .014));
    if (k === 'quad') st.tube(d, segPts(0, 1, 3, 1), C.white, .018);
    st.label('v →', d.W(2.7, -3, .15), '', HEX.muted); st.label('w →', d.W(-3, 2.6, .15), '', HEX.muted);
    const R = rng(9); ctx.starts = Array.from({ length: 30 }, () => [-3 + 6 * R(), -3 + 6 * R()]);
    ctx.F = ctx.starts.map(p => L(...gdRun(g, p, .05, 6000)));
    this.drop(ctx);
  },
  drop(ctx) {
    const st = ctx.stage; st.clearBalls();
    ctx.bs = ctx.starts.map(p => st.ball(ctx.fd, { start: p, step: opt.gd(.05), speed: 20, maxSteps: 6000, r: .045 }));
  },
  onControl(ctx) { this.build(ctx); },
  frame(ctx) {
    if (!ctx.bs) return;
    const z = ctx.F.filter(f => f < 1e-6).length, one = ctx.F.filter(f => Math.abs(f - 1) < 1e-3).length;
    ctx.stat(`φ = ${ctx.vals.act === 'quad' ? '−(z − 1)²/8' : 'tanh'} · step ${ctx.bs[0].steps} of 6000 · after 6000 steps: ${z} of 30 at loss 0, ${one} stuck at loss 1${ctx.vals.act === 'quad' ? ' · from a 41 × 41 grid of starts, 30.4% get stuck' : ' · from a 41 × 41 grid, only the start (0, 0) (a saddle) stays at 1'}`);
  },
};

// ---------- 4. the PL condition on a non-convex function ----------
const fP = x => x * x + 3 * Math.sin(x) ** 2, dP = x => 2 * x + 3 * Math.sin(2 * x), MU = .17553, LP = 8;
const pl = {
  title: 'Non-convex, yet GD converges geometrically',
  caption: 'f(x) = x² + 3 sin²x bends downward in places (f″ < 0, red), so it is not convex. But the PL ratio ½f′²/(f − f*) never drops below μ = 0.1755 (chart “PL ratio”), so every step removes a fixed fraction of the remaining gap. Chart “gap”: the real run against the PL guarantee.',
  note: 'f* = 0 at x = 0; L = max f″ = 8. Guarantee for 0 < η < 2/L: f(x_t) − f* ≤ (1 − 2μη(1 − Lη/2))ᵗ (f(x₀) − f*).',
  legend: [['#8ea0d0', 'f'], [HEX.neg, 'f″ < 0'], [HEX.ball, 'GD'], [HEX.opt2, 'PL guarantee']],
  mode: '2d',
  plot: { xlabel: 'x', ylabel: 'f(x)', xlim: [-6.2, 6.2], ylim: [-1, 40], legend: false },
  controls: [
    { id: 'x0', label: 'start x₀', min: -6, max: 6, step: .1, value: 5.5, digits: 1 },
    { id: 'eta', label: 'step size η', min: .01, max: .24, step: .005, value: .125, digits: 3 },
  ],
  charts: [
    { id: 'r', title: 'PL ratio ½ f′(x)² / (f(x) − f*)', xlabel: 'x', ylabel: 'ratio', xlim: [-6.2, 6.2], ylim: [0, 9], h: 160, legend: false },
    { id: 'g', title: 'gap f(x_t) − f*', xlabel: 'step t', ylabel: 'gap', logy: true, xlim: [0, 60], ylim: [1e-12, 100], h: 180, legend: false },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { x0, eta } = ctx.vals, P = ctx.plot.clear(), xs = [x0];
    for (let t = 0; t < 400; t++) xs.push(xs[t] - eta * dP(xs[t]));
    const neg = []; for (let x = -6.2; x <= 6.2; x += .01) neg.push([x, 2 + 6 * Math.cos(2 * x) < 0 ? fP(x) : NaN]);
    P.line(neg, { color: HEX.neg, width: 7, alpha: .45 });
    P.fn(fP, { color: '#8ea0d0', width: 2 });
    P.line(xs.slice(0, 30).map(x => [x, fP(x)]), { color: HEX.ball, width: 1.2, alpha: .7, dots: 3 });
    P.scatter([[0, 0]], { color: HEX.good, r: 5 });
    P.draw();
    const R = ctx.charts.r.clear();
    R.fn(x => Math.abs(x) < 1e-9 ? 8 : .5 * dP(x) ** 2 / fP(x), { color: '#8ea0d0', width: 2 });
    R.hline(MU, { color: HEX.opt2, dash: [5, 4], text: 'μ = 0.1755' });
    R.draw();
    const q = 1 - 2 * MU * eta * (1 - LP * eta / 2), f0 = fP(x0), G = ctx.charts.g.clear();
    G.line(xs.slice(0, 61).map((x, t) => [t, Math.max(fP(x), 1e-300)]), { color: HEX.ball, width: 2, dots: 1.5 });
    G.line(Array.from({ length: 61 }, (_, t) => [t, q ** t * f0]), { color: HEX.opt2, width: 2, dash: [5, 4] });
    G.draw();
    const hit = xs.findIndex(x => fP(x) < 1e-10), need = f0 < 1e-10 ? 0 : Math.ceil(Math.log(1e-10 / f0) / Math.log(q));
    ctx.stat(`x₀ = ${x0.toFixed(1)}, f(x₀) = ${f0.toFixed(2)} · η = ${eta.toFixed(3)} · guaranteed factor per step 1 − 2μη(1 − Lη/2) = ${q.toFixed(4)} · gap below 10⁻¹⁰: guaranteed by step ${need.toLocaleString('en')}, reached at step ${hit < 0 ? '> 400' : hit}`);
  },
};

// ---------- 5. wide networks: GD on f(x) = m^{-1/2} sum a_r tanh(w_r . x) ----------
const NW = 8, TT = Array.from({ length: NW }, (_, i) => 2 * Math.PI * i / NW + .3);
const XW = TT.map(t => [Math.cos(t) / Math.SQRT2, Math.sin(t) / Math.SQRT2, 1 / Math.SQRT2]), YW = TT.map(t => Math.sin(2 * t) + .5 * Math.cos(3 * t));
const WIDTHS = [4, 16, 64, 256, 1024], WCOL = { 4: HEX.neg, 16: HEX.flat, 64: HEX.pos, 256: HEX.opt2, 1024: HEX.ball };
function gram(W, m) {
  const S = XW.map(x => W.map(w => sech2(w[0] * x[0] + w[1] * x[1] + w[2] * x[2])));
  return XW.map((xi, i) => XW.map((xj, j) => { let s = 0; for (let r = 0; r < m; r++) s += S[i][r] * S[j][r]; return s / m * (xi[0] * xj[0] + xi[1] * xj[1] + xi[2] * xj[2]); }));
}
function trainWide(m, T = 1000) {
  const R = rng(7), W = Array.from({ length: m }, () => [R.normal(), R.normal(), R.normal()]), a = W.map((_, r) => r % 2 ? -1 : 1), W0 = W.map(w => w.slice());
  const e0 = eigSym(gram(W, m)).values, eta = 1 / e0[NW - 1], sm = Math.sqrt(m), L = [];
  for (let k = 0; k <= T; k++) {
    const Z = XW.map(x => W.map(w => w[0] * x[0] + w[1] * x[1] + w[2] * x[2]));
    const res = Z.map((z, i) => z.reduce((s, zr, r) => s + a[r] * Math.tanh(zr), 0) / sm - YW[i]);
    L.push(.5 * res.reduce((s, v) => s + v * v, 0));
    if (k === T) break;
    for (let r = 0; r < m; r++) {
      let g0 = 0, g1 = 0, g2 = 0;
      for (let i = 0; i < NW; i++) { const c = res[i] * sech2(Z[i][r]) * a[r]; g0 += c * XW[i][0]; g1 += c * XW[i][1]; g2 += c * XW[i][2]; }
      W[r][0] -= eta * g0 / sm; W[r][1] -= eta * g1 / sm; W[r][2] -= eta * g2 / sm;
    }
  }
  let num = 0, den = 0; W.forEach((w, r) => w.forEach((v, j) => { num += (v - W0[r][j]) ** 2; den += W0[r][j] ** 2; }));
  const eT = eigSym(gram(W, m)).values;
  return { lam0: e0[0], eta, L, move: Math.sqrt(num / den), dlam: Math.abs(eT[0] - e0[0]) / e0[0] };
}
const wide = {
  title: 'Wider networks: the weights barely move, and GD reaches zero loss',
  caption: 'A one-hidden-layer tanh network with m units, trained by full-batch GD on 8 points. Every width from 16 up drives the loss below 10⁻⁶ in 1000 steps. The wider the network, the less its weights have to move to do it (bottom: about 1/√m), which is what the NTK analyses exploit.',
  note: 'f(x) = m^(−1/2) Σ a_r tanh(w_r·x), a_r = ±1 fixed, w_r ~ N(0, I); step η = 1/λ_max(G₀). Dashed: Du et al.\'s rate (1 − ηλ₀/2)ᵏ L(0) for the highlighted width, with λ₀ = λ_min(G₀).',
  legend: WIDTHS.map(m => [WCOL[m], `m = ${m}`]),
  mode: '2d',
  plot: { xlabel: 'step k', ylabel: 'loss ½‖f − y‖²', logy: true, xlim: [0, 1000], ylim: [1e-12, 10], legend: false },
  controls: [{ id: 'm', label: 'highlight width m', type: 'seg', value: 1024, options: WIDTHS.map(m => [m, String(m)]) }],
  charts: [{ id: 'mv', title: 'relative weight movement ‖W_T − W₀‖ / ‖W₀‖', xlabel: 'width m', ylabel: 'movement', logx: true, logy: true, xlim: [2, 2048], ylim: [.05, 10], h: 180, legend: false }],
  build(ctx) { ctx.res = Object.fromEntries(WIDTHS.map(m => [m, trainWide(m)])); this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const sel = +ctx.vals.m, P = ctx.plot.clear(), R = ctx.res;
    for (const m of WIDTHS) P.line(R[m].L.map((v, k) => [k, Math.max(v, 1e-300)]), { color: WCOL[m], width: m === sel ? 2.4 : 1.1, alpha: m === sel ? 1 : .35 });
    const r = R[sel], q = 1 - r.eta * r.lam0 / 2;
    P.line(Array.from({ length: 101 }, (_, i) => [10 * i, q ** (10 * i) * r.L[0]]), { color: WCOL[sel], width: 1.4, dash: [6, 4] });
    P.draw();
    const M = ctx.charts.mv.clear();
    M.line([[2, R[1024].move * Math.sqrt(1024 / 2)], [2048, R[1024].move * Math.sqrt(1024 / 2048)]], { color: 'rgba(232,236,248,.4)', dash: [4, 4] });
    M.text(5, R[1024].move * Math.sqrt(1024 / 5) * .55, '∝ 1/√m', { color: 'rgba(232,236,248,.6)' });
    WIDTHS.forEach(m => M.scatter([[m, R[m].move]], { color: WCOL[m], r: m === sel ? 6 : 4 }));
    M.draw();
    ctx.stat(`m = ${sel} · λ_min(G₀) = ${r.lam0.toFixed(4)}, η = ${r.eta.toFixed(3)} · loss ${r.L[0].toFixed(3)} → ${r.L[1000].toExponential(2)} after 1000 steps · weights moved ${r.move.toFixed(3)} of their initial norm · λ_min(G) changed by ${(100 * r.dlam).toFixed(1)}%`);
  },
};

export default { mirror, benign, neuron, pl, wide };
