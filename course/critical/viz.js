// Stop 5 visuals. Numbers checked in scripts/check-math/course_critical.py.
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { rng, eigSym } from '../../src/lib/math.js';
import { contours, drawContours, heat } from '../../src/learn/plot2d.js';

const f2 = v => (Math.abs(v) < 5e-3 ? 0 : v).toFixed(2);

// ---------- the zoo of critical points at the origin ----------
const ZOO = {
  min: { name: 'x² + y²', L: (x, y) => x * x + y * y, g: (x, y) => [2 * x, 2 * y], H: [2, 0, 2], kind: 'local minimum' },
  max: { name: '−x² − y²', L: (x, y) => -x * x - y * y, g: (x, y) => [-2 * x, -2 * y], H: [-2, 0, -2], kind: 'local maximum' },
  saddle: { name: 'x² − y²', L: (x, y) => x * x - y * y, g: (x, y) => [2 * x, -2 * y], H: [2, 0, -2], kind: 'strict saddle' },
  monkey: { name: 'x³ − 3xy²', L: (x, y) => x ** 3 - 3 * x * y * y, g: (x, y) => [3 * x * x - 3 * y * y, -6 * x * y], H: [0, 0, 0], kind: 'degenerate saddle (monkey saddle)' },
  flatmin: { name: 'x² + y⁴', L: (x, y) => x * x + y ** 4, g: (x, y) => [2 * x, 4 * y ** 3], H: [2, 0, 0], kind: 'degenerate minimum' },
  flatsad: { name: 'x² − y⁴', L: (x, y) => x * x - y ** 4, g: (x, y) => [2 * x, -4 * y ** 3], H: [2, 0, 0], kind: 'degenerate saddle' },
};
const BZ = 1.5, RING = Array.from({ length: 12 }, (_, k) => [Math.round(Math.cos(k * Math.PI / 6) * 1e12) / 1e12, Math.round(Math.sin(k * Math.PI / 6) * 1e12) / 1e12]);
const zoo = {
  title: 'A zoo of flat points',
  caption: 'Every surface has ∇f = 0 at the centre. The Hessian there (tubes) sorts them: all curvatures up, all down, or mixed. Drop twelve balls from the unit circle and see which flat points actually catch gradient descent.',
  note: 'GD with η = 0.1, 400 steps; heights rescaled to fit. A ball "is caught" if it ends within 0.1 of the centre.',
  legend: [[HEX.pos, 'λ > 0'], [HEX.neg, 'λ < 0'], [HEX.flat, 'λ = 0'], [HEX.ball, 'GD balls']],
  mode: '3d', cam: [2.6, 4.7, 4.1], tgt: [0, .2, 0],
  controls: [
    { id: 'fa', type: 'seg', label: 'all λ ≠ 0', value: 'saddle', options: ['min', 'max', 'saddle'].map(k => [k, ZOO[k].name.replace(/ /g, '')]) },
    { id: 'fb', type: 'seg', label: 'some λ = 0', value: '', options: ['monkey', 'flatmin', 'flatsad'].map(k => [k, ZOO[k].name.replace(/ /g, '')]) },
  ],
  actions: [{ label: 'Drop 12 balls', fn: ctx => zoo.drop(ctx) }],
  build(ctx) { ctx.vals.fn = 'saddle'; this.scene(ctx); },
  onControl(ctx, id, v) {   // two button rows act as one choice: pressing one row clears the other
    ctx.vals.fn = v;
    ctx.widget.slot.querySelectorAll('.segrow').forEach((row, k) => { if ((k === 0) !== (id === 'fa')) row.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', 'false')); });
    this.scene(ctx);
  },
  scene(ctx) {
    const z = ZOO[ctx.vals.fn], st = ctx.stage;
    st.clear(); ctx.balls = null;
    ctx.d = def({ L: z.L, g: z.g, dom: [-BZ, BZ, -BZ, BZ], h: v => v, fit: 1.4 });
    ctx.d.hess = () => z.H;
    st.surface(ctx.d);
    if (z.H.every(v => v === 0)) {   // zero Hessian: every direction is an eigenvector with λ = 0; draw the two axes
      for (const [ux, uy] of [[1, 0], [0, 1]]) { const pts = []; for (let t = -1.1; t <= 1.1001; t += .0275) pts.push([t * ux, t * uy]); st.tube(ctx.d, pts, C.flat); }
      st.label('λ = 0', ctx.d.W(.9, 0, .22), '', HEX.flat); st.label('λ = 0', ctx.d.W(0, .9, .22), '', HEX.flat);
    } else st.eigenLines(ctx.d, 0, 0, { len: 1.1, eps: 1e-6 });
    st.dot(ctx.d.W(0, 0, .03), C.white, .05);
    this.info(ctx);
  },
  drop(ctx) {
    const st = ctx.stage; st.clearBalls();
    const out = b => Math.abs(b.p[0]) >= BZ - 1e-9 || Math.abs(b.p[1]) >= BZ - 1e-9 || Math.hypot(...b.p) < 1e-3;
    ctx.balls = RING.map(s => st.ball(ctx.d, { start: s, step: opt.gd(.1), speed: 1, maxSteps: 400, r: .05, stop: out }));
  },
  frame(ctx) { if (ctx.balls) this.info(ctx); },
  info(ctx) {
    const z = ZOO[ctx.vals.fn], [a, b, e] = z.H, idx = [a, e].filter(l => l < 0).length;
    let tail = 'drop the balls';
    if (ctx.balls) {
      const done = ctx.balls.every(b => b.done), caught = ctx.balls.filter(b => Math.hypot(...b.p) < .1).length;
      tail = `${done ? '' : 'running · '}caught ${caught} of 12`;
    }
    ctx.stat(`f = ${z.name} · Hessian at 0 = [[${a}, ${b}], [${b}, ${e}]] · index ${idx} → ${z.kind} · ${tail}`);
  },
};

// ---------- saddles made by symmetry: one hidden unit vs a deeper chain ----------
const SY = {
  one: { name: '(1 − uv)²', L: (u, v) => (1 - u * v) ** 2, g: (u, v) => [-2 * v * (1 - u * v), -2 * u * (1 - u * v)], H: [0, -2, 0] },
  two: { name: '(1 − uv²)²', L: (u, v) => (1 - u * v * v) ** 2, g: (u, v) => [-2 * v * v * (1 - u * v * v), -4 * u * v * (1 - u * v * v)], H: [0, 0, 0] },
};
const plateau = (z, s) => { let p = [s, .6 * s]; for (let t = 1; t <= 400000; t++) { const g = z.g(...p); p = [Math.max(-1.6, Math.min(1.6, p[0] - .05 * g[0])), Math.max(-1.6, Math.min(1.6, p[1] - .05 * g[1]))]; if (z.L(...p) < .01) return t; } return null; };
const sym = {
  title: 'Saddles made by symmetry',
  caption: 'The loss of the smallest two-weight network, (1 − uv)², is the same at (u, v) and (−u, −v), so its gradient vanishes at the centre: a saddle between the two green valleys of minima uv = 1. The violet chord joins two minima that the symmetry swaps and climbs to f = 1 in between. Switch to the deeper chain: its saddle is flat, and the ball lingers far longer.',
  note: 'Heights on a log scale. The deeper chain is (1 − abc)² seen on the slice c = b. GD with η = 0.05 from (s, 0.6s), stopped when f < 0.01.',
  legend: [[HEX.good, 'minima uv = 1 (or uv² = 1)'], [HEX.opt2, 'chord between two minima'], [HEX.ball, 'GD from near the saddle']],
  mode: '3d', cam: [3.3, 4.3, -3.3], tgt: [0, .25, 0],
  controls: [
    { id: 'fn', type: 'seg', label: 'network', value: 'one', options: [['one', '(1−uv)²'], ['two', 'deeper: (1−uv²)²']] },
    { id: 'ls', label: 'start size s', min: -3, max: -1, step: .05, value: -2, fmt: v => (10 ** v).toPrecision(2) },
  ],
  action: 'Drop the ball',
  build(ctx) { this.scene(ctx); },
  onControl(ctx, id) { if (id === 'fn') this.scene(ctx); else this.drop(ctx); },
  scene(ctx) {
    const z = SY[ctx.vals.fn], st = ctx.stage;
    st.clear();
    ctx.d = def({ L: z.L, g: z.g, dom: [-1.6, 1.6, -1.6, 1.6], fit: 1.6 });
    ctx.d.hess = () => z.H;
    st.surface(ctx.d);
    const deep = ctx.vals.fn === 'two', mins = [];
    for (const sg of [1, -1]) {
      const a = [];
      if (deep) for (let v = .79; v <= 1.6001; v += .01) { const u = 1 / (v * v); if (u <= 1.6) a.push([u, sg * v]); }   // u v² = 1, v = ±
      else for (let u = .625; u <= 1.6001; u += .01) a.push([sg * u, sg / u]);                                          // u v = 1, both branches
      mins.push(a);
    }
    mins.forEach(a => st.tube(ctx.d, a, C.good, .016));
    const A = [1, 1], Bm = deep ? [1, -1] : [-1, -1], chord = [];   // the symmetry maps one minimum to the other
    for (let t = 0; t <= 1.0001; t += .01) chord.push([A[0] + (Bm[0] - A[0]) * t, A[1] + (Bm[1] - A[1]) * t]);
    st.tube(ctx.d, chord, C.opt2, .013, .03);
    if (deep) { for (const [ux, uy] of [[1, 0], [0, 1]]) { const pts = []; for (let t = -.9; t <= .9001; t += .0225) pts.push([t * ux, t * uy]); st.tube(ctx.d, pts, C.flat); } }
    else st.eigenLines(ctx.d, 0, 0, { len: .9, eps: 1e-6 });
    this.drop(ctx);
  },
  drop(ctx) {
    const z = SY[ctx.vals.fn], s = 10 ** ctx.vals.ls, st = ctx.stage;
    ctx.counts = { one: plateau(SY.one, s), two: plateau(SY.two, s) };
    st.clearBalls();
    ctx.ball = st.ball(ctx.d, { start: [s, .6 * s], step: opt.gd(.05), speed: ctx.vals.fn === 'two' ? 6 : 1, maxSteps: 400000, stop: b => z.L(...b.p) < .01 });
  },
  frame(ctx) {
    const b = ctx.ball; if (!b) return;
    const z = SY[ctx.vals.fn], [a, c, e] = z.H;
    ctx.stat(`f = ${z.name} · Hessian at 0 = [[${a}, ${c}], [${c}, ${e}]] → ${ctx.vals.fn === 'one' ? 'strict saddle (λ = −2, 2)' : 'degenerate saddle (λ = 0, 0)'} · this ball: step ${b.steps}, f = ${b.loss().toFixed(3)} · steps until f < 0.01 from s = ${(10 ** ctx.vals.ls).toPrecision(2)}: one unit ${ctx.counts.one}, deeper chain ${ctx.counts.two}`);
  },
};

// ---------- random Hessians: how rare is "all eigenvalues positive"? ----------
function goe(r, N) {
  const A = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let i = 0; i < N; i++) for (let j = i; j < N; j++) { const v = r.normal() * (i === j ? Math.sqrt(2 / N) : Math.sqrt(1 / N)); A[i][j] = v; A[j][i] = v; }
  return A;
}
function cholPD(A) {   // Cholesky succeeds exactly when all eigenvalues are > 0
  const N = A.length, Lm = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j]; for (let k = 0; k < j; k++) s -= Lm[i][k] * Lm[j][k];
    if (i === j) { if (s <= 0) return false; Lm[i][i] = Math.sqrt(s); } else Lm[i][j] = s / Lm[j][j];
  }
  return true;
}
const SAMPLES = 2000;
function pmin(N, m) { const r = rng(1000 + N); let c = 0; for (let k = 0; k < SAMPLES; k++) { const A = goe(r, N); for (let i = 0; i < N; i++) A[i][i] += m; c += cholPD(A); } return c / SAMPLES; }
const EV100 = eigSym(goe(rng(7), 100)).values;
const random = {
  title: 'Random curvature: minima are rare, saddles are common',
  caption: 'A random symmetric 100×100 matrix as a stand-in for the Hessian at a critical point: its eigenvalues fill a semicircle. Shifting the spectrum right (lower loss, in Bray and Dean\'s picture) leaves fewer negative eigenvalues. The chart: how often every one of N eigenvalues is positive.',
  note: 'Gaussian orthogonal ensemble, variance 1/N off the diagonal; the shift m is our slider, not a fitted model. 2,000 matrices per N.',
  legend: [[HEX.neg, 'negative eigenvalues (downhill directions)'], [HEX.pos, 'positive eigenvalues'], ['#e8ecf8', 'semicircle law']],
  mode: '2d',
  plot: { xlabel: 'eigenvalue λ', ylabel: 'density', xlim: [-3, 4.5], ylim: [0, .5], legend: false },
  charts: [{ id: 'p', title: 'P(all N eigenvalues > 0)', xlabel: 'dimension N', ylabel: 'probability', logy: true, xlim: [.5, 10.5], ylim: [3e-4, 1], h: 200, legend: false }],
  controls: [{ id: 'm', label: 'shift m', min: -1, max: 2.5, step: .05, value: 0, digits: 2 }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const m = ctx.vals.m, ev = EV100.map(l => l + m), P = ctx.plot.clear(), bw = .2, bins = new Map();
    for (const l of ev) { const k = Math.floor(l / bw); bins.set(k, (bins.get(k) || 0) + 1); }
    P.custom((g, p) => {
      for (const [k, n] of bins) {
        const x0 = k * bw, x1 = x0 + bw, h = n / (100 * bw);
        g.fillStyle = x1 <= 0 ? 'rgba(255,93,115,.75)' : x0 >= 0 ? 'rgba(124,196,255,.7)' : 'rgba(255,200,87,.7)';
        g.fillRect(p.sx(x0) + .5, p.sy(h), p.sx(x1) - p.sx(x0) - 1, p.sy(0) - p.sy(h));
      }
    });
    P.fn(x => { const t = 4 - (x - m) ** 2; return t > 0 ? Math.sqrt(t) / (2 * Math.PI) : 0; }, { color: '#e8ecf8', width: 1.6 });
    P.vline(0, { color: 'rgba(232,236,248,.5)', dash: [4, 4], text: 'λ = 0' });
    P.draw();
    const ps = []; for (let N = 1; N <= 10; N++) ps.push([N, pmin(N, m)]);
    const Cp = ctx.charts.p.clear();
    Cp.line(ps.filter(q => q[1] > 0), { color: HEX.good, width: 1.8, dots: 3.5 });
    Cp.draw();
    const k = ev.filter(l => l < 0).length;
    ctx.stat(`shift m = ${m.toFixed(2)} · this 100×100 sample: ${k} negative eigenvalues (index ${k}, fraction ${(k / 100).toFixed(2)}) · P(all positive): N = 1: ${ps[0][1].toFixed(3)}, N = 4: ${ps[3][1].toFixed(4)}, N = 10: ${ps[9][1].toFixed(4)}`);
  },
};

// ---------- first-order vs second-order stationary points ----------
const fl = (x, y) => .5 * x * x + .25 * y ** 4 - .5 * y * y, gl = (x, y) => [x, y ** 3 - y];
const RHO = 9, CD = [-3.8, 3.8, -1.5, 1.5], FLEV = [-.2, -.1, 0, .15, .35, .6, .9, 1.3, 1.8, 2.4, 3.2, 4];
const sosp = {
  title: 'ε-FOSP or ε-SOSP?',
  caption: 'f = ½x² + ¼y⁴ − ½y² has a saddle at the origin and minima at (0, ±1). Amber: points with ‖∇f‖ ≤ ε (ε-first-order stationary). Green: the part that also has λ_min(∇²f) ≥ −√(ρε) (ε-second-order stationary). Make ε loose enough and the saddle sneaks in.',
  note: 'ρ = 9 is the Hessian-Lipschitz constant of f on |y| ≤ 1.5 (the drawn strip). Thin lines: contours of f.',
  legend: [[HEX.flat, 'ε-FOSP only'], [HEX.good, 'ε-SOSP'], [HEX.neg, 'saddle'], ['#ffffff', 'minima']],
  mode: '2d',
  plot: { xlabel: 'x', ylabel: 'y', xlim: [-1.5, 1.5], ylim: [CD[2], CD[3]], equal: true, legend: false, grid: false },
  controls: [{ id: 'le', label: 'tolerance ε', min: -2, max: -.3, step: .01, value: -1, fmt: v => (10 ** v).toFixed(3) }],
  build(ctx) { ctx.sets = contours(fl, CD, FLEV, 160); this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const eps = 10 ** ctx.vals.le, tol = Math.sqrt(RHO * eps), P = ctx.plot.clear(), n = 240, z = [];
    heat(P, fl, CD, { alpha: .45, n: 90 });
    drawContours(P, ctx.sets);
    for (let j = 0; j < n; j++) {
      const row = [], y = CD[2] + (CD[3] - CD[2]) * (j + .5) / n;
      for (let i = 0; i < n; i++) {
        const x = CD[0] + (CD[1] - CD[0]) * (i + .5) / n, gn = Math.hypot(...gl(x, y)), lmin = Math.min(1, 3 * y * y - 1);
        row.push(gn > eps ? 0 : lmin >= -tol ? 2 : 1);
      }
      z.push(row);
    }
    P.heat(z, CD, { cell: v => v === 2 ? 'rgba(95,224,176,.95)' : v === 1 ? 'rgba(255,200,87,.95)' : 'rgba(0,0,0,0)' });
    P.scatter([[0, 1], [0, -1]], { color: '#ffffff', r: 2.5 });
    P.scatter([[0, 0]], { color: HEX.neg, r: 3 });
    P.text(0, 0, 'saddle', { color: HEX.neg, dx: 16 });
    P.text(0, 1, 'minimum', { color: '#e8ecf8', dx: 16 }); P.text(0, -1, 'minimum', { color: '#e8ecf8', dx: 16 });
    P.draw();
    ctx.stat(`ε = ${eps.toFixed(3)} · curvature allowance −√(ρε) = −${tol.toFixed(3)} · saddle: ‖∇f‖ = 0, λ_min = −1 → ${-1 >= -tol ? 'counts as an ε-SOSP (ε too loose: needs ε < 1/ρ = 0.111)' : 'an ε-FOSP but not an ε-SOSP'}`);
  },
};

export default { zoo, sym, random, sosp };
