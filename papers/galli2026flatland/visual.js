// Galli et al. 2026 — large steps via nonmonotone equality line search, and the 2/K sharpness floor ("Flatland").
// Checked in scripts/check-math/galli2026flatland.py (same function, line search, constants and the flat-point spectrum).
import { def, C, HEX, segPts } from '../../src/lib/atlas3d.js';
import { rng, eigSym } from '../../src/lib/math.js';

const A = 6;
const f = (x, y) => .25 * (x * x - 1) ** 2 + .5 * A * (y - .3 * x * x) ** 2;
const g = (x, y) => { const r = y - .3 * x * x; return [x * (x * x - 1) - A * r * .6 * x, A * r]; };
const H = (x, y) => { const r = y - .3 * x * x; return [3 * x * x - 1 + A * .36 * x * x - A * r * .6, -A * .6 * x, A]; };
const lamMax = (x, y) => { const [a, b, d] = H(x, y); return (a + d) / 2 + Math.sqrt(((a - d) / 2) ** 2 + b * b); };
const valley = def({ L: f, g, hess: H, h: v => Math.log1p(v * 4) * .7, dom: [-1.9, 1.9, -1, 1.8] });

/** Algorithm 2: from the previous step, extrapolate (×2) while accepted, otherwise backtrack (×½) until accepted. */
function eqLS(p, eta0, R, c = 1e-4, beta = .5) {
  const [gx, gy] = g(...p), gg = gx * gx + gy * gy, ok = e => f(p[0] - e * gx, p[1] - e * gy) <= R - c * e * gg;
  let e = eta0;
  if (ok(e)) { while (ok(e / beta) && e < 1e3) e /= beta; } else { while (!ok(e) && e > 1e-12) e *= beta; }
  return e;
}
const lsStep = mode => (p, b) => {
  b.eta ??= 1; b.Q ??= 1; b.R ??= f(...p);
  const R = mode === 'LS' ? f(...p) : b.R;
  b.eta = eqLS(p, b.eta, R);
  b.log.push([b.steps, b.eta * lamMax(...p), f(...p)]);
  const [gx, gy] = g(...p), q = [p[0] - b.eta * gx, p[1] - b.eta * gy];
  if (mode === 'NLS') { const xi = .85, Qn = xi * b.Q + 1; b.R = (xi * b.Q * b.R + f(...q)) / Qn; b.Q = Qn; }
  return q;
};

// tiny tanh MSE classifier 2 -> 4 -> K (same as the check script)
function mlp(K, seed) {
  const r = rng(seed), X = Array.from({ length: 30 }, () => [r.normal(), r.normal()]), lab = X.map(() => Math.floor(r() * K)), h = 4, n = 2 * h + h + K * h + K;
  const loss = th => {
    let s = 0;
    for (let i = 0; i < X.length; i++) {
      const hid = []; for (let j = 0; j < h; j++) hid.push(Math.tanh(th[2 * j] * X[i][0] + th[2 * j + 1] * X[i][1] + th[2 * h + j]));
      for (let k = 0; k < K; k++) { let z = th[3 * h + K * h + k]; for (let j = 0; j < h; j++) z += th[3 * h + k * h + j] * hid[j]; s += (z - (lab[i] === k ? 1 : 0)) ** 2; }
    }
    return s / (X.length * K);
  };
  const hess = th => {
    const e = 1e-3, M = Array.from({ length: n }, () => new Array(n).fill(0)), f0 = loss(th), t = th.slice();
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
      const at = (di, dj) => { t[i] += di; t[j] += dj; const v = loss(t); t[i] -= di; t[j] -= dj; return v; };
      M[i][j] = M[j][i] = i === j ? (at(e, 0) - 2 * f0 + at(-e, 0)) / (e * e) : (at(e, e) - at(e, -e) - at(-e, e) + at(-e, -e)) / (4 * e * e);
    }
    return M;
  };
  const grad = th => { const t = th.slice(), e = 1e-6; return th.map((_, i) => { t[i] += e; const a = loss(t); t[i] -= 2 * e; const b = loss(t); t[i] += e; return (a - b) / (2 * e); }); };
  const means = Array.from({ length: K }, (_, k) => lab.filter(l => l === k).length / lab.length);
  return { n, loss, hess, grad, means, K, h, r };
}

export default {
  tabs: [
    {
      id: 'linesearch', name: 'Large steps', title: 'Accepting a bigger loss now buys bigger steps',
      caption: 'Both balls use an equality line search (grow the step while it is accepted). White: monotone, every step must lower f. Coral: nonmonotone, a step only has to beat a running average R_k, so it can bounce across the valley — at η·λ_max ≈ 2, the edge of stability — and still converge.',
      note: 'f = ¼(x² − 1)² + 3(y − 0.3x²)². Acceptance f(w − η∇f) ≤ R − cη‖∇f‖², c = 10⁻⁴; Zhang–Hager R with ξ = 0.85.',
      legend: [[HEX.opt3, 'monotone line search'], [HEX.ball, 'nonmonotone line search'], [HEX.good, 'minima']],
      action: 'Restart', cam: [0, 4.2, 4.6], tgt: [0, .3, 0],
      charts: [
        { id: 'el', title: 'η_k · λ_max(∇²f(w_k))', xlabel: 'iteration k', ylabel: 'η·λ_max', ylim: [0, 6], h: 210 },
        { id: 'f', title: 'f(w_k)', xlabel: 'iteration k', ylabel: 'f', logy: true, ylim: [1e-14, 20], h: 180, legend: false },
      ],
      build(ctx) {
        const s = ctx.stage; s.surface(valley);
        s.marker(valley, 1, .3, C.good); s.marker(valley, -1, .3, C.good);
        const v = []; for (let x = -1.9; x <= 1.9; x += .02) v.push([x, .3 * x * x]); s.line(valley, v, C.white, .35);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        ctx.ls = s.ball(valley, { start: [-1.6, 1.4], step: lsStep('LS'), color: C.opt3, speed: .2, maxSteps: 400, stop: b => Math.hypot(...g(...b.p)) < 1e-9 }); ctx.ls.log = [];
        ctx.nl = s.ball(valley, { start: [-1.6, 1.4], step: lsStep('NLS'), speed: .2, maxSteps: 400, stop: b => Math.hypot(...g(...b.p)) < 1e-9 }); ctx.nl.log = [];
      },
      frame(ctx) {
        const A = ctx.ls.log, B = ctx.nl.log, xm = Math.max(20, A.length, B.length);
        ctx.charts.el.clear().set({ xlim: [0, xm] }).hline(2, { color: HEX.neg, dash: [5, 4], text: 'η·λ = 2' })
          .line(A.map(r => [r[0], r[1]]), { color: HEX.opt3, width: 1.8, dots: 2, label: 'monotone' }).line(B.map(r => [r[0], r[1]]), { color: HEX.ball, width: 1.8, dots: 2, label: 'nonmonotone' }).draw();
        ctx.charts.f.clear().set({ xlim: [0, xm] }).line(A.map(r => [r[0], Math.max(r[2], 1e-14)]), { color: HEX.opt3, width: 1.8 }).line(B.map(r => [r[0], Math.max(r[2], 1e-14)]), { color: HEX.ball, width: 1.8 }).draw();
        const ups = B.filter((r, i) => i && r[2] > B[i - 1][2]).length;
        ctx.stat(`nonmonotone: ${B.length} steps, ${ups} accepted loss increases, median η·λ_max = ${B.length ? [...B.map(r => r[1])].sort((a, b) => a - b)[B.length >> 1].toFixed(2) : '–'} · monotone: ${A.length} steps`);
      },
    },
    {
      id: 'flatland', name: 'Flatland', mode: '2d', title: 'The flattest place is a dead end',
      caption: 'Top Hessian eigenvalues of a tiny MSE classifier with K outputs. Sharpness can never drop below 2/K (red line). At the “flat” point — hidden layer silent, output = class frequencies — it equals 2/K exactly, K times over, the gradient is zero, and there are negative eigenvalues: a maximally flat saddle.',
      note: 'Net 2 → 4 (tanh) → K, MSE averaged over K outputs, 30 random points with random labels. Hessian by finite differences, live.',
      hint: '', legend: [],
      controls: [{ id: 'K', type: 'seg', label: 'classes K', value: 4, options: [[2, '2'], [4, '4'], [8, '8']] }, { id: 'pt', type: 'seg', label: 'point', value: 'flat', options: [['random', 'random weights'], ['trained', 'after GD'], ['flat', 'flat point']] }],
      plot: { xlabel: 'eigenvalue rank (largest first)', ylabel: 'Hessian eigenvalue', legend: false },
      build(ctx) { this.compute(ctx); }, onControl(ctx) { this.compute(ctx); },
      compute(ctx) {
        const K = ctx.vals.K, net = mlp(K, 7), n = net.n;
        let th;
        if (ctx.vals.pt === 'flat') { th = new Array(n).fill(0); for (let k = 0; k < K; k++) th[n - K + k] = net.means[k]; }
        else { const r = rng(3); th = Array.from({ length: n }, () => 1.5 * r.normal()); }
        if (ctx.vals.pt === 'trained') for (let t = 0; t < 400; t++) { const gr = net.grad(th); th = th.map((v, i) => v - .5 * gr[i]); }
        const ev = eigSym(net.hess(th), { sweeps: 40 }).values.sort((a, b) => b - a), top = ev, neg = ev.filter(v => v < -1e-6);
        const P = ctx.plot.clear().set({ xlim: [-.6, top.length - .4], ylim: [Math.min(-.15, ...top) * 1.1, Math.max(...top) * 1.12], xfmt: v => String(Math.round(v) + 1) });
        P.bars(top.map(v => ({ y: v, color: Math.abs(v - 2 / K) < 1e-3 ? HEX.flat : v < 0 ? HEX.neg : HEX.pos })), { width: .7 });
        P.hline(2 / K, { color: HEX.neg, dash: [5, 4], text: `2/K = ${(2 / K).toFixed(3)}` }).hline(0, { color: HEX.muted });
        P.draw();
        const gn = Math.hypot(...net.grad(th));
        ctx.stat(`λ_max = ${ev[0].toFixed(4)} ≥ 2/K = ${(2 / K).toFixed(4)} · λ_min = ${ev[ev.length - 1].toFixed(3)} (${neg.length} negative) · ‖∇f‖ = ${gn.toExponential(1)} · loss ${net.loss(th).toFixed(4)}`);
      },
    },
  ],
};
