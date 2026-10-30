// Gupta, Koren, Singer 2018 — Shampoo: precondition a matrix gradient from both sides, W ← W − η L^{-1/4} G R^{-1/4}.
// Lemmas 4 and 8, the toy runs and the matrix least-squares comparison are in scripts/check-math/gupta2018shampoo.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const EIG = [8, .2], WS = [.8, -.6], D0 = [-1.2, 1.1], ETA = .5, EPS = 1e-4;
const Q = th => [[Math.cos(th), -Math.sin(th)], [Math.sin(th), Math.cos(th)]];
const Amat = th => { const q = Q(th); return [0, 1].map(i => [0, 1].map(j => EIG[0] * q[i][0] * q[j][0] + EIG[1] * q[i][1] * q[j][1])); };
/** symmetric 2x2 matrix power via its eigen-decomposition */
function spow(M, p) {
  const a = M[0][0], b = M[0][1], c = M[1][1], m = (a + c) / 2, d = Math.hypot((a - c) / 2, b), l1 = m + d, l2 = m - d;
  let v = Math.abs(b) > 1e-14 ? [l1 - c, b] : (a >= c ? [1, 0] : [0, 1]); const n = Math.hypot(...v); v = [v[0] / n, v[1] / n];
  const u = [-v[1], v[0]], p1 = Math.max(l1, 1e-300) ** p, p2 = Math.max(l2, 1e-300) ** p;
  return [[p1 * v[0] * v[0] + p2 * u[0] * u[0], p1 * v[0] * v[1] + p2 * u[0] * u[1]], [p1 * v[1] * v[0] + p2 * u[1] * u[0], p1 * v[1] * v[1] + p2 * u[1] * u[1]]];
}
const mv = (M, x) => [M[0][0] * x[0] + M[0][1] * x[1], M[1][0] * x[0] + M[1][1] * x[1]];
const COL = { sgd: HEX.muted, diag: HEX.opt2, shampoo: HEX.ball, full: HEX.opt3 };
const NAME = { sgd: 'GD', diag: 'diagonal AdaGrad', shampoo: 'Shampoo', full: 'full-matrix AdaGrad' };

function makeSurf(th) {
  const A = Amat(th);
  return def({ L: (x, y) => { const d = [x - WS[0], y - WS[1]], Ad = mv(A, d); return .5 * (d[0] * Ad[0] + d[1] * Ad[1]); }, g: (x, y) => mv(A, [x - WS[0], y - WS[1]]), dom: [-2.2, 2.2, -2.2, 2.2], h: v => Math.log1p(v) });
}
const steps = {
  diag: d => (p, b) => { const g = d.g(...p); b.D = b.D || [EPS, EPS]; b.D = [b.D[0] + g[0] * g[0], b.D[1] + g[1] * g[1]]; return [p[0] - ETA * g[0] / Math.sqrt(b.D[0]), p[1] - ETA * g[1] / Math.sqrt(b.D[1])]; },
  // W in R^{1x2}: L is a scalar, R is 2x2; update L^{-1/4} g R^{-1/4}
  shampoo: d => (p, b) => { const g = d.g(...p); b.L = (b.L ?? EPS) + g[0] * g[0] + g[1] * g[1]; b.R = b.R || [[EPS, 0], [0, EPS]]; b.R = [[b.R[0][0] + g[0] * g[0], b.R[0][1] + g[0] * g[1]], [b.R[1][0] + g[1] * g[0], b.R[1][1] + g[1] * g[1]]]; const s = mv(spow(b.R, -.25), g), l = b.L ** -.25; return [p[0] - ETA * l * s[0], p[1] - ETA * l * s[1]]; },
  full: d => (p, b) => { const g = d.g(...p); b.H = b.H || [[EPS, 0], [0, EPS]]; b.H = [[b.H[0][0] + g[0] * g[0], b.H[0][1] + g[0] * g[1]], [b.H[1][0] + g[1] * g[0], b.H[1][1] + g[1] * g[1]]]; const s = mv(spow(b.H, -.5), g); return [p[0] - ETA * s[0], p[1] - ETA * s[1]]; },
};

export default {
  tabs: [
    {
      id: 'rotate', name: 'Rotated valley', title: 'Diagonal scaling cares about the axes; Shampoo does not',
      caption: 'A 1 × 2 weight matrix on a narrow quadratic valley (curvatures 8 and 0.2). Rotate the whole problem: diagonal AdaGrad (violet) rescales each coordinate separately, so it is fast when the valley is axis-aligned and slow when it is tilted. Shampoo (coral) accumulates R = Σ gᵀg as a full 2 × 2 matrix and applies R^{-1/4} (and the scalar L^{-1/4}), so its run is the same at every angle, as is full-matrix AdaGrad (white).',
      note: 'Deterministic gradients, step 0.5 for all, ε = 10⁻⁴; the start is the same point in the valley’s own axes for every angle. For a 1 × 2 matrix Shampoo is a full-matrix method with power −1/4 instead of −1/2; its memory advantage appears for real matrices (next tab).',
      legend: [[HEX.opt2, 'diagonal AdaGrad'], [HEX.ball, 'Shampoo'], [HEX.opt3, 'full-matrix AdaGrad'], [HEX.good, 'minimum']],
      action: 'Restart', cam: [0, 4.6, 4.2], tgt: [0, .1, 0],
      controls: [{ id: 'deg', label: 'valley rotation (degrees)', min: 0, max: 90, step: 1, value: 45, fmt: v => `${v}°` }],
      charts: [{ id: 'loss', title: 'loss (log)', xlabel: 'step', ylabel: 'f', logy: true, ylim: [1e-16, 100], xlim: [0, 300], h: 230, legend: false }],
      build(ctx) { ctx.d = makeSurf(ctx.vals.deg * Math.PI / 180); ctx.stage.surface(ctx.d); ctx.stage.marker(ctx.d, ...WS, C.good, .1); this.drop(ctx); },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      drop(ctx) {
        const s = ctx.stage, d = ctx.d, q = Q(ctx.vals.deg * Math.PI / 180), st = [WS[0] + q[0][0] * D0[0] + q[0][1] * D0[1], WS[1] + q[1][0] * D0[0] + q[1][1] * D0[1]];
        s.clearBalls(); ctx.H = { diag: [], shampoo: [], full: [] }; ctx.b = {};
        for (const [k, col] of [['diag', C.opt2], ['full', C.opt3], ['shampoo', C.ball]]) {
          const b = ctx.b[k] = s.ball(d, { start: st, speed: .6, maxSteps: 300, step: steps[k](d), color: col });
          b.onStep = bb => ctx.H[k].push([bb.steps, d.L(...bb.p)]);
        }
      },
      frame(ctx) {
        const P = ctx.charts.loss.clear(); for (const k of ['diag', 'full', 'shampoo']) P.line(ctx.H[k].map(h => [h[0], Math.max(h[1], 1e-16)]), { color: COL[k], width: 1.6 }); P.draw();
        const last = k => (ctx.H[k][ctx.H[k].length - 1]?.[1] ?? NaN).toExponential(1);
        ctx.stat(`${ctx.vals.deg}° · step ${ctx.b.shampoo.steps} · loss: diagonal AdaGrad ${last('diag')} · Shampoo ${last('shampoo')} · full AdaGrad ${last('full')}`);
      },
    },
    {
      id: 'matrix', name: 'Matrix problem', mode: '2d', title: 'Two small preconditioners instead of one huge one',
      caption: 'Matrix least squares ½‖AWB − C‖² with an 8 × 8 weight W and ill-conditioned A and B, so the curvature is a Kronecker product. Each method is run with its best step from a small grid. Shampoo’s left and right statistics (8² + 8² = 128 numbers) capture that structure and converge far faster than GD or diagonal AdaGrad; full-matrix AdaGrad needs a 64 × 64 = 4096-number preconditioner.',
      note: 'Offline, 1500 deterministic steps, ε = 10⁻⁶. Side chart: preconditioner memory. Theorem 7 (online convex regret O(√T)) is the paper’s guarantee; this toy only illustrates the structure.',
      hint: '', legend: [[HEX.muted, 'GD'], [HEX.opt2, 'diagonal AdaGrad'], [HEX.ball, 'Shampoo'], [HEX.opt3, 'full-matrix AdaGrad']],
      plot: { xlabel: 'step', ylabel: 'loss (log)', logy: true, ylim: [1e-20, 1e3], xlim: [0, 1500], legend: false },
      charts: [{ id: 'mem', title: 'preconditioner memory (numbers)', xlabel: '', ylabel: 'count (log)', logy: true, ylim: [10, 1e4], h: 220, legend: false, xfmt: () => '' }],
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear();
        for (const k of ['sgd', 'diag', 'full', 'shampoo']) P.line(D.curves[k].loss.map((v, i) => [5 * i + 1, Math.max(v, 1e-20)]), { color: COL[k], width: 2 });
        P.draw();
        const ks = ['diag', 'shampoo', 'full'], M = ctx.charts.mem.clear().bars(ks.map(k => ({ y: D.mem[k], color: COL[k] })));
        ks.forEach((k, i) => M.text(i, D.mem[k] * 1.5, String(D.mem[k]), { color: HEX.white, align: 'center' }));
        M.draw();
        const fin = k => D.curves[k].loss[D.curves[k].loss.length - 1].toExponential(1);
        ctx.stat(`final loss (best step): GD ${fin('sgd')} (η ${D.curves.sgd.eta}) · diagonal AdaGrad ${fin('diag')} (${D.curves.diag.eta}) · Shampoo ${fin('shampoo')} (${D.curves.shampoo.eta}) · full AdaGrad ${fin('full')} (${D.curves.full.eta})`);
      },
    },
  ],
};
