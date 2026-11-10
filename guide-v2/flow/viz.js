// Chapter 12 (survey Section IV-E). New visual 'centralflow'; every number is checked in scripts/check-math/guide_flow.py.
// Toy: L(x, y, z) = (2/eta + y) x^2/2 + V(y, z), V = k (y - m z)^2/2 + e (z - z0)^2/2, eta = 1.
// x: the oscillation direction (curvature S = 2/eta + y); y = 0 is the edge S = 2/eta.
import { HEX } from '../../src/lib/atlas3d.js';
import { contours, drawContours } from '../../src/learn/plot2d.js';

const ETA = 1, K = .05, E = .005, M = .5, Z0 = 1, X0 = 1e-2, Y0 = -.3, ZS = -.5, T = 3000;
const V = (y, z) => K * (y - M * z) ** 2 / 2 + E * (z - Z0) ** 2 / 2;
const dV = (y, z) => [K * (y - M * z), E * (z - Z0) - K * M * (y - M * z)];
const Lf = (x, y, z) => (2 / ETA + y) * x * x / 2 + V(y, z);

function simulate() {
  const gd = [[X0, Y0, ZS]];
  let [x, y, z] = gd[0];
  for (let t = 0; t < T; t++) { const d = dV(y, z); [x, y, z] = [x - ETA * (2 / ETA + y) * x, y - ETA * (x * x / 2 + d[0]), z - ETA * d[1]]; gd.push([x, y, z]); }
  const rk4 = (f, q, h) => { const a = f(q), b = f([q[0] + h / 2 * a[0], q[1] + h / 2 * a[1]]), c = f([q[0] + h / 2 * b[0], q[1] + h / 2 * b[1]]), d = f([q[0] + h * c[0], q[1] + h * c[1]]); return [q[0] + h / 6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0]), q[1] + h / 6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1])]; };
  const gfF = q => { const d = dV(q[0], q[1]); return [-ETA * d[0], -ETA * d[1]]; };
  const cfF = q => { const v = gfF(q); return q[0] >= -1e-12 && v[0] > 0 ? [0, v[1]] : v; };   // on the edge and pushed up: project out e_y
  const gf = [[Y0, ZS]], cf = [[Y0, ZS]];
  let q1 = gf[0].slice(), q2 = cf[0].slice();
  for (let t = 0; t < T; t++) {
    for (let s = 0; s < 10; s++) { q1 = rk4(gfF, q1, .1); q2 = rk4(cfF, q2, .1); if (q2[0] > 0) q2[0] = 0; }
    gf.push(q1.slice()); cf.push(q2.slice());
  }
  return { gd, gf, cf };
}
const SIM = simulate();
const ZEDGE = E * Z0 / (E + K * M * M), SIG2 = 2 * K * M * ZEDGE;

const centralflow = {
  title: 'Gradient descent at the edge follows the central flow, not the gradient flow',
  caption: 'Seen from above, the two slow coordinates of a three-coordinate toy loss. The loss pulls towards a point in the too-sharp region (y > 0). Gradient flow (violet) walks straight in. Gradient descent (coral dots) cannot: it reaches the edge y = 0, oscillates, and slides along the edge, where the central flow (white) predicts it.',
  note: 'L = (2/η + y)x²/2 + 0.05(y − z/2)²/2 + 0.005(z − 1)²/2 with η = 1; x, the oscillation direction, is not drawn (its curvature is S = 2/η + y). GD: exact, 3000 steps from (x, y, z) = (0.01, −0.3, −0.5). Flows: RK4, 10 substeps per step.',
  legend: [[HEX.opt2, 'gradient flow'], ['#f2f4ff', 'central flow (dashed)'], [HEX.ball, 'gradient descent'], [HEX.flat, 'edge S = 2/η'], [HEX.neg, 'too sharp for this η']],
  mode: '2d',
  plot: { xlabel: 'z (progress direction)', ylabel: 'y (S − 2/η, how sharp)', xlim: [-.6, 1.1], ylim: [-.35, .55], legend: false },
  actions: [{ label: 'Replay', fn: ctx => { ctx.k = 0; } }],
  controls: [{ id: 'sp', type: 'seg', label: 'speed', value: 15, options: [[5, 'slow'], [15, 'normal'], [60, 'fast']] }],
  charts: [
    { id: 's', title: 'sharpness excess S_t − 2/η', xlabel: 'step t', ylabel: 'S − 2/η', xlim: [0, T], ylim: [-.32, .55], h: 190, legend: false },
    { id: 'l', title: 'loss (log)', xlabel: 'step t', ylabel: 'loss', logy: true, xlim: [0, T], ylim: [1e-4, 1], h: 190, legend: false },
    { id: 'x', title: 'GD oscillation x_t² and the central flow’s σ²', xlabel: 'step t', ylabel: 'x²', logy: true, xlim: [0, T], ylim: [1e-6, 1], h: 190, legend: false },
  ],
  build(ctx) {
    ctx.cont = contours((z, y) => V(y, z), [-.6, 1.1, -.35, .55], [.0005, .001, .002, .004, .007, .011, .016, .022, .03], 80);
    ctx.k = 0; this.draw(ctx);
  },
  frame(ctx) { if (ctx.k < T) { ctx.k = Math.min(T, ctx.k + (+ctx.vals.sp)); this.draw(ctx); } },
  onControl() {},
  draw(ctx) {
    const k = ctx.k, { gd, gf, cf } = SIM, P = ctx.plot.clear();
    P.custom((g, p) => { g.fillStyle = 'rgba(255,93,115,.08)'; g.fillRect(p.sx(-.6), p.sy(.55), p.sx(1.1) - p.sx(-.6), p.sy(0) - p.sy(.55)); });
    drawContours(P, ctx.cont);
    P.hline(0, { color: HEX.flat, width: 2, text: 'edge: S = 2/η' });
    P.text(-.55, .45, 'too sharp for this step (S > 2/η)', { color: HEX.neg });
    // GD first (underneath), then the two flows on top; before the edge all three coincide
    P.scatter(gd.slice(0, k + 1).filter((_, i) => i % 3 === 0).map(q => [q[2], q[1]]), { color: HEX.ball, r: 2.2, alpha: .8 });
    P.line(gf.slice(0, k + 1).map(q => [q[1], q[0]]), { color: HEX.opt2, width: 2.2 });
    P.line(cf.slice(0, k + 1).map(q => [q[1], q[0]]), { color: '#f2f4ff', width: 3, dash: [7, 5] });
    P.scatter([[Z0, M * Z0]], { color: HEX.opt2, r: 6 }); P.text(Z0, M * Z0, 'min of V (too sharp) ', { color: HEX.opt2, align: 'right', dx: -8 });
    P.scatter([[ZEDGE, 0]], { color: '#f2f4ff', r: 6 }); P.text(ZEDGE, 0, ' best point on the edge', { color: '#f2f4ff', dx: 6, dy: -12 });
    P.draw();
    const S = ctx.charts.s.clear(), Lc = ctx.charts.l.clear(), X = ctx.charts.x.clear();
    S.hline(0, { color: HEX.flat, dash: [3, 4] });
    S.line(gf.slice(0, k + 1).map((q, t) => [t, q[0]]), { color: HEX.opt2, width: 1.8 });
    S.line(cf.slice(0, k + 1).map((q, t) => [t, q[0]]), { color: '#f2f4ff', width: 1.8 });
    S.line(gd.slice(0, k + 1).map((q, t) => [t, q[1]]), { color: HEX.ball, width: 1.1 });
    Lc.line(gf.slice(0, k + 1).map((q, t) => [t, Math.max(1e-9, V(q[0], q[1]))]), { color: HEX.opt2, width: 1.8 });
    Lc.line(cf.slice(0, k + 1).map((q, t) => [t, V(q[0], q[1])]), { color: '#f2f4ff', width: 1.8 });
    Lc.line(gd.slice(0, k + 1).map((q, t) => [t, Lf(q[0], q[1], q[2])]), { color: HEX.ball, width: 1 });
    X.hline(SIG2, { color: '#f2f4ff', dash: [4, 4], text: `σ² = ${SIG2.toFixed(4)}` });
    X.line(gd.slice(0, k + 1).map((q, t) => [t, Math.max(1e-12, q[0] * q[0])]), { color: HEX.ball, width: 1 });
    S.draw(); Lc.draw(); X.draw();
    const g = gd[k], c = cf[k], f = gf[k];
    ctx.stat(`step ${k} · sharpness excess S − 2/η: GD ${g[1] >= 0 ? '+' : ''}${g[1].toFixed(3)}, central flow ${c[0] >= 0 ? '+' : ''}${c[0].toFixed(3)}, gradient flow ${f[0] >= 0 ? '+' : ''}${f[0].toFixed(3)} · progress z: GD ${g[2].toFixed(3)}, central flow ${c[1].toFixed(3)}, gradient flow ${f[1].toFixed(3)}${k === T ? ` · GD's x² averages ${(gd.slice(-400).reduce((a, q) => a + q[0] * q[0], 0) / 400).toFixed(5)} vs predicted σ² = ${SIG2.toFixed(5)}` : ''}`);
  },
};

export default { centralflow };
