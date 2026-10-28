// Du et al. 2019 — wide nets train like their linearization. Two-layer softplus net trained live in the browser.
// Same model and settings as scripts/check-math/du2019gradient.py.
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng, eigSym } from '../../src/lib/math.js';

const sp = z => z > 30 ? z : Math.log1p(Math.exp(z)), sg = z => 1 / (1 + Math.exp(-z));
const TH = [.3, 2.4, 4.4], Y = [.8, -.5, .6];
const data = n => ({ X: TH.slice(0, n).map(t => [Math.cos(t) / Math.SQRT2, Math.sin(t) / Math.SQRT2, 1 / Math.SQRT2]), y: Y.slice(0, n) });

/** f(x) = (1/sqrt m) sum_r a_r softplus(w_r . x), all weights N(0,1), full-batch GD on 1/2 sum (f(x_i) - y_i)^2 */
function Net(m, D, seed) {
  const r = rng(seed), W = Array.from({ length: m }, () => [r.normal(), r.normal(), r.normal()]), a = Array.from({ length: m }, () => r.normal());
  const W0 = W.map(w => w.slice()), n = D.X.length, s = 1 / Math.sqrt(m);
  const z = () => D.X.map(x => W.map(w => w[0] * x[0] + w[1] * x[1] + w[2] * x[2]));
  const residual = () => { const Z = z(); return Z.map((zr, i) => zr.reduce((acc, v, k) => acc + a[k] * sp(v), 0) * s - D.y[i]); };
  const gram = () => {
    const Z = z(), G = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
      const xx = D.X[i][0] * D.X[j][0] + D.X[i][1] * D.X[j][1] + D.X[i][2] * D.X[j][2]; let v = 0;
      for (let k = 0; k < m; k++) v += a[k] * a[k] * sg(Z[i][k]) * sg(Z[j][k]) * xx + sp(Z[i][k]) * sp(Z[j][k]);
      G[i][j] = G[j][i] = v / m;
    }
    return G;
  };
  const step = eta => {
    const Z = z(), res = Z.map((zr, i) => zr.reduce((acc, v, k) => acc + a[k] * sp(v), 0) * s - D.y[i]);
    for (let k = 0; k < m; k++) {
      let ga = 0, g0 = 0, g1 = 0, g2 = 0;
      for (let i = 0; i < n; i++) { ga += sp(Z[i][k]) * res[i]; const c = a[k] * sg(Z[i][k]) * res[i]; g0 += c * D.X[i][0]; g1 += c * D.X[i][1]; g2 += c * D.X[i][2]; }
      a[k] -= eta * s * ga; W[k][0] -= eta * s * g0; W[k][1] -= eta * s * g1; W[k][2] -= eta * s * g2;
    }
  };
  const movement = () => { let d = 0, t = 0; W.forEach((w, k) => w.forEach((v, j) => { d += (v - W0[k][j]) ** 2; t += W0[k][j] ** 2; })); return Math.sqrt(d / t); };
  return { residual, gram, step, movement, m };
}
const lam = G => eigSym(G).values;
const matvec = (M, v) => M.map(row => row.reduce((s, x, j) => s + x * v[j], 0));

const bowl = def({ L: (u, v) => .5 * (u * u + v * v), g: (u, v) => [u, v], h: L => L * .55, dom: [-1.6, 1.6, -1.6, 1.6] });

export default {
  tabs: [
    {
      id: 'prediction', name: 'Prediction space', title: 'In prediction space the loss is a bowl',
      caption: 'Height = ½‖u − y‖², the training loss as a function of the network’s two predictions. The ball is a real network training; the white line is the linear prediction (I − ηG₀)ᵏr₀. Wide nets follow it; narrow ones drift off.',
      note: 'Two training points, two-layer softplus net, all weights trained, η = 1/λ_max(G₀). The non-convexity lives in weight space, not here.',
      legend: [[HEX.ball, 'actual residual u(k) − y'], ['#ffffff', 'linear prediction'], [HEX.good, 'zero loss']],
      action: 'New initialization', cam: [0, 4.3, 4.6], tgt: [0, .2, 0],
      controls: [{ id: 'm', type: 'seg', label: 'width m', value: 4096, options: [[4, '4'], [32, '32'], [512, '512'], [4096, '4096']] }],
      build(ctx) { const s = ctx.stage; s.surface(bowl); s.marker(bowl, 0, 0, C.good); ctx.seed = 2; this.drop(ctx); },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, D = data(2), net = Net(ctx.vals.m, D, ctx.seed++);
        s.clearBalls(); if (ctx.lin) s.remove(ctx.lin);
        const G0 = net.gram(), eta = 1 / lam(G0)[1], r0 = net.residual(), M = G0.map((row, i) => row.map((g, j) => (i === j ? 1 : 0) - eta * g));
        const lin = [r0]; for (let k = 1; k < 200; k++) lin.push(matvec(M, lin[k - 1]));
        ctx.lin = s.line(bowl, lin, C.white, .85, .02); ctx.linPts = lin; ctx.dev = 0;
        ctx.b = s.ball(bowl, { start: r0, speed: .5, maxSteps: 199, step: (p, b) => { net.step(eta); const q = net.residual(); const L = ctx.linPts[b.steps + 1]; ctx.dev = Math.max(ctx.dev, Math.hypot(q[0] - L[0], q[1] - L[1]) / Math.hypot(...r0)); return q; } });
        ctx.net = net;
      },
      frame(ctx) { const b = ctx.b; if (b) ctx.stat(`m = ${ctx.vals.m} · step ${b.steps} · loss ${b.loss().toExponential(2)} · max deviation from the linear prediction ${(100 * ctx.dev).toFixed(3)}% of ‖r₀‖`); },
    },
    {
      id: 'rate', name: 'Linear rate', mode: '2d', title: 'Geometric convergence, and a Gram matrix that barely moves',
      caption: 'Training loss for four widths (solid) against the theorem-shaped bound (1 − ηλ₀/2)ᵏ L(0) (dashed), with λ₀ the smallest eigenvalue of that network’s Gram matrix at initialization.',
      note: 'Three points on the unit sphere, η = 1/λ_max(G₀). The paper’s widths and step sizes are far more conservative; this checks the shape of the result.',
      hint: '', legend: [], action: 'New initialization',
      plot: { logy: true, xlabel: 'GD iteration k', ylabel: 'training loss', xlim: [0, 300], ylim: [1e-12, 2] },
      charts: [
        { id: 'g', title: 'λ_min(G_k) / λ_min(G_0)', xlabel: 'iteration k', ylabel: 'ratio', xlim: [0, 300], h: 200 },
        { id: 'mv', title: 'weight movement ‖W_k − W_0‖ / ‖W_0‖ after 300 steps', xlabel: 'width m', ylabel: 'relative change', logx: true, logy: true, h: 200, legend: false },
      ],
      build(ctx) { ctx.seed = ctx.seed || 0; this.compute(ctx); },
      drop(ctx) { ctx.seed++; this.compute(ctx); },
      compute(ctx) {
        const D = data(3), cols = [HEX.opt2, HEX.flat, HEX.pos, HEX.ball], widths = [8, 64, 1024, 8192], P = ctx.plot.clear(), gch = ctx.charts.g.clear(), mvp = [];
        let summary = [];
        widths.forEach((m, wi) => {
          const net = Net(m, D, 100 * ctx.seed + wi), G0 = net.gram(), ev = lam(G0), eta = 1 / ev[ev.length - 1], l0 = ev[0], loss = [], gr = [];
          for (let k = 0; k <= 300; k++) {
            const r = net.residual(); loss.push([k, Math.max(.5 * r.reduce((s, v) => s + v * v, 0), 1e-13)]);
            if (k % 15 === 0) gr.push([k, lam(net.gram())[0] / l0]);
            net.step(eta);
          }
          P.line(loss, { color: cols[wi], width: 2, label: `m = ${m}` });
          P.fn(k => loss[0][1] * (1 - eta * l0 / 2) ** k, { color: cols[wi], dash: [5, 5], width: 1.2, alpha: .8 });
          gch.line(gr, { color: cols[wi], width: 2, label: `m = ${m}` });
          mvp.push([m, net.movement()]);
          summary.push(`m=${m}: λ₀=${l0.toFixed(2)}`);
        });
        P.draw(); gch.draw();
        ctx.charts.mv.clear().line(mvp, { color: HEX.ball, width: 2, dots: 4 }).fn(m => mvp[0][1] * Math.sqrt(8 / m), { color: HEX.muted, dash: [4, 4] }).draw();
        ctx.stat(`${summary.join(' · ')} · movement shrinks ≈ 1/√m (grey)`);
      },
    },
  ],
};
