// Cohen, Kaur, Li, Kolter, Talwalkar 2021 — full-batch GD typically trains at the edge of stability (sharpness ≈ 2/η).
// Thresholds, the Fig. 2 quadratic and the offline MLP replication (data.json) are checked in scripts/check-math/cohen2021gradient.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const A1 = 20, A2 = 1;   // Fig. 2 of the paper: eigenvalues 20 and 1
const quad = def({ L: (x, y) => .5 * (A1 * x * x + A2 * y * y), g: (x, y) => [A1 * x, A2 * y], hess: () => [A1, 0, A2], dom: [-.42, .42, -1.6, 1.6], square: true, h: L => Math.log1p(L) });
const ETA_COL = { '0.1': HEX.pos, '0.3': HEX.ball, '0.36': HEX.opt2, '0.42': HEX.opt3 };

/** Spectral radius of heavy-ball / Nesterov on a 1D quadratic with curvature a (2x2 companion matrix). */
function rho(a, eta, beta, nest) {
  const m = nest ? [[beta - eta * a * beta, -eta * a], [beta - eta * a * beta, 1 - eta * a]] : [[beta, -eta * a], [beta, 1 - eta * a]];
  const tr = m[0][0] + m[1][1], det = m[0][0] * m[1][1] - m[0][1] * m[1][0], disc = tr * tr - 4 * det;
  if (disc >= 0) { const s = Math.sqrt(disc); return Math.max(Math.abs((tr + s) / 2), Math.abs((tr - s) / 2)); }
  return Math.sqrt(det);
}
/** Largest curvature a with spectral radius < 1, by bisection (checks Eq. 1 numerically). */
function mssNumeric(eta, beta, nest) { let lo = 1e-6, hi = 200 / eta; for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (rho(m, eta, beta, nest) < 1) lo = m; else hi = m; } return lo; }

export default {
  tabs: [
    {
      id: 'quad', name: 'Why 2/η', title: 'On a quadratic, GD is stable only while sharpness < 2/η',
      caption: 'GD on f = ½(20x² + y²), the paper’s Fig. 2 example. Along the sharp axis the error is multiplied by (1 − 20η) each step: below η = 0.1 it shrinks, above it flips sign and grows, so GD diverges even though the flat axis still converges.',
      note: 'Exact quadratic: the ball runs plain GD. The surface height is log(1 + f); the ball stops at the edge of the drawn box once it diverges.',
      legend: [[HEX.ball, 'gradient descent'], [HEX.pos, 'eigen-directions (λ = 20, 1)']],
      action: 'Restart', cam: [3.2, 3.4, 3.6], tgt: [0, .2, 0],
      controls: [{ id: 'eta', label: 'step size η', min: .04, max: .12, step: .002, value: .108, digits: 3 }],
      charts: [{ id: 'x', title: '|x_t| along the sharp axis (log)', xlabel: 'step', ylabel: '|x|', logy: true, ylim: [1e-6, 10], h: 230, legend: false }],
      build(ctx) {
        const s = ctx.stage; s.surface(quad); s.eigenLines(quad, 0, 0, { len: .4, fmtL: l => l.toFixed(0) }); this.drop(ctx);
      },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, eta = ctx.vals.eta; s.clearBalls(); ctx.hist = [];
        ctx.b = s.ball(quad, { start: [.02, 1.4], speed: .25, maxSteps: 200, step: (p) => { const q = [p[0] - eta * A1 * p[0], p[1] - eta * A2 * p[1]]; ctx.hist.push([ctx.hist.length + 1, Math.abs(q[0])]); return q; }, stop: b => Math.abs(b.p[0]) >= .419 });
      },
      frame(ctx) {
        const eta = ctx.vals.eta, f = 1 - eta * A1, H = ctx.hist;
        ctx.charts.x.clear().set({ xlim: [0, Math.max(40, H.length)] }).fn(t => .02 * Math.abs(f) ** t, { color: HEX.muted, dash: [4, 4] })
          .line([[0, .02], ...H], { color: HEX.ball, width: 1.8, dots: 1.6 }).draw();
        ctx.stat(`η = ${eta.toFixed(3)} · sharpness 20 vs 2/η = ${(2 / eta).toFixed(2)} · factor 1 − 20η = ${f.toFixed(2)} → ${Math.abs(f) < 1 ? 'converges' : 'oscillates and diverges'} · step ${ctx.b.steps}`);
      },
    },
    {
      id: 'mlp', name: 'Edge of stability', mode: '2d', title: 'Sharpness rises until it reaches 2/η, then stays there',
      caption: 'Our offline replication: a tanh network trained by full-batch GD at four step sizes, sharpness (top Hessian eigenvalue) measured by power iteration. At η = 0.1 it rises and saturates below 2/η; at the three larger steps progressive sharpening carries it to 2/η (dashed), it overshoots once and then hovers right at the line while the loss keeps falling.',
      note: 'tanh MLP 10–48–48–1, 128 synthetic regression points, MSE, 3000 steps, sharpness every 10 steps (8 warm-started power iterations on exact-gradient HVPs). The paper’s Figs. 1, 3–5 show the same on CIFAR-10 (reported, not reproduced).',
      hint: '', legend: [[HEX.pos, 'η = 0.1'], [HEX.ball, 'η = 0.3'], [HEX.opt2, 'η = 0.36'], [HEX.opt3, 'η = 0.42']],
      plot: { xlabel: 'GD iteration', ylabel: 'sharpness λ_max(∇²L)', xlim: [0, 3000], ylim: [3.5, 8.5], legend: false },
      charts: [{ id: 'loss', title: 'train loss (log)', xlabel: 'GD iteration', ylabel: 'loss', logy: true, ylim: [1e-7, 1], h: 260, legend: false }],
      build(ctx) {
        const D = ctx.data.curves, P = ctx.plot.clear(), lc = ctx.charts.loss.clear();
        for (const [k, rows] of Object.entries(D)) {
          const col = ETA_COL[k], e = +k;
          if (2 / e < 8.5) P.hline(2 / e, { color: col, dash: [5, 4], alpha: .7, text: `2/η = ${(2 / e).toFixed(2)}` });
          P.line(rows.map(r => [r[0], r[2]]), { color: col, width: 1.8 });
          lc.line(rows.map(r => [r[0], r[1]]), { color: col, width: 1.5 });
        }
        P.draw(); lc.draw();
        const late = k => { const r = D[k].slice(-100); return r.reduce((s, x) => s + x[2], 0) / r.length; };
        ctx.stat(`late sharpness vs 2/η: ${['0.3', '0.36', '0.42'].map(k => `${late(k).toFixed(2)} vs ${(2 / +k).toFixed(2)}`).join(' · ')} · η = 0.1 saturates at ${late('0.1').toFixed(2)} (2/η = 20)`);
      },
    },
    {
      id: 'mom', name: 'With momentum', mode: '2d', title: 'Momentum moves the edge to (2 + 2β)/η',
      caption: 'Maximum stable sharpness (MSS) on a quadratic, Eq. (1): heavy ball (Polyak) tolerates curvature up to (2 + 2β)/η, Nesterov only (2 + 2β)/(η(1 + 2β)). Dots are found numerically by bisecting the spectral radius of the two-step iteration; the paper reports that with momentum the sharpness plateaus at these values (Fig. 7).',
      note: 'Curves: Eq. (1) times η. Dots: largest curvature a with spectral radius < 1, computed live.',
      hint: '', legend: [[HEX.ball, 'heavy ball (Polyak)'], [HEX.opt2, 'Nesterov'], [HEX.muted, 'plain GD: 2']],
      controls: [{ id: 'beta', label: 'momentum β', min: 0, max: .99, step: .01, value: .9 }],
      plot: { xlabel: 'momentum β', ylabel: 'η × MSS', xlim: [0, 1], ylim: [0, 4.2], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const b = ctx.vals.beta, eta = .01, P = ctx.plot.clear();
        P.hline(2, { color: HEX.muted, dash: [4, 4], text: 'GD: 2' })
          .fn(x => 2 + 2 * x, { color: HEX.ball, width: 2 }).fn(x => (2 + 2 * x) / (1 + 2 * x), { color: HEX.opt2, width: 2 });
        const pts = []; for (let x = 0; x <= .99; x += .09) pts.push(x);
        P.scatter(pts.map(x => [x, eta * mssNumeric(eta, x, false)]), { color: HEX.ball, r: 3.5 }).scatter(pts.map(x => [x, eta * mssNumeric(eta, x, true)]), { color: HEX.opt2, r: 3.5 });
        P.vline(b, { color: HEX.flat, dash: [2, 3] }).draw();
        const hb = mssNumeric(eta, b, false), ne = mssNumeric(eta, b, true);
        ctx.stat(`β = ${b.toFixed(2)}, η = 0.01: heavy ball MSS = ${hb.toFixed(1)} (formula ${((2 + 2 * b) / eta).toFixed(1)}) · Nesterov MSS = ${ne.toFixed(1)} (formula ${((2 + 2 * b) / (eta * (1 + 2 * b))).toFixed(1)}) · GD 2/η = 200`);
      },
    },
  ],
};
