// Li, Wang, Li 2022 (survey key wang2022analyzing; the survey lists the authors as "Z. Wang, Z. Li, J. Li") —
// four phases of EoS and the output-layer norm |A|^2 as an indicator of sharpness. Eq. (2): d|A|^2/dt = -(4/n) D.F.
// Checked in scripts/check-math/wang2022analyzing.py (offline two-layer linear network in data.json).
import { def, C, HEX, hyperbola } from '../../src/lib/atlas3d.js';

// scalar two-layer linear net, x = 1, y = 1, L = |D|^2 / n with n = 1: L(a, w) = (aw - 1)^2; a is the output layer
const sc = def({ L: (a, w) => (a * w - 1) ** 2, g: (a, w) => [2 * (a * w - 1) * w, 2 * (a * w - 1) * a], hess: (a, w) => [2 * w * w, 4 * a * w - 2, 2 * a * a], dom: [-.4, 2.2, -.4, 2.2], h: v => Math.log1p(v) });
const top = (a, w) => { const [p, q, r] = sc.hess(a, w), m = (p + r) / 2, d = Math.hypot((p - r) / 2, q); return m + d; };
const phase = (S, dS, thr) => Math.abs(dS) < 1e-9 ? 'settled on a minimum' : S < thr ? (dS >= 0 ? 'I · progressive sharpening' : 'IV · back below 2/η') : (dS >= 0 ? 'II · above 2/η, still rising' : 'III · above 2/η, falling');
const DFcol = (a, w) => { const D = a * w - 1, s = D * (a * w); return s < 0 ? [.32, .55, .78] : [.62, .3, .38]; };

export default {
  tabs: [
    {
      id: 'sign', name: 'Output-layer norm', title: 'The output weight grows while the network undershoots, shrinks when it overshoots',
      caption: 'Loss (aw − 1)² of a one-unit linear network; a is the output layer. Eq. (2): under gradient flow ‖A‖² grows exactly where the residual D and the output F have opposite signs (blue: 0 < aw < 1) and shrinks where they agree (red). Starting small, GD first moves through blue, so a² and the sharpness rise together. When the sharpness passes 2/η the ball starts to bounce across the valley into red, where a² and the sharpness fall again.',
      note: 'Scalar case m = d = n = 1 (x = 1, label 1) of the paper’s Sec. 4 model, normalised as L = ‖D‖²/n. Sharpness = top Hessian eigenvalue at the current point; on the valley it equals 2(a² + w²) ≥ 4, so above η = 0.5 GD can never settle.',
      legend: [[HEX.ball, 'gradient descent'], [HEX.good, 'aw = 1 (minima)'], ['#528cc7', 'D·F < 0: ‖A‖² grows'], ['#9e4d61', 'D·F > 0: ‖A‖² shrinks']],
      action: 'Restart', cam: [3.4, 3.4, 4.2], tgt: [.3, .1, 0],
      controls: [{ id: 'eta', label: 'step size η', min: .2, max: .56, step: .01, value: .45 }],
      charts: [{ id: 'S', title: 'sharpness (blue) and a² (coral)', xlabel: 'step', ylabel: 'value', ylim: [0, 8], h: 230, legend: false }],
      build(ctx) {
        const s = ctx.stage; s.surface(sc, { colorFn: (a, w, t) => DFcol(a, w).map(c => c * (.55 + .45 * t)) });
        s.tube(sc, hyperbola(1, 2.2, 1), C.good); this.drop(ctx);
      },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, eta = ctx.vals.eta; s.clearBalls(); ctx.hist = [];
        ctx.b = s.ball(sc, { start: [.15, .1], speed: .4, maxSteps: 600, step: p => { const [ga, gw] = sc.g(p[0], p[1]), q = [p[0] - eta * ga, p[1] - eta * gw]; ctx.hist.push([ctx.hist.length + 1, top(q[0], q[1]), q[0] * q[0]]); return q; } });
      },
      frame(ctx) {
        const eta = ctx.vals.eta, H = ctx.hist, xl = [0, Math.max(60, H.length)];
        ctx.charts.S.clear().set({ xlim: xl }).hline(2 / eta, { color: HEX.neg, dash: [5, 4], text: '2/η' }).line(H.map(h => [h[0], h[1]]), { color: HEX.pos, width: 1.6 }).line(H.map(h => [h[0], h[2]]), { color: HEX.ball, width: 1.6 }).draw();
        if (H.length < 2) return;
        const [, S, a2] = H[H.length - 1], dS = S - H[H.length - 2][1], D = ctx.b.p[0] * ctx.b.p[1] - 1;
        ctx.stat(`step ${H.length} · ${phase(S, dS, 2 / eta)} · sharpness ${S.toFixed(3)} vs 2/η = ${(2 / eta).toFixed(3)} · a² = ${a2.toFixed(3)} · D·F ${D * (D + 1) < 0 ? '< 0' : '> 0'}`);
      },
    },
    {
      id: 'trace', name: 'Four phases', mode: '2d', title: 'In a real two-layer linear network, ‖A‖² tracks the sharpness',
      caption: 'Our offline replication: a 64-wide two-layer linear network fitting 32 labels ±1 with full-batch GD at a step whose threshold 2/η lies between the initial and the gradient-flow sharpness. The sharpness climbs past 2/η (I → II), overshoots and falls back (III → IV), keeps oscillating around 2/η while the loss drops non-monotonically, and finally converges just below the edge. The output-layer norm ‖A‖² rises and falls with it.',
      note: 'f(x) = AᵀWx/√m, n = 32, d = 48, m = 64, loss ‖D‖²/n, sharpness by power iteration on exact Hessian-vector products every step. Shading: steps with sharpness above 2/η. The paper’s Figs. 1–3 show the same on CIFAR-10 subsets (reported).',
      hint: '', legend: [[HEX.pos, 'sharpness'], [HEX.ball, '‖A‖² (rescaled)'], [HEX.neg, '2/η']],
      controls: [{ id: 'T', type: 'seg', label: 'window', value: 300, options: [[60, 'first 60'], [300, 'first 300'], [1200, 'all']] }],
      plot: { xlabel: 'GD step', ylabel: 'sharpness', legend: false },
      charts: [{ id: 'loss', title: 'loss ‖D‖²/n (log)', xlabel: 'GD step', ylabel: 'loss', logy: true, ylim: [1e-18, 10], h: 220, legend: false }],
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const D = ctx.data, T = +ctx.vals.T, R = D.rows.slice(0, T), thr = 2 / D.eta;
        const smax = Math.max(...R.map(r => r[2])), amin = Math.min(...R.map(r => r[3])), amax = Math.max(...R.map(r => r[3]));
        const resc = a => .3 + (smax - .3) * (a - amin) / (amax - amin || 1);
        const P = ctx.plot.clear().set({ xlim: [0, T], ylim: [.3, smax * 1.05] });
        let s0 = null; R.forEach((r, i) => { const up = r[2] > thr; if (up && s0 === null) s0 = r[0]; if ((!up || i === R.length - 1) && s0 !== null) { P.band(s0, r[0], { color: 'rgba(255,93,115,.10)' }); s0 = null; } });
        P.hline(thr, { color: HEX.neg, dash: [5, 4], text: `2/η = ${thr.toFixed(3)}` }).line(R.map(r => [r[0], resc(r[3])]), { color: HEX.ball, width: 1.4, alpha: .85 }).line(R.map(r => [r[0], r[2]]), { color: HEX.pos, width: 1.8 }).draw();
        ctx.charts.loss.clear().set({ xlim: [0, T] }).line(R.map(r => [r[0], Math.max(r[1], 1e-18)]), { color: HEX.opt3, width: 1.4 }).draw();
        const dS = [], dA = []; for (let i = 0; i + 10 < 300; i++) { dS.push(D.rows[i + 10][2] - D.rows[i][2]); dA.push(D.rows[i + 10][3] - D.rows[i][3]); }
        const agree = dS.filter((v, i) => Math.sign(v) === Math.sign(dA[i])).length / dS.length;
        ctx.stat(`η = ${D.eta} · sharpness ${D.rows[0][2].toFixed(2)} → peak ${Math.max(...D.rows.map(r => r[2])).toFixed(2)} → final ${D.rows[D.rows.length - 1][2].toFixed(3)} (2/η = ${thr.toFixed(3)}) · ‖A‖² and sharpness move together in ${(100 * agree).toFixed(0)}% of sliding 10-step windows (first 300 steps)`);
      },
    },
  ],
};
