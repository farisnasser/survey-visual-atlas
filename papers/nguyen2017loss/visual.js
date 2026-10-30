// Nguyen & Hein 2017 — a wide layer (n_k ≥ N − 1) makes regular critical points global minima.
// Lemma 3.5's mechanism on the notes' 3-sample example, and an empirical width sweep (scripts/check-math/nguyen2017loss.py, data.json).
import { C, HEX } from '../../src/lib/atlas3d.js';

const THREE = window.THREE;
const X = [-1, 0, 1], Y = [1, 0, 2], S = 1.3;
const W = (h1, h2, y) => new THREE.Vector3(h1 * S, y * .9 + .3, -h2 * S);
function lstsq(F, y) {   // 3x3 normal equations with a tiny ridge so singular cases still return the least-squares fit
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  for (let i = 0; i < F.length; i++) for (let j = 0; j < 3; j++) { b[j] += F[i][j] * y[i]; for (let k = 0; k < 3; k++) A[j][k] += F[i][j] * F[i][k]; }
  for (let j = 0; j < 3; j++) A[j][j] += 1e-10;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < 3; c++) { let p = c; for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]]; for (let r = 0; r < 3; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k]; } }
  return M.map((r, i) => r[3] / r[i]);
}
const det3 = F => F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1]) - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0]) + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);

export default {
  tabs: [
    {
      id: 'rank', name: 'Feature rank', title: 'Independent features leave a stationary readout no room for error',
      caption: 'Three samples x = −1, 0, 1 through two tanh units: h₁ = tanh(x), h₂ = tanh(w x + c). Each dot is a sample at (h₁, h₂, target y). The stationary readout is the least-squares plane. If the three feature points are not collinear, [F, 1] has full rank and the plane passes through every target: zero training error. Make them collinear and the residuals (red) cannot vanish.',
      note: 'Linear readout only (Lemma 3.5’s mechanism: ∇_v L = [F, 1]ᵀr = 0 with full row rank forces r = 0). The paper proves it for deep nets with analytic activations and n_k ≥ N − 1 = 2 here.',
      legend: [[HEX.good, 'sample (h₁, h₂, y)'], [HEX.opt2, 'least-squares readout plane'], [HEX.neg, 'residual'], [HEX.muted, 'feature point in the (h₁, h₂) floor']],
      cam: [3.2, 2.6, 3.6], tgt: [0, .9, 0],
      controls: [{ id: 'w', label: 'weight w of unit 2', min: -3, max: 3, step: .05, value: 2 }, { id: 'c', label: 'bias c of unit 2', min: -3, max: 3, step: .05, value: 1 }],
      build(ctx) {
        const s = ctx.stage, { w, c } = ctx.vals;
        const F = X.map(x => [Math.tanh(x), Math.tanh(w * x + c), 1]), v = lstsq(F, Y), r = F.map((f, i) => f[0] * v[0] + f[1] * v[1] + v[2] - Y[i]), dt = det3(F);
        for (let i = -1; i <= 1; i += .5) { s.segment(W(i, -1, 0), W(i, 1, 0), C.muted, .25); s.segment(W(-1, i, 0), W(1, i, 0), C.muted, .25); }
        const lim = 1.05, P = (a, b) => W(a, b, Math.max(-1.2, Math.min(3, v[0] * a + v[1] * b + v[2])));
        const g = new THREE.BufferGeometry().setFromPoints([P(-lim, -lim), P(lim, -lim), P(lim, lim), P(-lim, -lim), P(lim, lim), P(-lim, lim)]);
        s.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: C.opt2, transparent: true, opacity: .25, side: THREE.DoubleSide, depthWrite: false })));
        F.forEach((f, i) => {
          s.dot(W(f[0], f[1], Y[i]), C.good, .07); s.dot(W(f[0], f[1], 0), C.muted, .04); s.segment(W(f[0], f[1], 0), W(f[0], f[1], Y[i]), C.muted, .4);
          if (Math.abs(r[i]) > 1e-6) s.segment(W(f[0], f[1], Y[i]), W(f[0], f[1], Y[i] + r[i]), C.neg);
          s.label(`x = ${X[i]}`, W(f[0], f[1], Y[i] + .25), '', HEX.good);
        });
        const rn = Math.hypot(...r);
        ctx.stat(`det [F, 1] = ${dt.toFixed(4)} · readout v = (${v.map(z => z.toFixed(3)).join(', ')}) · residual ‖r‖ = ${rn < 1e-6 ? '0' : rn.toFixed(4)}${Math.abs(dt) < 1e-3 ? ' · rank-deficient: features collinear' : ''}`);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
    },
    {
      id: 'width', name: 'Width sweep', mode: '2d', title: 'Narrow nets stall at bad critical points; from N − 1 units on, every run fits',
      caption: 'One-hidden-layer tanh nets of width n trained by full-batch GD (momentum) on N = 8 points with random targets, 6 seeds per width. Widths 1–2 stall at near-critical points with positive loss. From n = N − 1 = 7 on (the theorem’s width) every seed reaches zero loss. Here n = 5–6 also do, and n = 3–4 on some seeds: N − 1 is sufficient, not necessary.',
      note: 'Empirical illustration, not the theorem: Theorem 3.8 is about critical points (plus non-degeneracy and full-rank conditions), and says nothing about how GD finds them. 40 000 steps, lr 0.05, β = 0.9.',
      hint: '', legend: [[HEX.ball, 'final loss (each seed)'], [HEX.neg, 'n = N − 1']],
      controls: [{ id: 'n', label: 'show the fit of width n', min: 1, max: 10, step: 1, value: 2, fmt: v => String(+v) }],
      charts: [{ id: 'fit', title: 'data and the best seed’s fitted function', xlabel: 'x', ylabel: 'y', xlim: [-2.3, 2.3], ylim: [-1.6, 1.6], h: 260, legend: false }],
      plot: { xlabel: 'hidden width n', ylabel: 'final training loss', logy: true, xlim: [.5, 10.5], ylim: [1e-31, 1], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const D = ctx.data, P = ctx.plot.clear().set({ xfmt: v => Number.isInteger(v) ? String(v) : '' });
        P.vline(D.N - 1, { color: HEX.neg, dash: [5, 4], text: 'N − 1' });
        D.rows.forEach(([n, L]) => P.scatter(L.map(l => [n, Math.max(l, 1e-31)]), { color: HEX.ball, r: 3.5 }));
        P.draw();
        const n = ctx.vals.n, xs = D.xs, pr = D.preds[n];
        ctx.charts.fit.clear().line(xs.map((x, i) => [x, pr[i]]), { color: HEX.ball, width: 2 }).scatter(D.X.map((x, i) => [x, D.Y[i]]), { color: HEX.good, r: 4 }).draw();
        const row = D.rows[n - 1];
        ctx.stat(`width ${n}: losses ${row[1].map(l => l < 1e-20 ? '≈0' : l.toExponential(1)).join(', ')} · ‖∇L‖ median ${[...row[2]].sort((a, b) => a - b)[3].toExponential(1)} · rank [F, 1] ${row[3].join(', ')} (N = ${D.N})`);
      },
    },
  ],
};
