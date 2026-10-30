// Nguyen 2021 — track only the last hidden layer: if σ_min(F_{L−1}) stays ≥ α₀/2, GD on all layers converges
// geometrically (Theorem 2.2). Runs, constants and feature matrices from scripts/check-math/nguyen2021proof.py (data.json):
// a 3-layer ReLU net with widths 4 → 8 → N = 3 → 1 on N = 3 samples.
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';

/** Last-layer least squares with the features frozen at a given F (3×3): Φ(w) = ½‖F w − Y‖², sliced over (w₁, w₂) with w₃ at its optimum. */
function slice(F, Y) {
  const sol = solve3(F, Y);
  const L = (a, b) => { let s = 0; for (let i = 0; i < 3; i++) { const r = F[i][0] * a + F[i][1] * b + F[i][2] * sol[2] - Y[i]; s += r * r; } return .5 * s; };
  const g = (a, b) => { let ga = 0, gb = 0; for (let i = 0; i < 3; i++) { const r = F[i][0] * a + F[i][1] * b + F[i][2] * sol[2] - Y[i]; ga += r * F[i][0]; gb += r * F[i][1]; } return [ga, gb]; };
  let haa = 0, hab = 0, hbb = 0; for (let i = 0; i < 3; i++) { haa += F[i][0] ** 2; hab += F[i][0] * F[i][1]; hbb += F[i][1] ** 2; }
  const R = 1.4 * Math.max(Math.abs(sol[0]), Math.abs(sol[1]), 1);
  return { d: def({ L, g, hess: () => [haa, hab, hbb], h: v => Math.log1p(v * 4), dom: [-R, R, -R, R] }), sol, lmax: (haa + hbb) / 2 + Math.sqrt(((haa - hbb) / 2) ** 2 + hab * hab) };
}
function solve3(A, b) {   // Cramer's rule for the 3×3 system A x = b
  const det = M => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  const D = det(A); return [0, 1, 2].map(c => det(A.map((row, i) => row.map((v, j) => (j === c ? b[i] : v)))) / D);
}

export default {
  tabs: [
    {
      id: 'lastlayer', name: 'Full-rank features', title: 'With full-rank last-layer features, the output layer alone faces a strictly convex bowl',
      caption: 'The loss as a function of the output weights (w₁, w₂), with the last hidden features F_{L−1} frozen at a moment of training and w₃ at its best value. Because the N = 3 feature rows are linearly independent (σ_min(F_{L−1}) > 0), this is a strictly convex quadratic whose minimum fits the labels exactly; the ball runs GD on it. Nguyen’s proof shows that training all layers keeps σ_min above α₀/2, so this picture never breaks.',
      note: 'Ordinary run (β = 1, step 0.23, no guarantee). Slide through training: the features move (σ_min grows from 0.15 to 0.69) but never lose rank, so the bowl stays strictly convex. Curvatures (eigen-lines) are σ² of the feature matrix slice.',
      legend: [[HEX.ball, 'GD on the output weights'], [HEX.pos, 'positive curvature'], [HEX.good, 'exact fit']],
      cam: [3.2, 3.6, 3.6], tgt: [0, .2, 0], action: 'Run again',
      controls: [{ id: 't', label: 'training step', min: 0, max: 3980, step: 20, value: 0, fmt: v => String(Math.round(v)), live: false }],
      build(ctx) {
        const D = ctx.data, s = ctx.stage; s.clear();
        const F = D.Fp[Math.round(ctx.vals.t / 20)], S = ctx.S = slice(F, D.Y);
        s.surface(S.d); s.marker(S.d, S.sol[0], S.sol[1], C.good); s.eigenLines(S.d, S.sol[0], S.sol[1], { len: S.d.dom[1] * .5 });
        this.drop(ctx);
      },
      drop(ctx) { const s = ctx.stage, S = ctx.S; s.clearBalls(); ctx.b = s.ball(S.d, { start: [S.d.dom[0] * .8, S.d.dom[3] * .8], step: opt.gd(1 / S.lmax), speed: .2, maxSteps: 200 }); },
      onControl(ctx) { this.build(ctx); },
      frame(ctx) {
        const b = ctx.b, D = ctx.data; if (!b) return;
        const k = Math.round(ctx.vals.t / 10);
        ctx.stat(`training step ${Math.round(ctx.vals.t)}: σ_min(F_{L−1}) = ${D.Ap[Math.min(k, D.Ap.length - 1)].toFixed(4)} (α₀ = ${D.a0p.toFixed(4)}), network loss ${D.Lp[Math.min(k, D.Lp.length - 1)].toExponential(2)} · GD on the slice: step ${b.steps}, loss ${b.loss().toExponential(2)}`);
      },
    },
    {
      id: 'rate', name: 'The guarantee', mode: '2d', title: 'Theorem 2.2’s conditions can be met, but the guaranteed rate is tiny',
      caption: 'Same network and base weights. Theorem run (white): Section 3.1’s recipe — W_L = 0, other layers scaled by β = 212 so conditions (7)–(9) hold — and the step allowed by (10), η = 1.7·10⁻¹⁶. The loss stays below the guaranteed curve (1 − ηα₀²/8)ᵏΦ₀ (dashed) at every step, but that rate is ~10⁻⁹ per step. Ordinary run (coral): β = 1, step 0.23, same last-layer width N; it reaches zero loss, without a guarantee.',
      note: 'Both checked in the script: Eq. (11) holds at all 4 000 steps of the theorem run and σ_min(F_{L−1}) never drops below α₀/2. The result is about existence and width (N neurons in one layer), not speed.',
      hint: '', legend: [[HEX.opt3, 'theorem run'], [HEX.muted, 'bound (11)'], [HEX.ball, 'ordinary run']],
      plot: { xlabel: 'GD step', ylabel: 'loss Φ', logy: true, xlim: [0, 4000], ylim: [1e-30, 10], legend: false },
      charts: [{ id: 'a', title: 'σ_min(F_{L−1}) / α₀', xlabel: 'GD step', ylabel: 'ratio', ylim: [0, 1.2], h: 200 }],
      build(ctx) {
        const D = ctx.data, pts = a => a.map((v, i) => [i * 10, v]);
        ctx.plot.clear().line(pts(D.B), { color: HEX.muted, dash: [5, 4], width: 1.3 }).line(pts(D.L), { color: HEX.opt3, width: 2 }).line(pts(D.Lp), { color: HEX.ball, width: 2.4 }).draw();
        ctx.charts.a.clear().hline(.5, { color: HEX.neg, dash: [4, 4], text: 'α₀/2' }).line(pts(D.A.map(v => v / D.a0)), { color: HEX.opt3, width: 2 }).line(pts(D.Ap.map(v => v / D.a0p)), { color: HEX.ball, width: 1.6, dash: [3, 3] }).draw();
        ctx.stat(`theorem run: β = ${D.beta.toFixed(1)}, α₀ = ${D.a0.toFixed(0)}, η = ${D.eta.toExponential(2)}, guaranteed rate ηα₀²/8 = ${D.rate.toExponential(1)} per step, loss ${D.L[0].toFixed(4)} → ${D.L[D.L.length - 1].toFixed(4)} · ordinary run: loss → ${D.Lp[D.Lp.length - 1].toExponential(1)}`);
      },
    },
  ],
};
