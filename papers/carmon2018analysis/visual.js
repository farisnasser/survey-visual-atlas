// Carmon & Duchi 2018 — Krylov subspace (Lanczos) solutions of the trust-region problem: Theorem 1's two rates and the hard case.
// The notes' 2D hard case is solved exactly here; the d = 400 curves come from scripts/check-math/carmon2018analysis.py (data.json).
import { def, C, HEX, segPts, circlePts, ramp } from '../../src/lib/atlas3d.js';

const LAM = [-2, 1], R = 1;
/** Global trust-region solution for A = diag(LAM), linear term b, radius R (secular equation, hard case included). */
function trSolve(b) {
  const lmin = Math.min(...LAM), lo0 = Math.max(0, -lmin) + 1e-12, nr = l => Math.hypot(b[0] / (LAM[0] + l), b[1] / (LAM[1] + l));
  if (nr(lo0) < R) { const x1 = -b[1] / (LAM[1] + lo0), x0 = Math.sqrt(Math.max(R * R - x1 * x1, 0)); return [[x0, x1], [-x0, x1]]; }
  let lo = lo0, hi = lo0 + 1; while (nr(hi) > R) hi *= 2;
  for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; if (nr(m) > R) lo = m; else hi = m; }
  return [[-b[0] / (LAM[0] + hi), -b[1] / (LAM[1] + hi)]];
}
const q = (x, b) => .5 * (LAM[0] * x[0] ** 2 + LAM[1] * x[1] ** 2) + b[0] * x[0] + b[1] * x[1];
/** Krylov t = 1: minimize over x = a * b/|b|, |a| <= R. */
function krylov1(b) {
  const nb = Math.hypot(...b), u = [b[0] / nb, b[1] / nb], c = LAM[0] * u[0] ** 2 + LAM[1] * u[1] ** 2;
  const cand = [-R, R]; if (c > 0 && Math.abs(nb / c) <= R) cand.push(-nb / c);
  const a = cand.reduce((s, v) => (.5 * c * v * v + nb * v < .5 * c * s * s + nb * s ? v : s));
  return [a * u[0], a * u[1]];
}

export default {
  tabs: [
    {
      id: 'hardcase', name: 'Hard case', title: 'When b has no component on the bottom eigenvector, Krylov methods are stuck',
      caption: 'q(x) = ½xᵀAx + bᵀx on the unit disk with A = diag(−2, 1) and b = (σ, 1). For σ = 0 every Krylov space span{b, Ab, …} is the line x₁ = 0 (amber), so Lanczos returns −½ forever, while the true minimizers sit off that line at −7/6. A tiny σ ≠ 0 (the paper’s random perturbation) makes the second Krylov space all of R², and it finds the global solution.',
      note: 'The notes’ worked example. Outside the trust region the surface is dimmed. Markers: white = Krylov solution in K₁, green = global solution of the current problem.',
      legend: [[HEX.flat, 'Krylov space K₁ = span{b}'], [HEX.white, 'Krylov solution (t = 1)'], [HEX.good, 'global TR solution'], [HEX.muted, 'trust region ‖x‖ ≤ 1']],
      cam: [2.6, 3.4, 3.8], tgt: [0, .3, 0],
      controls: [{ id: 'sig', label: 'perturbation σ', min: 0, max: .3, step: .005, value: 0, digits: 3 }],
      build(ctx) {
        const b = [ctx.vals.sig, 1], L = (x, y) => q([x, y], b);
        const S = def({ L, g: (x, y) => [LAM[0] * x + b[0], LAM[1] * y + b[1]], hess: () => [LAM[0], 0, LAM[1]], h: v => v, dom: [-1.15, 1.15, -1.15, 1.15] });
        const s = ctx.stage; s.surface(S, { colorFn: (x, y, t) => ramp(Math.pow(t, .6)).map(v => Math.hypot(x, y) > R ? v * .28 : v) });
        s.tube(S, circlePts(0, 0, R), C.muted, .01);
        const nb = Math.hypot(...b), u = [b[0] / nb, b[1] / nb]; s.tube(S, segPts(-1.1 * u[0], -1.1 * u[1], 1.1 * u[0], 1.1 * u[1]), C.flat, .012);
        const k1 = krylov1(b), G = trSolve(b);
        s.marker(S, ...k1, C.white, .08); G.forEach(x => s.marker(S, ...x, C.good, .09));
        const v1 = q(k1, b), vs = q(G[0], b), k2 = ctx.vals.sig > 0 ? vs : v1;
        ctx.stat(`K₁: q = ${v1.toFixed(4)} · K₂ = ${ctx.vals.sig > 0 ? 'R² → q = ' + k2.toFixed(4) : 'still span{b} → q = ' + v1.toFixed(4)} · global q* = ${vs.toFixed(4)} at ${G.map(x => `(${x[0].toFixed(3)}, ${x[1].toFixed(3)})`).join(' and ')}`);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
    },
    {
      id: 'rates', name: 'Theorem 1', mode: '2d', title: 'Lanczos error against the linear and sublinear bounds',
      caption: 'A d = 400 trust-region problem with λ_min = −0.6. Generic b: the Krylov gap stays under both bounds of Theorem 1 and converges. Hard case (u_minᵀb = 0, λ* = −λ_min): the gap never closes. The perturbed hard case converges, as Corollary 3 promises.',
      note: 'Offline (exact Lanczos with full reorthogonalization). Bounds: 36[f(0) − f*]·exp(−4t√((λ_min + λ*)/(λ_max + λ*))) and (λ_max − λ_min)‖s*‖²/(t − ½)² · [4 + 𝟙{λ_min < 0}·log²(4‖b‖²/(u_minᵀb)²)/8].',
      hint: '', legend: [[HEX.ball, 'f(s_t) − f(s*)'], [HEX.pos, 'linear bound (4)'], [HEX.opt2, 'sublinear bound (5)']],
      controls: [{ id: 'case', type: 'seg', label: 'instance', value: 'generic', options: [['generic', 'generic b'], ['hard', 'hard case'], ['perturbed', 'hard case + perturbation']] }],
      plot: { xlabel: 'Krylov dimension t', ylabel: 'optimality gap', logy: true, xlim: [1, 120], ylim: [1e-16, 1e3] },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const D = ctx.data.cases[ctx.vals.case], P = ctx.plot.clear(), ts = D.gap.map((_, i) => i + 1);
        if (D.lin) P.line(ts.map((t, i) => [t, D.lin[i]]), { color: HEX.pos, width: 1.6, dash: [5, 4], label: 'linear bound' });
        if (D.sub) P.line(ts.map((t, i) => [t, D.sub[i]]), { color: HEX.opt2, width: 1.6, dash: [5, 4], label: 'sublinear bound' });
        P.line(ts.map((t, i) => [t, D.gap[i]]), { color: HEX.ball, width: 2.4, label: 'Lanczos gap' }).draw();
        const first = D.gap.findIndex(g => g < 1e-10);
        ctx.stat(`λ* = ${D.lam.toFixed(4)}, f* = ${D.fstar.toFixed(5)}${D.hard ? ' · hard case: both bounds are vacuous' : ''} · gap at t = 10: ${D.gap[9].toExponential(2)} · ${first >= 0 ? `below 10⁻¹⁰ from t = ${first + 1}` : `still ${D.gap[D.gap.length - 1].toExponential(2)} at t = 120`}`);
      },
    },
  ],
};
