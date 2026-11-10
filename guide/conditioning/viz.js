// Chapter 4 (survey Sections II-D, II-E; Appendices A-A, C-A, D-B). New visual 'pgdstab' (numbers in
// scripts/check-math/guide3_numbers.py); reused crash-course visuals 'precond' (course_adaptive.py) and 'kappa' (course_curvature.py).
import { HEX } from '../../src/lib/atlas3d.js';
import ad from '../../course/adaptive/viz.js';
import curv from '../../course/curvature/viz.js';

/** H = R(phi) diag(L, 1) R(phi)^T as [a, b, c] = [[a, b], [b, c]] */
// the sign of s is chosen so that 45° gives the survey's Q^T diag(100, 1) Q of Eq. (70), coupling -49.5
const Hrot = (deg, L) => { const t = deg * Math.PI / 180, c = Math.cos(t), s = -Math.sin(t); return [L * c * c + s * s, (L - 1) * c * s, L * s * s + c * c]; };
const eig2 = ([a, b, c]) => { const m = (a + c) / 2, d = Math.hypot((a - c) / 2, b); return [m - d, m + d]; };
/** eigenvalues of K = M^{-1/2} H M^{-1/2} for M = I, diag(H), H */
function Keig(H, kind) {
  if (kind === 'I') return eig2(H);
  if (kind === 'H') return [1, 1];
  const [a, b, c] = H; return eig2([1, b / Math.sqrt(a * c), 1]);
}
const KINDS = [['I', 'GD, M = I', HEX.opt2], ['D', 'diagonal, M = diag(H)', HEX.ball], ['H', 'Newton, M = H', HEX.opt3]];
const rho = (lams, eta) => Math.max(...lams.map(l => Math.abs(1 - eta * l)));

const pgdstab = {
  title: 'When is a preconditioned step stable? Eq. (87) for three choices of M',
  caption: 'The valley ½θᵀHθ has curvatures κ and 1 along axes rotated by the angle. For each preconditioner M, the step θ ← (I − ηM⁻¹H)θ multiplies the error along each eigen-direction of K = M^{-1/2}HM^{-1/2} by 1 − ηλ(K). The curves show the worst factor max|1 − ηλ| against η; below the dashed line at 1 the step contracts.',
  note: 'Exact eigenvalues of the 2 × 2 matrix K; no simulation. At 45° and κ = 100 this is the matrix of the survey\'s Eq. (72): diag(H) = 50.5 I.',
  legend: KINDS.map(k => [k[2], k[1]]),
  mode: '2d',
  plot: { xlabel: 'step size η (log scale)', ylabel: 'worst factor max |1 − ηλ(K)|', logx: true, xlim: [1e-3, 10], ylim: [0, 2], legend: false },
  controls: [
    { id: 'deg', label: 'valley angle (degrees)', min: 0, max: 90, step: 1, value: 45, digits: 0 },
    { id: 'k', type: 'seg', label: 'κ', value: 100, options: [[10, '10'], [100, '100']] },
    { id: 'le', label: 'step size η', min: -3, max: 1, step: .01, value: 0, fmt: v => (10 ** v).toPrecision(3) },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const H = Hrot(ctx.vals.deg, +ctx.vals.k), eta = 10 ** ctx.vals.le, P = ctx.plot.clear();
    P.band(1e-3, 10, { color: 'rgba(0,0,0,0)' });
    P.hline(1, { color: 'rgba(255,255,255,.45)', dash: [4, 4], text: 'contracts below this line' });
    const parts = [];
    for (const [k, name, col] of KINDS) {
      const lam = Keig(H, k), etas = Array.from({ length: 400 }, (_, i) => 10 ** (-3 + 4 * i / 399));
      P.line(etas.map(e => [e, Math.min(rho(lam, e), 2)]), { color: col, width: 2 });
      const emax = 2 / Math.max(...lam);
      P.vline(emax, { color: col, dash: [2, 4], width: 1 });
      const r = rho(lam, eta);
      P.scatter([[eta, Math.min(r, 2)]], { color: col, r: 5 });
      const kap = Math.max(...lam) / Math.min(...lam), best = (kap - 1) / (kap + 1);
      const steps = best < 1e-12 ? 1 : Math.ceil(Math.log(1e-6) / Math.log(best));
      parts.push(`${name.split(',')[0]}: λ(K) = ${lam.map(l => l.toPrecision(3)).join(', ')}, stable for η < ${emax.toPrecision(4)}, factor at this η ${r.toPrecision(3)}${r < 1 ? '' : ' (unstable)'}, best-η steps to 10⁻⁶: ${steps.toLocaleString('en')}`);
    }
    P.vline(eta, { color: 'rgba(255,200,87,.6)', dash: [3, 3] });
    P.draw();
    const [a, b, c] = H;
    ctx.stat(`H = [[${a.toFixed(2)}, ${b.toFixed(2)}], [${b.toFixed(2)}, ${c.toFixed(2)}]] · η = ${eta.toPrecision(3)} · ${parts.join(' · ')}`);
  },
};

export default { pgdstab, precond: ad.precond, kappa: curv.kappa };
