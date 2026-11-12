// Chapter 13 (survey Sections V-A, V-B). New visual 'kron': the exact Fisher block (survey Section V-B) against K-FAC's A ⊗ S
// (Eq. 36) for jointly Gaussian a and delta, computed with Isserlis' theorem; numbers in scripts/check-math/guide3_numbers.py.
import { HEX } from '../../src/lib/atlas3d.js';

const n = 3, m = 2;                                   // layer s = W a with a in R^3, s in R^2: the Fisher block is 6 x 6
const SA = [[1, .4, .1], [.4, .8, -.2], [.1, -.2, .6]];          // Cov(a)
const SZ = [[.9, .3], [.3, .5]];                                   // Cov of the part of delta independent of a
const B = [[.8, -.5, .3], [.2, .7, -.6]];                         // delta's dependent part: rho * B a
const mm = (X, Y) => X.map(r => Y[0].map((_, j) => r.reduce((s, v, k) => s + v * Y[k][j], 0)));
const tr = X => X[0].map((_, j) => X.map(r => r[j]));
function blocks(rho) {
  const A = SA, BSA = mm(B, SA), S0 = mm(BSA, tr(B));
  const S = S0.map((r, i) => r.map((v, j) => rho * rho * v + (1 - rho * rho) * SZ[i][j]));
  const Cm = tr(BSA).map(r => r.map(v => rho * v));              // C = E[a delta^T] = rho Sigma_a B^T  (n x m)
  const N6 = n * m, F = [], K = [], Dl = [];
  for (let p = 0; p < N6; p++) { F.push([]); K.push([]); Dl.push([]); }
  for (let j = 0; j < n; j++) for (let k = 0; k < m; k++) for (let l = 0; l < n; l++) for (let r = 0; r < m; r++) {
    const p = j * m + k, q = l * m + r;                               // vec(delta a^T) = a ⊗ delta (column-major)
    const kf = A[j][l] * S[k][r], extra = Cm[j][k] * Cm[l][r] + Cm[j][r] * Cm[l][k];   // Isserlis: E[a_j d_k a_l d_r]
    F[p][q] = kf + extra; K[p][q] = kf; Dl[p][q] = extra;
  }
  const fro = X => Math.sqrt(X.flat().reduce((s, v) => s + v * v, 0));
  return { F, K, Dl, rel: fro(Dl) / fro(F) };
}
const col = (v, vmax) => { const t = Math.max(-1, Math.min(1, v / vmax)); return t >= 0 ? `rgba(124,196,255,${.08 + .92 * t})` : `rgba(255,93,115,${.08 - .92 * t})`; };
const kron = {
  title: 'K-FAC\'s Kronecker factorization is exact only when activations and errors are independent',
  caption: 'For one layer with n = 3 inputs and m = 2 outputs, the Fisher block F_W = E[(aaᵀ) ⊗ (δδᵀ)] is 6 × 6 (left). K-FAC replaces it by A ⊗ S with A = E[aaᵀ] and S = E[δδᵀ] (middle): two small matrices, 9 + 4 numbers. Right: what K-FAC drops. The slider sets how strongly the back-propagated error δ depends on the activation a.',
  note: 'Toy model: a and δ jointly Gaussian, δ = ρBa + √(1 − ρ²)z with z independent of a. Then every entry is exact by Isserlis\' theorem: F = A ⊗ S + (cross terms in C = E[aδᵀ]). Blue positive, red negative, one shared scale. In a real network δ depends on a nonlinearly; the toy only shows what the factorization assumes.',
  legend: [[HEX.pos, 'positive entry'], [HEX.neg, 'negative entry']],
  mode: '2d',
  plot: { xlabel: '', ylabel: '', xlim: [-.5, 21], ylim: [-1.4, 6.3], legend: false, grid: false, xfmt: () => '', yfmt: () => '' },
  controls: [{ id: 'rho', label: 'dependence of δ on a, ρ', min: 0, max: .99, step: .01, value: .6 }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { F, K, Dl, rel } = blocks(ctx.vals.rho), P = ctx.plot.clear();
    const vmax = Math.max(...F.flat().map(Math.abs), ...K.flat().map(Math.abs));
    const flip = X => X.slice().reverse();
    P.heat(flip(F), [0, 6, 0, 6], { cell: v => col(v, vmax) });
    P.heat(flip(K), [7.5, 13.5, 0, 6], { cell: v => col(v, vmax) });
    P.heat(flip(Dl), [15, 21, 0, 6], { cell: v => col(v, vmax) });
    P.text(0, -.8, 'exact block F_W (36 numbers)', { color: '#e8ecf8' }); P.text(7.5, -.8, 'A ⊗ S (9 + 4 numbers)', { color: '#e8ecf8' }); P.text(15, -.8, 'F_W − A ⊗ S', { color: '#e8ecf8' });
    P.draw();
    ctx.stat(`ρ = ${ctx.vals.rho.toFixed(2)} · relative error ‖F_W − A⊗S‖/‖F_W‖ = ${rel.toFixed(4)}${ctx.vals.rho === 0 ? ' (exact: a and δ independent)' : ''} · storage (mn)² = ${(n * m) ** 2} → n² + m² = ${n * n + m * m}; for a 1000 × 1000 layer: 10¹² → 2·10⁶`);
  },
};

export default { kron };
