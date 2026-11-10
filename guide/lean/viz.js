// Chapter 9 (survey Sections III-I to III-K). New visuals 'factor' (Eq. 35) and 'lion' (Eqs. 36-38); numbers in
// scripts/check-math/guide3_numbers.py. Reused: crash-course 'lookahead' (course_memory.py), defazio2024road 'three'.
import { HEX } from '../../src/lib/atlas3d.js';
import { rampCss } from '../../src/lib/charts.js';
import { rng } from '../../src/lib/math.js';
import mem from '../../course/memory/viz.js';
import { paperTab } from '../lib.js';

// ---------- Adafactor: rank-one reconstruction of a nonnegative second-moment matrix from its row and column sums
const M = 8, N = 12;
const R0 = rng(35), U = Array.from({ length: M }, () => .2 + R0()), W = Array.from({ length: N }, () => .2 + R0());
const Z = Array.from({ length: M }, () => Array.from({ length: N }, () => R0.normal()));
function factorOf(s) {
  const V = U.map((u, i) => W.map((w, j) => u * w * Math.exp(s * Z[i][j])));
  const r = V.map(row => row.reduce((a, b) => a + b, 0)), c = W.map((_, j) => V.reduce((a, row) => a + row[j], 0)), S = r.reduce((a, b) => a + b, 0);
  const Vt = r.map(ri => c.map(cj => ri * cj / S));                                  // Eq. (35)
  let num = 0, den = 0, worst = 0;
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) { num += (V[i][j] - Vt[i][j]) ** 2; den += V[i][j] ** 2; worst = Math.max(worst, Math.abs(Math.log(Vt[i][j] / V[i][j]))); }
  return { V, Vt, rel: Math.sqrt(num / den), worst: Math.exp(worst) };
}
const factor = {
  title: 'Adafactor keeps a row sum and a column sum instead of the whole matrix',
  caption: 'Left: a nonnegative second-moment matrix V for an 8 × 12 weight (one cell per weight). Middle: Adafactor\'s reconstruction Ṽᵢⱼ = rᵢcⱼ / Σrₖ from only the 8 row sums and 12 column sums (Eq. 35). Right: the ratio Ṽ/V per cell. The slider adds structure that a rank-one product cannot capture.',
  note: 'V = (u wᵀ) ⊙ exp(s·Z) with fixed positive u, w and a fixed Gaussian matrix Z. At s = 0, V is exactly rank one and the reconstruction is exact. Colour scales: V and Ṽ share one scale; the ratio is drawn on a log scale from ½ (dark) to 2 (bright).',
  legend: [],
  mode: '2d',
  plot: { xlabel: '', ylabel: '', xlim: [-.5, 41], ylim: [-1.6, 8.5], legend: false, grid: false, xfmt: () => '', yfmt: () => '' },
  controls: [{ id: 's', label: 'departure from rank one, s', min: 0, max: 1.5, step: .01, value: .5 }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { V, Vt, rel, worst } = factorOf(ctx.vals.s), P = ctx.plot.clear();
    const vmax = Math.max(...V.flat(), ...Vt.flat());
    const flip = A => A.slice().reverse();
    P.heat(flip(V), [0, 12, 0, 8], { cell: z => rampCss(Math.sqrt(z / vmax)) });
    P.heat(flip(Vt), [14, 26, 0, 8], { cell: z => rampCss(Math.sqrt(z / vmax)) });
    const ratio = Vt.map((row, i) => row.map((v, j) => v / V[i][j]));
    P.heat(flip(ratio), [28, 40, 0, 8], { cell: z => rampCss(Math.min(1, Math.max(0, (Math.log2(z) + 1) / 2))) });
    P.text(0, -.9, `V: ${M * N} numbers`, { color: '#e8ecf8' }); P.text(14, -.9, `Ṽ from ${M} + ${N} = ${M + N} sums`, { color: '#e8ecf8' }); P.text(28, -.9, 'ratio Ṽ/V (½ … 2)', { color: '#e8ecf8' });
    P.draw();
    ctx.stat(`s = ${ctx.vals.s.toFixed(2)} · storage ${M}×${N} = ${M * N} numbers → ${M} + ${N} = ${M + N} (ratio (m + n)/(mn) = ${((M + N) / (M * N)).toFixed(3)}) · relative error ‖Ṽ − V‖/‖V‖ = ${rel.toFixed(3)} · worst cell off by a factor ${worst.toFixed(2)}`);
  },
};

// ---------- Lion: the sign of a blend of momentum and gradient; every coordinate moves by exactly eta
const T = 120, MU = .3, SIG = 1.2;
const R1 = rng(36), G = Array.from({ length: T }, () => MU + SIG * R1.normal());
const lion = {
  title: 'Lion moves every coordinate by exactly η; the gradient only decides the sign',
  caption: 'One coordinate with noisy gradients: true mean 0.3, noise 1.2 (white dots). Lion\'s direction is the sign of c_t = β₁m_{t−1} + (1 − β₁)ĝ_t (coral), where the momentum m_t = β₂m_{t−1} + (1 − β₂)ĝ_t (violet) uses a second, slower coefficient. The bars show the step taken: always ±η, never larger or smaller.',
  note: 'Eqs. (36)-(38) without weight decay; one fixed seeded gradient sequence. SGD would step by −ηĝ_t, proportional to the gradient\'s size. "Right direction" means the step opposes the true mean gradient 0.3.',
  legend: [['#e8ecf8', 'gradient ĝ_t'], [HEX.opt2, 'momentum m_t (β₂)'], [HEX.ball, 'c_t (β₁ blend)'], [HEX.good, 'step −η·sign(c_t), in units of η']],
  mode: '2d',
  plot: { xlabel: 'step t', ylabel: 'value', xlim: [0, T], ylim: [-4, 4], legend: false },
  controls: [
    { id: 'b1', label: 'β₁ (blend for the sign)', min: 0, max: .99, step: .01, value: .9 },
    { id: 'b2', label: 'β₂ (momentum memory)', min: 0, max: .999, step: .001, value: .99, digits: 3 },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const b1 = ctx.vals.b1, b2 = ctx.vals.b2, P = ctx.plot.clear();
    let m = 0; const cs = [], ms = [], steps = [];
    let right = 0, rightSgd = 0;
    for (let t = 0; t < T; t++) {
      const c = b1 * m + (1 - b1) * G[t]; const st = -Math.sign(c);             // Eq. (36), (37)
      m = b2 * m + (1 - b2) * G[t];                                               // Eq. (38)
      cs.push([t, c]); ms.push([t, m]); steps.push([t, st]);
      if (st < 0) right++; if (G[t] > 0) rightSgd++;
    }
    P.hline(0, { color: 'rgba(255,255,255,.3)' }); P.hline(MU, { color: 'rgba(95,224,176,.35)', dash: [3, 4], text: 'true mean 0.3' });
    P.custom((g, p) => { for (const [t, s] of steps) { g.fillStyle = s < 0 ? 'rgba(95,224,176,.5)' : 'rgba(255,93,115,.5)'; g.fillRect(p.sx(t) - 1.5, p.sy(0), 3, p.sy(s) - p.sy(0)); } });
    P.scatter(G.map((v, t) => [t, v]), { color: '#e8ecf8', r: 1.8 });
    P.line(ms, { color: HEX.opt2, width: 1.8 }); P.line(cs, { color: HEX.ball, width: 1.6 });
    P.draw();
    ctx.stat(`β₁ = ${b1.toFixed(2)}, β₂ = ${b2.toFixed(3)} · every step has size exactly η · Lion stepped the right way ${right} of ${T} times · a sign of the raw gradient (sign-SGD) would have: ${rightSgd} of ${T} · one array of state (m) instead of Adam's two`);
  },
};

export default { factor, lion, lookahead: mem.lookahead, three: await paperTab('defazio2024road', 'three') };
