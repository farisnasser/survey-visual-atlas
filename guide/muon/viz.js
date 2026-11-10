// Chapter 16 (survey Sections V-F to V-H, Appendices A-C, C-C). New visuals 'ns' (Eqs. 79-80 and Muon's quintic) and
// 'spectral' (Eq. 69); numbers in scripts/check-math/guide3_numbers.py.
import { HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

// ---------- Newton-Schulz acts on each singular value separately (Eq. 80), so the spectrum is iterated exactly
const SIG = Array.from({ length: 12 }, (_, i) => .03 ** (i / 11));                 // singular values 1 ... 0.03 (as in Fig. 8)
const S0 = (() => { const f = Math.hypot(...SIG); return SIG.map(s => s / f); })();   // X0 = M / ||M||_F
const MAPS = { cubic: s => .5 * s * (3 - s * s), quintic: s => 3.4445 * s - 4.7750 * s ** 3 + 2.0315 * s ** 5 };
const MULS = { cubic: 2, quintic: 3 };
const ns = {
  title: 'Newton–Schulz pushes every singular value toward 1 using only matrix products',
  caption: 'The momentum matrix M (16 × 12, singular values from 1 down to 0.03) is first divided by its Frobenius norm. Each Newton–Schulz step X ← ½X(3I − XᵀX) (Eq. 79) changes only the singular values, by the scalar map s ← ½s(3 − s²) (Eq. 80), and leaves the singular vectors alone. Muon\'s quintic uses tuned coefficients to move small singular values up faster, at the price of not converging exactly to 1.',
  note: 'Bars are the singular values of X_k, computed exactly by iterating the scalar map on the spectrum of X₀, which is what the matrix iteration does. The ideal polar factor UVᵀ has all singular values equal to 1 (dashed). Quintic coefficients (3.4445, −4.7750, 2.0315) from Jordan et al.',
  legend: [['#8e99bd', 'singular values of M / ‖M‖_F'], [HEX.opt2, 'after k cubic steps'], [HEX.ball, 'after k quintic steps']],
  mode: '2d',
  plot: { xlabel: 'singular value index i', ylabel: 'σᵢ(X_k)', xlim: [.3, 12.7], ylim: [0, 1.4], legend: false },
  controls: [
    { id: 'k', label: 'Newton–Schulz steps k', min: 0, max: 15, step: 1, value: 5, digits: 0 },
    { id: 'map', type: 'seg', label: 'iteration', value: 'cubic', options: [['cubic', 'cubic, Eq. (79)'], ['quintic', 'Muon quintic']] },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const k = ctx.vals.k, kind = ctx.vals.map, f = MAPS[kind];
    let s = S0.slice(); for (let t = 0; t < k; t++) s = s.map(f);
    const P = ctx.plot.clear();
    P.hline(1, { color: 'rgba(255,255,255,.4)', dash: [4, 4], text: 'UVᵀ: all σ = 1' });
    P.custom((g, p) => {
      const w = (p.sx(1) - p.sx(0)) * .34;
      for (let i = 0; i < 12; i++) {
        const x = p.sx(i + 1);
        g.fillStyle = 'rgba(142,153,189,.55)'; g.fillRect(x - w, p.sy(S0[i]), w, p.sy(0) - p.sy(S0[i]));
        g.fillStyle = kind === 'cubic' ? 'rgba(178,140,255,.9)' : 'rgba(255,122,89,.9)'; const y = p.sy(Math.min(1.4, s[i])); g.fillRect(x, y, w, p.sy(0) - y);
      }
    });
    P.draw();
    const dist = Math.hypot(...s.map(v => v - 1));
    ctx.stat(`${kind === 'cubic' ? 'cubic Eq. (79)' : 'Muon quintic'}, k = ${k} · singular values now in [${Math.min(...s).toFixed(3)}, ${Math.max(...s).toFixed(3)}] · ‖X_k − UVᵀ‖_F = ${dist.toFixed(3)} · cost: ${k * MULS[kind]} matrix multiplications, no SVD · the smallest starts at ${S0[11].toFixed(4)} and grows by about ×1.5 per cubic step while it is small`);
  },
};

// ---------- Eq. (69): which spectral-norm-1 update decreases the linear model most?
const R = rng(69);
function orth(rows, cols) {                     // Gram-Schmidt on seeded Gaussian columns: rows x cols with orthonormal columns
  const Q = [];
  for (let j = 0; j < cols; j++) {
    let v = Array.from({ length: rows }, () => R.normal());
    for (const q of Q) { const d = v.reduce((s, x, i) => s + x * q[i], 0); v = v.map((x, i) => x - d * q[i]); }
    const n = Math.hypot(...v); Q.push(v.map(x => x / n));
  }
  return Q;                                     // list of columns
}
const UC = orth(6, 5), VC = orth(5, 5);
const build = sig => Array.from({ length: 6 }, (_, i) => Array.from({ length: 5 }, (_, j) => sig.reduce((s, sk, k) => s + sk * UC[k][i] * VC[k][j], 0)));
function specNorm(A) {                          // power iteration on A^T A
  let v = [1, .9, .8, .7, .6]; for (let t = 0; t < 300; t++) { const Av = A.map(r => r.reduce((s, x, j) => s + x * v[j], 0)); const w = v.map((_, j) => A.reduce((s, r, i) => s + r[j] * Av[i], 0)); const n = Math.hypot(...w); v = w.map(x => x / n); }
  return Math.hypot(...A.map(r => r.reduce((s, x, j) => s + x * v[j], 0)));
}
const spectral = {
  title: 'Eq. (69): under a spectral-norm budget, the polar factor is the steepest step',
  caption: 'One gradient matrix G (6 × 5) with chosen singular values. Each candidate update Δ is scaled so its spectral norm ‖Δ‖₂ (largest singular value) is exactly 1, and the bar is the decrease of the linearized loss, −⟨G, Δ⟩. Von Neumann\'s trace inequality says no such Δ beats Σσᵢ(G), the nuclear norm, and −UVᵀ attains it.',
  note: 'Exact for a seeded G = U diag(σ) Vᵀ; spectral norms by power iteration. Under a Frobenius-norm budget instead, the best step would be −G/‖G‖_F with decrease ‖G‖_F: the geometry decides which step is steepest.',
  legend: [[HEX.opt2, '−G/‖G‖₂: scaled gradient'], [HEX.flat, '−sign(G)/‖sign(G)‖₂: elementwise sign'], [HEX.ball, '−UVᵀ: polar factor (Muon)']],
  mode: '2d',
  plot: { xlabel: '', ylabel: 'linear decrease −⟨G, Δ⟩ with ‖Δ‖₂ = 1', xlim: [-.6, 2.6], ylim: [0, 5.4], legend: false, xfmt: () => '' },
  controls: [{ id: 'p', label: 'spectrum decay p: σᵢ = 2·i^(−p)', min: 0, max: 2, step: .05, value: 1 }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const sig = [1, 2, 3, 4, 5].map(i => 2 * i ** -ctx.vals.p), G = build(sig);
    const s1 = Math.max(...sig), nuc = sig.reduce((a, b) => a + b, 0), fro2 = sig.reduce((a, b) => a + b * b, 0);
    const Sg = G.map(r => r.map(Math.sign)), dSign = G.flat().reduce((s, v) => s + Math.abs(v), 0) / specNorm(Sg);
    const vals = [fro2 / s1, dSign, nuc], cols = [HEX.opt2, HEX.flat, HEX.ball];
    const P = ctx.plot.clear();
    P.hline(nuc, { color: 'rgba(255,255,255,.45)', dash: [4, 4], text: 'Σσᵢ(G): the maximum' });
    P.custom((g, p) => { vals.forEach((v, i) => { g.fillStyle = cols[i]; const y = p.sy(v); g.fillRect(p.sx(i) - 22, y, 44, p.sy(0) - y); }); });
    P.draw();
    ctx.stat(`σ(G) = ${sig.map(v => v.toFixed(2)).join(', ')} · decrease per unit spectral norm: scaled gradient ‖G‖²_F/σ₁ = ${vals[0].toFixed(3)} · elementwise sign ${vals[1].toFixed(3)} · polar factor Σσᵢ = ${nuc.toFixed(3)} (the bound)${ctx.vals.p === 0 ? ' · at p = 0 all σ are equal, so G/‖G‖₂ already is UVᵀ' : ''}`);
  },
};

export default { ns, spectral };
