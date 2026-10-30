// Crawshaw, Modi, Liu, Gower 2026 — Muon and its variants as non-Euclidean steepest descent: CSD w + ηLMO(m) and
// RSD w + η‖m‖_* LMO(m) (Props. 3.1–3.2); for the spectral norm LMO(M) = −polar(M), ‖M‖_* = nuclear norm.
// Checked in scripts/check-math/crawshaw2026exploration.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const H = [[4, .6], [.6, .3]], CEN = [.6, -.5], W0 = [-1.6, 1.5];
const f = (a, b) => { const x = a - CEN[0], y = b - CEN[1]; return .5 * (H[0][0] * x * x + 2 * H[0][1] * x * y + H[1][1] * y * y); };
const g = (a, b) => { const x = a - CEN[0], y = b - CEN[1]; return [H[0][0] * x + H[0][1] * y, H[0][1] * x + H[1][1] * y]; };
const surf = def({ L: f, g, dom: [-2, 2, -2, 2], h: v => Math.log1p(v) });
const sgn = v => v > 0 ? 1 : v < 0 ? -1 : 0;
const steps = {
  gd: eta => p => { const q = g(...p); return [p[0] - eta * q[0], p[1] - eta * q[1]]; },
  muon: eta => p => { const q = g(...p); return [p[0] - eta * sgn(q[0]), p[1] - eta * sgn(q[1])]; },
  rsd: eta => p => { const q = g(...p), n = Math.abs(q[0]) + Math.abs(q[1]); return [p[0] - eta * n * sgn(q[0]), p[1] - eta * n * sgn(q[1])]; },
};
/** symmetric 2x2 matrix power (eigen-decomposition) */
function spow(M, p) {
  const a = M[0][0], b = M[0][1], c = M[1][1], m = (a + c) / 2, d = Math.hypot((a - c) / 2, b), l1 = m + d, l2 = m - d;
  let v = Math.abs(b) > 1e-14 ? [l1 - c, b] : (a >= c ? [1, 0] : [0, 1]); const n = Math.hypot(...v); v = [v[0] / n, v[1] / n];
  const u = [-v[1], v[0]], p1 = Math.max(l1, 1e-300) ** p, p2 = Math.max(l2, 1e-300) ** p;
  return [[p1 * v[0] * v[0] + p2 * u[0] * u[0], p1 * v[0] * v[1] + p2 * u[0] * u[1]], [p1 * v[1] * v[0] + p2 * u[1] * u[0], p1 * v[1] * v[1] + p2 * u[1] * u[1]]];
}
const mm = (X, Y) => X.map(r => [0, 1].map(j => r[0] * Y[0][j] + r[1] * Y[1][j]));
const tr = X => [[X[0][0], X[1][0]], [X[0][1], X[1][1]]];
/** polar(G) = G (G^T G)^{-1/2}; singular values = eigenvalues of (G^T G)^{1/2} */
function polar2(Gm) {
  const GtG = mm(tr(Gm), Gm), S = spow(GtG, .5), a = S[0][0], b = S[0][1], c = S[1][1], m = (a + c) / 2, d = Math.hypot((a - c) / 2, b);
  return { s: [m + d, m - d], polar: mm(Gm, spow(GtG, -.5)) };
}

export default {
  tabs: [
    {
      id: 'steps', name: 'Steepest in which norm?', title: 'Same gradient, different geometry, different optimizer',
      caption: 'A 2 × 2 weight matrix kept diagonal, W = diag(a, b), on an ill-conditioned quadratic. Steepest descent in the Euclidean norm with regularization is plain GD (white). In the spectral norm the steepest unit step is −polar(G), which for a diagonal matrix is −sign(G): Muon’s constrained step (violet) moves every coordinate by exactly η, so it rushes along the flat direction but then zig-zags at a distance set by η. The regularized version (coral) multiplies the same direction by the dual (nuclear) norm ‖G‖_* = |g₁| + |g₂|, so its steps shrink near the minimum and it converges.',
      note: 'Diagonal restriction so the parameter space is a plane; for full matrices polar(G) mixes entries. Steps: GD 0.4, Muon (CSD) 0.05, spectral RSD 0.1. No momentum (m = g).',
      legend: [[HEX.opt3, 'GD (Euclidean RSD)'], [HEX.opt2, 'Muon (spectral CSD)'], [HEX.ball, 'spectral RSD'], [HEX.good, 'minimum']],
      action: 'Restart', cam: [0, 4.6, 4], tgt: [0, .1, 0],
      charts: [{ id: 'L', title: 'loss (log)', xlabel: 'step', ylabel: 'f', logy: true, ylim: [1e-12, 100], xlim: [0, 400], h: 230, legend: false }],
      build(ctx) { ctx.stage.surface(surf); ctx.stage.marker(surf, ...CEN, C.good, .1); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls(); ctx.H = { gd: [], muon: [], rsd: [] }; ctx.b = {};
        for (const [k, eta, col] of [['gd', .4, C.opt3], ['muon', .05, C.opt2], ['rsd', .1, C.ball]]) {
          const b = ctx.b[k] = s.ball(surf, { start: W0, speed: .8, maxSteps: 400, step: steps[k](eta), color: col });
          b.onStep = bb => ctx.H[k].push([bb.steps, f(...bb.p)]);
        }
      },
      frame(ctx) {
        const P = ctx.charts.L.clear(), col = { gd: HEX.opt3, muon: HEX.opt2, rsd: HEX.ball };
        for (const k of ['gd', 'muon', 'rsd']) P.line(ctx.H[k].map(h => [h[0], Math.max(h[1], 1e-12)]), { color: col[k], width: 1.5 });
        P.draw();
        const last = k => (ctx.H[k][ctx.H[k].length - 1]?.[1] ?? NaN).toExponential(1);
        ctx.stat(`step ${ctx.b.rsd.steps} · loss: GD ${last('gd')} · Muon (CSD) ${last('muon')} · spectral RSD ${last('rsd')}`);
      },
    },
    {
      id: 'lmo', name: 'Polar factor', mode: '2d', title: 'The spectral-norm steepest step is −UVᵀ',
      caption: 'For a 2 × 2 gradient G = U diag(σ₁, σ₂) Vᵀ, the steepest unit step in the Frobenius norm is −G/‖G‖_F and decreases the linear model by ‖G‖_F; in the spectral norm it is −UVᵀ (Muon’s polar factor) and decreases it by σ₁ + σ₂, the nuclear norm. Both singular directions get a full unit step, however different σ₁ and σ₂ are. Bars: the decrease ⟨G, −D⟩ each geometry achieves with a unit step.',
      note: 'Exact 2 × 2 SVD. Default: the notes’ example G = diag(3, 1), where the Frobenius step gives √10 ≈ 3.16 and the spectral step gives 4.',
      hint: '', legend: [[HEX.opt3, 'Frobenius: ‖G‖_F'], [HEX.opt2, 'spectral (Muon): ‖G‖_nuc']],
      controls: [{ id: 's2', label: 'σ₂ (σ₁ = 3)', min: .05, max: 3, step: .05, value: 1 }, { id: 'rot', label: 'rotation of G', min: 0, max: 180, step: 1, value: 0, fmt: v => `${v}°` }],
      plot: { xlabel: '', ylabel: 'decrease of ⟨G, ·⟩ per unit step', ylim: [0, 6.5], legend: false, xfmt: () => '' },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const t = ctx.vals.rot * Math.PI / 180, R = [[Math.cos(t), -Math.sin(t)], [Math.sin(t), Math.cos(t)]], s2 = ctx.vals.s2;
        const Gm = [[3 * R[0][0], s2 * R[0][1]], [3 * R[1][0], s2 * R[1][1]]];   // G = R diag(3, s2)
        const { s, polar } = polar2(Gm), fro = Math.hypot(...Gm.flat()), nuc = s[0] + s[1], ip = Gm[0][0] * polar[0][0] + Gm[0][1] * polar[0][1] + Gm[1][0] * polar[1][0] + Gm[1][1] * polar[1][1];
        const P = ctx.plot.clear().bars([{ y: fro, color: HEX.opt3 }, { y: ip, color: HEX.opt2 }]);
        P.text(0, fro + .3, `Frobenius ${fro.toFixed(3)}`, { color: HEX.white, align: 'center' }).text(1, ip + .3, `spectral ${ip.toFixed(3)}`, { color: HEX.white, align: 'center' }).draw();
        ctx.stat(`σ = (${s[0].toFixed(2)}, ${s[1].toFixed(2)}) · ‖G‖_F = ${fro.toFixed(3)} · ⟨G, polar(G)⟩ = ${ip.toFixed(3)} = σ₁ + σ₂ = ${nuc.toFixed(3)} · polar(G) = [[${polar[0].map(v => v.toFixed(2)).join(', ')}], [${polar[1].map(v => v.toFixed(2)).join(', ')}]]`);
      },
    },
  ],
};
