// Antonakopoulos, Mertikopoulos, Piliouras, Wang 2022 — deterministic AdaGrad (norm, diagonal, full-matrix) avoids strict
// saddles from almost every start. Rotated toy saddle; escape-time sweep: scripts/check-math/antonakopoulos2022adagrad.py
import { def, C, HEX, segPts } from '../../src/lib/atlas3d.js';
import { eig2 } from '../../src/lib/math.js';

const TH = Math.PI / 6, c = Math.cos(TH), s = Math.sin(TH), D0 = 10;
const uv = (x, y) => [c * x + s * y, -s * x + c * y];
/** f = 5u² − v²/2 + v⁴/4 in coordinates rotated by 30°: saddle at 0 (eigenvalues 10, −1 at 30° to the axes), minima v = ±1. */
const F = def({
  L: (x, y) => { const [u, v] = uv(x, y); return 5 * u * u - v * v / 2 + v ** 4 / 4; },
  g: (x, y) => { const [u, v] = uv(x, y), gu = 10 * u, gv = -v + v ** 3; return [c * gu - s * gv, s * gu + c * gv]; },
  dom: [-2, 2, -2, 2],
});
/** One AdaGrad step; b.acc keeps the history (sum of ‖g‖², diag or full outer products, s = 0..t inclusive). */
const ada = kind => (p, b, d) => {
  const g = d.g(p[0], p[1]);
  if (kind === 'gd') return [p[0] - .09 * g[0], p[1] - .09 * g[1]];
  const A = b.A ||= [0, 0, 0]; A[0] += g[0] * g[0]; A[1] += g[0] * g[1]; A[2] += g[1] * g[1];
  if (kind === 'norm') { const k = 1 / Math.sqrt(D0 * D0 + A[0] + A[2]); return [p[0] - k * g[0], p[1] - k * g[1]]; }
  if (kind === 'diag') return [p[0] - g[0] / Math.sqrt(D0 * D0 + A[0]), p[1] - g[1] / Math.sqrt(D0 * D0 + A[2])];
  let sx = 0, sy = 0;   // full: Γ = (δ₀²I + Σ g gᵀ)^{-1/2}
  for (const [lam, v] of eig2(D0 * D0 + A[0], A[1], D0 * D0 + A[2])) { const pr = (g[0] * v[0] + g[1] * v[1]) / Math.sqrt(lam); sx += pr * v[0]; sy += pr * v[1]; }
  return [p[0] - sx, p[1] - sy];
};
const base = [1.5 * c, 1.5 * s], nrm = [-s, c], at = d => [base[0] + d * nrm[0], base[1] + d * nrm[1]];
const NAMES = { gd: 'GD', norm: 'AdaGrad-norm', diag: 'AdaGrad-diag', full: 'AdaGrad-full' };

export default {
  tabs: [
    {
      id: 'adagrad', name: 'AdaGrad near a saddle', title: 'A preconditioner built from the gradient history still lets the saddle go',
      caption: 'A strict saddle whose curvature directions sit at 30° to the coordinate axes (blue: λ = 10, red: λ = −1), minima in the two valleys. All balls start 10⁻³ off the saddle’s stable manifold. GD (white) and full-matrix AdaGrad (violet) slide toward the saddle, stall, and leave. Diagonal AdaGrad (coral) rescales each axis separately, which is not aligned with the curvature, so its own stable set lies elsewhere: from this start it leaves almost at once, while a second coral ball started 10⁻⁹ from its stable set (amber ring) lingers before escaping.',
      note: 'f = 5u² − v²/2 + v⁴/4, (u, v) = 30° rotation of (x, y). AdaGrad: x − Γₜ∇f with Γₜ = (δ₀²I + Σ_{s≤t} gₛgₛᵀ)^−½ (full), its diagonal (diag), δ₀ = 10 ≥ L. GD step 0.09.',
      legend: [[HEX.opt3, 'GD'], [HEX.opt2, 'AdaGrad-full'], [HEX.ball, 'AdaGrad-diag'], [HEX.flat, 'AdaGrad-diag stable set'], [HEX.good, 'minima']],
      action: 'Release balls', cam: [-1.2, 4.2, 4.6], tgt: [0, .3, 0],
      build(ctx) {
        const st = ctx.stage, ds = ctx.data.dstar.diag; st.surface(F); st.marker(F, -s, c, C.good); st.marker(F, s, -c, C.good);
        st.eigenLines(F, 0, 0, { len: 1.1 }); st.marker(F, ...at(ds), C.flat, .08);
        this.drop(ctx);
      },
      drop(ctx) {
        const st = ctx.stage, ds = ctx.data.dstar.diag; st.clearBalls(); ctx.esc = {};
        const mk = (kind, d, color, key) => { const b = st.ball(F, { start: at(d), step: ada(kind), color, speed: 2, maxSteps: 1200 }); b.onStep = q => { if (ctx.esc[key] == null && Math.abs(uv(...q.p)[1]) > .5) ctx.esc[key] = q.steps; }; return b; };
        ctx.bs = [mk('gd', 1e-3, C.opt3, 'gd'), mk('full', 1e-3, C.opt2, 'full'), mk('diag', 1e-3, C.ball, 'diag'), mk('diag', ds + 1e-9, C.ball, 'diag*')];
      },
      frame(ctx) {
        if (!ctx.bs) return;
        const e = k => ctx.esc[k] != null ? `escaped at step ${ctx.esc[k]}` : 'still near the saddle';
        ctx.stat(`step ${ctx.bs[0].steps} · GD: ${e('gd')} · AdaGrad-full: ${e('full')} · AdaGrad-diag from the same start: ${e('diag')} · AdaGrad-diag from 10⁻⁹ off its own stable set: ${e('diag*')}`);
      },
    },
    {
      id: 'escape', name: 'Escape time', mode: '2d', title: 'Every start off the stable set escapes, but closer starts take longer',
      caption: 'Steps until |v| > 0.5 when starting a distance d from each method’s own stable set, on a log scale in d. Every method escapes from every d > 0 (Theorem 2 is an almost-every-start statement), and the time grows like log(1/d), so there is no uniform bound. Diagonal AdaGrad’s stable set crosses the start segment at d* ≈ 0.18, not at the Hessian’s stable manifold.',
      note: 'Offline. Starts (1.5 cos 30°, 1.5 sin 30°) + (d* + d)·(−sin 30°, cos 30°); d* found by bisection on which valley the run ends in.',
      hint: '', legend: [[HEX.opt3, 'GD'], [HEX.pos, 'AdaGrad-norm'], [HEX.ball, 'AdaGrad-diag'], [HEX.opt2, 'AdaGrad-full']],
      plot: { xlabel: 'distance d from the stable set', ylabel: 'steps to escape', logx: true, xlim: [5e-13, .2], ylim: [0, 600], legend: false },
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear(), col = { gd: HEX.opt3, norm: HEX.pos, diag: HEX.ball, full: HEX.opt2 };
        for (const k of ['gd', 'norm', 'diag', 'full']) P.line(D.ds.map((d, i) => [d, D.esc[k][i]]), { color: col[k], width: 2.2, dots: 3.5 });
        P.draw();
        ctx.stat(Object.keys(NAMES).map(k => `${NAMES[k]}: d* = ${(Math.abs(D.dstar[k]) < 1e-6 ? 0 : D.dstar[k]).toFixed(3)}, ${D.esc[k][0]} steps from d = 10⁻¹², ${D.esc[k][11]} from d = 10⁻¹`).join(' · '));
      },
    },
  ],
};
