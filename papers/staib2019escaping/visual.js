// Staib, Reddi, Kale, Kumar, Sra 2019 — adaptive methods whiten gradient noise near stationary points (Eq. 3),
// replicated on the paper's Sec. 5 problem. Median curves from scripts/check-math/staib2019escaping.py (data.json).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const BV = [[1, .1], [1, -.1], [-1, .1], [-1, -.1]];          // E b = 0, Cov b = diag(1, 0.01)
const fL = (x, y) => .5 * x * x - .05 * y * y + x ** 10 + y ** 10;
const fG = (x, y) => [x + 10 * x ** 9, -.1 * y + 10 * y ** 9];
const F = def({ L: fL, g: fG, hess: (x, y) => [1 + 90 * x ** 8, 0, -.1 + 90 * y ** 8], h: v => Math.cbrt(v), dom: [-.5, .5, -.75, .75], square: true });
const YS = .01 ** (1 / 8);

const sgd = (eta, R) => (p) => { const b = BV[Math.floor(R() * 4)], g = fG(...p); return [p[0] - eta * (g[0] + b[0]), p[1] - eta * (g[1] + b[1])]; };
/** Diagonal RMSProp, β = 1 − η^{2/3}, with a burn-in of η^{−2/3} gradient samples at the start (Algorithm 3). */
const rms = (eta, R) => (p, s) => {
  const beta = 1 - eta ** (2 / 3), samp = q => { const b = BV[Math.floor(R() * 4)], g = fG(...q); return [g[0] + b[0], g[1] + b[1]]; };
  if (!s.v) { s.v = [0, 0]; for (let k = 0; k < Math.round(eta ** (-2 / 3)); k++) { const g = samp(p); s.v = [beta * s.v[0] + (1 - beta) * g[0] * g[0], beta * s.v[1] + (1 - beta) * g[1] * g[1]]; } }
  const g = samp(p); s.v = [beta * s.v[0] + (1 - beta) * g[0] * g[0], beta * s.v[1] + (1 - beta) * g[1] * g[1]];
  return [p[0] - eta * g[0] / Math.sqrt(s.v[0] + 1e-8), p[1] - eta * g[1] / Math.sqrt(s.v[1] + 1e-8)];
};

export default {
  tabs: [
    {
      id: 'escape', name: 'Escape race', title: 'Same step size, very different escape: RMSProp boosts the weak noise direction',
      caption: 'The paper’s Sec. 5 problem: f_i(x) = ½xᵀdiag(1, −0.1)x + b_iᵀx + ‖x‖₁₀¹⁰, saddle at 0, escape direction x₂, but the gradient noise there is 10× smaller than across. All balls start at the saddle. With η = 0.001, SGD (violet) barely moves in 10⁴ steps while diagonal RMSProp (coral) escapes. SGD needs η = 0.01 (white) to escape as fast, and then it rattles around the minimum.',
      note: 'b_i = (±1, ±0.1) gives exactly Cov(b) = diag(1, 0.01); RMSProp β = 1 − η^{2/3} with a burn-in, as in the paper. Heights are cube-root scaled; the minima (0, ±0.562) are only 0.013 deep.',
      legend: [[HEX.ball, 'RMSProp η = 0.001'], [HEX.opt2, 'SGD η = 0.001'], [HEX.opt3, 'SGD η = 0.01'], [HEX.good, 'minima']],
      action: 'Run again', cam: [3.4, 3.0, 3.4], tgt: [0, .1, 0],
      charts: [{ id: 'f', title: 'f(x_t)  (dashed: median of 20 runs, check script)', xlabel: 'iteration t', ylabel: 'f', xlim: [0, 10000], ylim: [-.014, .004], h: 230 }],
      build(ctx) { const s = ctx.stage; s.surface(F); s.eigenLines(F, 0, 0, { len: .35, fmtL: l => l.toFixed(1) }); s.marker(F, 0, YS, C.good, .08); s.marker(F, 0, -YS, C.good, .08); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, seed = Date.now() % 1e6; s.clearBalls();
        ctx.b = [s.ball(F, { start: [0, 0], step: rms(.001, rng(seed)), speed: 40, maxSteps: 10000 }),
          s.ball(F, { start: [0, 0], step: sgd(.001, rng(seed + 1)), color: C.opt2, speed: 40, maxSteps: 10000 }),
          s.ball(F, { start: [0, 0], step: sgd(.01, rng(seed + 2)), color: C.opt3, speed: 40, maxSteps: 10000 })];
        for (const q of ctx.b) { q.h = []; q.onStep = z => { if (z.steps % 25 === 0) z.h.push([z.steps, z.loss()]); }; }
      },
      frame(ctx) {
        const B = ctx.b, D = ctx.data; if (!B) return;
        const med = (k, c) => D.curves[k].map((v, i) => [i * 50, v]);
        ctx.charts.f.clear().hline(D.fstar, { color: HEX.good, dash: [2, 3], text: 'f*' })
          .line(med('sgd .01'), { color: HEX.opt3, dash: [4, 4], width: 1, alpha: .6 }).line(med('sgd .001'), { color: HEX.opt2, dash: [4, 4], width: 1, alpha: .6 }).line(med('rms .001'), { color: HEX.ball, dash: [4, 4], width: 1, alpha: .6 })
          .line(B[2].h, { color: HEX.opt3, width: 1.3, label: 'SGD .01' }).line(B[1].h, { color: HEX.opt2, width: 1.5, label: 'SGD .001' }).line(B[0].h, { color: HEX.ball, width: 1.8, label: 'RMSProp .001' }).draw();
        ctx.stat(`t = ${B[0].steps} · |x₂|: RMSProp ${Math.abs(B[0].p[1]).toFixed(3)}, SGD(.001) ${Math.abs(B[1].p[1]).toFixed(3)}, SGD(.01) ${Math.abs(B[2].p[1]).toFixed(3)} · check script medians to |x₂| > 0.4: ${Math.round(D.esc['rms .001'])} / ${D.esc['sgd .001'] >= 10000 ? '> 10⁴' : Math.round(D.esc['sgd .001'])} / ${Math.round(D.esc['sgd .01'])}`);
      },
    },
    {
      id: 'whiten', name: 'Isotropic noise', mode: '2d', title: 'Preconditioning by E[ggᵀ]^{−1/2} makes the noise isotropic near stationary points',
      caption: 'Covariance ellipses (1 standard deviation) of the gradient noise at the point (0, s). Violet: raw noise g − ∇f, almost flat along the escape direction x₂. Coral: the preconditioned noise ξ = G^{−1/2}(g − ∇f) with G = E[ggᵀ], whose covariance is exactly I − G^{−1/2}∇f∇fᵀG^{−1/2} (Eq. 3): a unit circle at the saddle, shrinking only along ∇f as the gradient grows.',
      note: 'Full-matrix idealized preconditioner (Algorithm 1). Diagonal RMSProp gives the same picture here because Cov(b) and ∇f are axis-aligned.',
      hint: '', legend: [[HEX.opt2, 'raw noise'], [HEX.ball, 'preconditioned noise'], [HEX.muted, 'unit circle']],
      controls: [{ id: 's', label: 'point (0, s), s', min: 0, max: .7, step: .01, value: 0 }],
      plot: { xlabel: 'noise along x₁', ylabel: 'noise along x₂ (escape)', xlim: [-1.6, 1.6], ylim: [-1.2, 1.2], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const s = ctx.vals.s, gr = fG(0, s), G = [1 + gr[0] * gr[0], gr[0] * gr[1], .01 + gr[1] * gr[1]];   // G = Cov(b) + ∇f∇fᵀ
        const isq = [1 / Math.sqrt(G[0]), 0, 1 / Math.sqrt(G[2])];   // G is diagonal here (gr[0] = 0)
        const u = [isq[0] * gr[0], isq[2] * gr[1]], cov = [1 - u[0] * u[0], -u[0] * u[1], 1 - u[1] * u[1]];
        const ell = (a, b, c) => { const tr = a + c, det = a * c - b * b, d = Math.sqrt(Math.max(tr * tr / 4 - det, 0)), l1 = tr / 2 + d, l2 = Math.max(tr / 2 - d, 0), th = .5 * Math.atan2(2 * b, a - c);
          return Array.from({ length: 121 }, (_, i) => { const t = 2 * Math.PI * i / 120, x = Math.sqrt(l1) * Math.cos(t), y = Math.sqrt(l2) * Math.sin(t); return [x * Math.cos(th) - y * Math.sin(th), x * Math.sin(th) + y * Math.cos(th)]; }); };
        ctx.plot.clear().line(ell(1, 0, 1), { color: HEX.muted, dash: [4, 4], width: 1 }).line(ell(1, 0, .01), { color: HEX.opt2, width: 2 }).line(ell(...cov), { color: HEX.ball, width: 2.4 })
          .arrows([[0, 0, 0, Math.sign(s || 1) * .9]], { color: HEX.neg, width: 1.5 }).text(.08, .95, 'escape direction', { color: HEX.neg }).draw();
        ctx.stat(`s = ${s.toFixed(2)} · ‖∇f‖ = ${Math.hypot(...gr).toFixed(4)} · raw noise std along x₂: 0.10 · preconditioned std along x₂: ${Math.sqrt(Math.max(cov[2], 0)).toFixed(3)}, along x₁: ${Math.sqrt(cov[0]).toFixed(3)}`);
      },
    },
  ],
};
