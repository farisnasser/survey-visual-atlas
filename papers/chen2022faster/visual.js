// Chen, Zhou, Gu 2022 — LENA (Algorithm 1) with the SPIDER estimator (Algorithm 2) in the online setting.
// Noise model, B, b = q = √B, ε, η, η_H, r, D, ℓ_thres from scripts/check-math/chen2022faster.py (data.json).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const fL = (x, y) => .25 * (x * x - 1) ** 2 + .5 * y * y;
const fG = (x, y) => [x ** 3 - x, y];
const F = def({ L: fL, g: fG, hess: (x, y) => [3 * x * x - 1, 0, 1], h: v => Math.log1p(v * 3), dom: [-1.6, 1.6, -1.6, 1.6] });
const X0 = [0, 1.3];
const sg = (D, p, R) => { const b = R.normal() * D.sa, c = R.normal() * D.sc; return [b, c]; };
const gs = (p, xi) => [p[0] ** 3 - p[0] + xi[1] * p[0], p[1] + xi[0] - xi[1] * p[1]];

/** One iteration of LENA; the ball's state machine holds the SPIDER estimate d, the phase and the escape-phase movement. */
const lena = (D, R) => (p, s) => {
  const est = (xo, xn) => {
    s.t++;
    if (s.t % D.b === 0) { let gx = 0, gy = 0; for (let i = 0; i < D.B; i++) { const g = gs(xn, sg(D, xn, R)); gx += g[0]; gy += g[1]; } s.sfo += D.B; return [gx / D.B, gy / D.B]; }
    let dx = 0, dy = 0; for (let i = 0; i < D.b; i++) { const xi = sg(D, xn, R), u = gs(xn, xi), w = gs(xo, xi); dx += u[0] - w[0]; dy += u[1] - w[1]; }
    s.sfo += 2 * D.b; return [s.dd[0] + dx / D.b, s.dd[1] + dy / D.b];
  };
  const go = (eta) => { const q = [p[0] - eta * s.dd[0], p[1] - eta * s.dd[1]]; s.dd = est(p, q); return q; };
  const nd = Math.hypot(...s.dd);
  if (s.phase === 'gd') {
    if (nd > D.eps) return go(D.eta / nd);
    s.ms = p.slice(); s.phase = 'esc'; s.k = 0; s.mov = 0; s.budget = [[0, 0, D.D]];
    const th = 2 * Math.PI * R(), rr = D.r * Math.sqrt(R()), q = [p[0] + rr * Math.cos(th), p[1] + rr * Math.sin(th)];
    s.kicks.push(q.slice()); s.dd = est(p, q); return q;
  }
  const st2 = D.etaH * D.etaH * (s.dd[0] ** 2 + s.dd[1] ** 2);
  if (s.mov + st2 > (s.k + 1) * D.D) {   // last step shrinkage
    const eta = Math.sqrt(Math.max((s.k + 1) * D.D - s.mov, 0) / (s.dd[0] ** 2 + s.dd[1] ** 2));
    s.mov = (s.k + 1) * D.D; s.budget.push([s.k + 1, s.mov, (s.k + 1) * D.D]); s.shrinks.push(p.slice()); s.phase = 'gd'; return go(eta);
  }
  s.mov += st2; s.k++; s.budget.push([s.k, s.mov, (s.k + 1) * D.D]);
  if (s.k >= D.lthres) { s.out = s.ms; s.phase = 'out'; s.done = true; }
  return go(D.etaH);
};

export default {
  tabs: [
    {
      id: 'lena', name: 'Last-step shrinkage', title: 'Perturb, then cap the average movement so the gradient estimate stays accurate',
      caption: 'Stochastic gradients only. GD phase: normalized steps of length η along the SPIDER estimate d_t until ‖d_t‖ ≤ ε. Escape phase: a kick of radius r (amber), then steps η_H·d_t while tracking Σ η_i²‖d_i‖². If that exceeds the budget (steps + 1)·D, the last step is shrunk to meet it exactly (red), which signals a successful escape, and a new GD phase starts. If ℓ_thres steps pass under budget, LENA returns the point before the kick (green ring).',
      note: 'F = (x² − 1)²/4 + y²/2, noise in ∇f(x; ξ) = ∇F + (c·x, b − c·y) vanishes in x on the line x = 0, so only the kick can start the escape. B = 400, b = q = 20, ε = 0.05, η = 0.03, η_H = 0.1, D = 10⁻⁴, ℓ_thres = 120.',
      legend: [[HEX.ball, 'LENA-SPIDER'], [HEX.flat, 'perturbation'], [HEX.neg, 'last-step shrinkage'], [HEX.good, 'output']],
      action: 'Run again', cam: [3.2, 3.6, 3.6], tgt: [0, .2, 0],
      charts: [{ id: 'mv', title: 'escape phase: Σ η_i²‖d_i‖² vs budget', xlabel: 'escape step k', ylabel: 'movement²', logy: true, ylim: [1e-8, 2e-2], h: 170 },
        { id: 'g', title: '‖d_t‖ (estimate) and ‖∇F(x_t)‖', xlabel: 'iteration t', ylabel: 'norm', logy: true, ylim: [1e-4, 3], h: 170 }],
      build(ctx) { const s = ctx.stage; s.surface(F); s.eigenLines(F, 0, 0, { len: .55 }); s.marker(F, 1, 0, C.good, .06); s.marker(F, -1, 0, C.good, .06); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data, R = rng(Date.now() % 1e6); s.clearBalls(); (ctx.marks || []).forEach(o => s.remove(o)); ctx.marks = []; ctx.nk = ctx.ns = 0; ctx.outShown = false;
        const b = ctx.b = s.ball(F, { start: X0, step: lena(D, R), speed: .5, maxSteps: 3000 });
        let gx = 0, gy = 0; for (let i = 0; i < D.B; i++) { const g = gs(X0, sg(D, X0, R)); gx += g[0]; gy += g[1]; }
        Object.assign(b, { dd: [gx / D.B, gy / D.B], t: 0, sfo: D.B, phase: 'gd', kicks: [], shrinks: [], budget: [], h: [], hd: [] });
        b.onStep = z => { z.h.push([z.steps, Math.max(Math.hypot(...fG(...z.p)), 1e-4)]); z.hd.push([z.steps, Math.max(Math.hypot(...z.dd), 1e-4)]); };
      },
      frame(ctx) {
        const b = ctx.b, D = ctx.data; if (!b) return;
        while (ctx.nk < b.kicks.length) { const k = b.kicks[ctx.nk++]; ctx.marks.push(ctx.stage.marker(F, k[0], k[1], C.flat, .07)); }
        while (ctx.ns < b.shrinks.length) { const k = b.shrinks[ctx.ns++]; ctx.marks.push(ctx.stage.dot(F.W(k[0], k[1], .05), C.neg, .045)); }
        if (b.out && !ctx.outShown) { ctx.outShown = true; ctx.marks.push(ctx.stage.marker(F, b.out[0], b.out[1], C.good, .12)); }
        ctx.charts.mv.clear().set({ xlim: [0, Math.max(10, b.budget.length)] }).line(b.budget.map(r => [r[0], r[2]]), { color: HEX.muted, dash: [4, 4], width: 1.2, label: '(k+1)·D' })
          .line(b.budget.map(r => [r[0], Math.max(r[1], 1e-8)]), { color: HEX.ball, width: 1.8, label: 'Σ η²‖d‖²' }).draw();
        ctx.charts.g.clear().set({ xlim: [0, Math.max(50, b.steps)] }).hline(D.eps, { color: HEX.muted, dash: [4, 4], text: 'ε' })
          .line(b.hd, { color: HEX.opt2, width: 1.2, label: '‖d_t‖' }).line(b.h, { color: HEX.ball, width: 1.6, label: '‖∇F‖' }).draw();
        const ph = b.phase === 'gd' ? 'GD phase' : b.phase === 'esc' ? `escape phase, step ${b.k}/${D.lthres}` : 'returned x_{m_s}';
        ctx.stat(`t = ${b.t} · ${ph} · ${b.kicks.length} kick${b.kicks.length === 1 ? '' : 's'}, ${b.shrinks.length} shrink${b.shrinks.length === 1 ? '' : 's'} · ${b.sfo} stochastic gradients${b.out ? ` · output (${b.out.map(v => v.toFixed(3)).join(', ')}), ‖∇F‖ = ${Math.hypot(...fG(...b.out)).toFixed(3)}` : ''}`);
      },
    },
    {
      id: 'rates', name: 'ε^−3', mode: '2d', title: 'Perturbed stochastic gradients reach the ε^−3 rate',
      caption: 'Stochastic gradients to an (ε, ε_H)-approximate local minimum (orders). LENA: Õ(ε⁻³ + ε_H⁻⁶) (Theorems 5.1, 5.3), so Õ(ε⁻³) when ε_H = √ε; the previous best perturbed method, SSRGD, needs Õ(ε⁻³·⁵) online. The lower bound for any stochastic first-order method is Ω(ε⁻³ + ε_H⁻⁵) (Arjevani et al. 2020): LENA is optimal in ε but not in ε_H.',
      note: 'Constants, σ, L, ρ, Δ set to 1 and logs dropped. Slide the exponent p in ε_H = ε^p.',
      hint: '', legend: [[HEX.ball, 'LENA ε⁻³ + ε_H⁻⁶'], [HEX.opt2, 'SSRGD (online, ε_H = √ε) ε⁻³·⁵'], [HEX.muted, 'lower bound ε⁻³ + ε_H⁻⁵']],
      controls: [{ id: 'p', label: 'ε_H = ε^p, p', min: .25, max: 1, step: .05, value: .5 }],
      plot: { xlabel: 'accuracy ε', ylabel: 'stochastic gradients', logx: true, logy: true, xlim: [1e-5, .1], ylim: [10, 1e30], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const p = ctx.vals.p, P = ctx.plot.clear();
        P.fn(e => e ** -3 + e ** (-5 * p), { color: HEX.muted, dash: [5, 4], width: 1.4 }).fn(e => e ** -3.5, { color: HEX.opt2, width: 1.8 }).fn(e => e ** -3 + e ** (-6 * p), { color: HEX.ball, width: 2.6 }).draw();
        const e = 1e-4;
        ctx.stat(`ε = 10⁻⁴, ε_H = ε^${p.toFixed(2)}: LENA ${(e ** -3 + e ** (-6 * p)).toExponential(1)} · lower bound ${(e ** -3 + e ** (-5 * p)).toExponential(1)} · SSRGD (p = ½) ${(e ** -3.5).toExponential(1)}`);
      },
    },
  ],
};
