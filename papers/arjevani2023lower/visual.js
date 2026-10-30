// Arjevani, Carmon, Duchi, Foster, Srebro, Woodworth 2023 — the chain F_T with a probability-p stochastic gradient (Eq. 17):
// the next coordinate's gradient is shown only with probability p (scaled by 1/p to stay unbiased).
// Query counts from scripts/check-math/arjevani2023lower.py (data.json).
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const erf = x => { const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + .3275911 * a); return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-a * a)); };
const Psi = t => t <= .5 ? 0 : Math.exp(1 - 1 / (2 * t - 1) ** 2);
const dPsi = t => t <= .5 ? 0 : Psi(t) * 4 / (2 * t - 1) ** 3;
const Phi = t => Math.sqrt(Math.E) * Math.sqrt(2 * Math.PI) * .5 * (1 + erf(t / Math.SQRT2));
const dPhi = t => Math.sqrt(Math.E) * Math.exp(-t * t / 2);
const f2 = (a, b) => -Psi(1) * Phi(a) + Psi(-a) * Phi(-b) - Psi(a) * Phi(b);
const g2 = (a, b) => [-Psi(1) * dPhi(a) - dPsi(-a) * Phi(-b) - dPsi(a) * Phi(b), -Psi(-a) * dPhi(-b) - Psi(a) * dPhi(b)];
const F = def({ L: f2, g: g2, h: v => v, dom: [-.6, 3, -.6, 3], N: 120 });
const prog = (p, a) => (Math.abs(p[1]) > a ? 2 : Math.abs(p[0]) > a ? 1 : 0);
/** Eq. (17): coordinates beyond prog_{1/4}(x) are multiplied by z/p, z ~ Bernoulli(p). */
const noisy = (h, p, R) => (x, b) => { const g = g2(...x), k = prog(x, .25), z = R() < p ? 1 : 0; const G = g.map((v, i) => (i >= k ? v * z / p : v)); return [x[0] - h * G[0], x[1] - h * G[1]]; };
const PCOL = { 1: HEX.opt3, .2: HEX.opt2, .05: HEX.ball };

export default {
  tabs: [
    {
      id: 'oracle', name: 'Rare progress', title: 'An unbiased noisy gradient that hides the next coordinate most of the time',
      caption: 'Same hard chain as Carmon et al. (here T = 2). White: exact gradients. Coral: the paper’s stochastic oracle (Eq. 17) — the gradient component of the next undiscovered coordinate is returned only with probability p (multiplied by 1/p, so the estimate stays unbiased with bounded variance). Every other query shows no sign of x₂, so the ball must wait about 1/p queries per new coordinate; over T coordinates that is ≈ T/p queries (Lemma 1).',
      note: 'p = 0.05, step 0.05, real function. Variance of the oracle ≤ 23²(1 − p)/p (Lemma 3); scaling the instance turns T/p into the ε⁻⁴ bound.',
      legend: [[HEX.opt3, 'exact gradient'], [HEX.ball, 'probability-p oracle, p = 0.05']],
      cam: [3.6, 3.4, 3.8], tgt: [0, .2, 0], action: 'Run again',
      charts: [{ id: 'x2', title: 'x₂ (the hidden coordinate)', xlabel: 'query', ylabel: 'x₂', ylim: [-.1, 3], h: 210 }],
      build(ctx) { const s = ctx.stage; s.surface(F); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls(); const R = rng(Date.now() % 1e6);
        const stop = z => Math.hypot(...g2(...z.p)) < .9;
        ctx.a = s.ball(F, { start: [0, 0], step: opt.gd(.05), color: C.opt3, speed: .3, maxSteps: 300, stop });
        ctx.b = s.ball(F, { start: [0, 0], step: noisy(.05, .05, R), speed: .3, maxSteps: 300, stop });
        for (const q of [ctx.a, ctx.b]) { q.h = [[0, 0]]; q.onStep = z => z.h.push([z.steps, z.p[1]]); }
      },
      frame(ctx) {
        const { a, b } = ctx; if (!a) return;
        ctx.charts.x2.clear().set({ xlim: [0, 300] }).line(a.h, { color: HEX.opt3, width: 1.6, label: 'exact' }).line(b.h, { color: HEX.ball, width: 2, label: 'p = 0.05' }).draw();
        ctx.stat(`query ${b.steps} · exact: x = (${a.p.map(v => v.toFixed(2)).join(', ')}) · noisy oracle: x = (${b.p.map(v => v.toFixed(2)).join(', ')}), coordinates discovered ${prog(b.p, 0)}`);
      },
    },
    {
      id: 'bounds', name: 'ε⁻⁴ and ε⁻³', mode: '2d', title: 'Stochastic first-order methods need ε⁻⁴ queries, ε⁻³ with mean-squared smoothness',
      caption: 'Left: queries a zero-respecting method (SGD, step 0.05) needs to activate all T coordinates of the chain under the probability-p oracle, averaged over 20 runs: proportional to T/p. Right: the resulting lower bounds. With bounded variance σ² (Theorem 1) any method needs Ω(ΔLσ²ε⁻⁴), matched by SGD; if every stochastic gradient is also mean-squared smooth (Theorem 2), Ω(ΔL̄σε⁻³ + σ²ε⁻²), matched by SPIDER/SNVRG.',
      note: 'Left: our runs (p = 1 is exact GD). Right: orders with Δ, L, σ = 1. Theorem 3 extends both to all randomized algorithms.',
      hint: '', legend: [[HEX.ball, 'p = 0.05'], [HEX.opt2, 'p = 0.2'], [HEX.opt3, 'p = 1']],
      plot: { xlabel: 'chain length T', ylabel: 'queries to activate all T', xlim: [0, 17], legend: false },
      charts: [{ id: 'lb', title: 'lower bounds', xlabel: 'accuracy ε', ylabel: 'queries', logx: true, logy: true, xlim: [1e-4, .1], h: 240 }],
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear();
        for (const p of [1, .2, .05]) P.line(D.res.filter(r => r[0] === p).map(r => [r[1], r[2]]), { color: PCOL[p], width: 2.2, dots: 4 });
        P.draw();
        ctx.charts.lb.clear().fn(e => e ** -4, { color: HEX.ball, width: 2, label: 'bounded variance ε⁻⁴ (SGD)' }).fn(e => e ** -3 + e ** -2, { color: HEX.opt2, width: 2, label: 'mean-sq. smooth ε⁻³ (SPIDER)' }).fn(e => e ** -2, { color: HEX.muted, dash: [4, 4], width: 1.2, label: 'deterministic ε⁻²' }).draw();
        ctx.stat(D.res.filter(r => r[1] === 16).map(r => `T = 16, p = ${r[0]}: ${Math.round(r[2])} queries`).join(' · '));
      },
    },
  ],
};
