// Carmon, Duchi, Hinder, Sidford 2020 — the hard instance f̄_T of Eq. (10), a zero-chain: each derivative query can
// uncover at most one new coordinate. Counts from scripts/check-math/carmon2020lower.py (data.json).
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';

const erf = x => { const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + .3275911 * a); return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-a * a)); };
const Psi = t => t <= .5 ? 0 : Math.exp(1 - 1 / (2 * t - 1) ** 2);
const dPsi = t => t <= .5 ? 0 : Psi(t) * 4 / (2 * t - 1) ** 3;
const Phi = t => Math.sqrt(Math.E) * Math.sqrt(2 * Math.PI) * .5 * (1 + erf(t / Math.SQRT2));
const dPhi = t => Math.sqrt(Math.E) * Math.exp(-t * t / 2);
const f2 = (a, b) => -Psi(1) * Phi(a) + Psi(-a) * Phi(-b) - Psi(a) * Phi(b);
const g2 = (a, b) => [-Psi(1) * dPhi(a) - dPsi(-a) * Phi(-b) - dPsi(a) * Phi(b), -Psi(-a) * dPhi(-b) - Psi(a) * dPhi(b)];
const F = def({ L: f2, g: g2, h: v => v, dom: [-.6, 3, -.6, 3], N: 120 });

export default {
  tabs: [
    {
      id: 'chain', name: 'The hard function', title: 'A function that reveals its coordinates one at a time',
      caption: 'The two-dimensional hard instance f̄₂(x₁, x₂) = −Ψ(1)Φ(x₁) + Ψ(−x₁)Φ(−x₂) − Ψ(x₁)Φ(x₂) of Eq. (10), where Ψ is exactly zero below ½. Gradient descent from 0 sees no slope at all in x₂ until x₁ has passed ½: the path is an L. The gradient stays larger than 1 until every coordinate is at least 1 in size (Lemma 2), so in T dimensions any method that only explores coordinates it has seen needs about T rounds.',
      note: 'Real function and gradient, GD step 0.05. Ψ(t) = exp(1 − 1/(2t − 1)²) for t > ½; Φ(t) = √e ∫_{−∞}^t e^{−s²/2} ds. The amber line marks x₁ = ½.',
      legend: [[HEX.ball, 'gradient descent from 0'], [HEX.flat, 'x₁ = ½']],
      cam: [3.6, 3.4, 3.8], tgt: [0, .2, 0], action: 'Run again',
      charts: [{ id: 'g', title: '‖∇f̄₂‖ along the path (Lemma 2 threshold 1)', xlabel: 'GD step', ylabel: '‖∇f‖', ylim: [0, 3], h: 210 }],
      build(ctx) { const s = ctx.stage; s.surface(F); s.line(F, Array.from({ length: 61 }, (_, i) => [.5, -.6 + 3.6 * i / 60]), C.flat, .9, .03); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        ctx.b = s.ball(F, { start: [0, 0], step: opt.gd(.05), speed: .25, maxSteps: 120, stop: z => Math.hypot(...g2(...z.p)) < .9 });
        ctx.b.h = [[0, Math.hypot(...g2(0, 0))]]; ctx.b.first = null;
        ctx.b.onStep = z => { z.h.push([z.steps, Math.hypot(...g2(...z.p))]); if (z.first === null && z.p[1] !== 0) z.first = [z.steps, z.p[0]]; };
      },
      frame(ctx) {
        const b = ctx.b; if (!b) return;
        ctx.charts.g.clear().set({ xlim: [0, 120] }).hline(1, { color: HEX.neg, dash: [4, 4], text: '1' }).line(b.h, { color: HEX.ball, width: 2 }).draw();
        ctx.stat(`step ${b.steps} · x = (${b.p.map(v => v.toFixed(3)).join(', ')}) · ‖∇f‖ = ${Math.hypot(...g2(...b.p)).toFixed(3)} · ${b.first ? `x₂ first moved at step ${b.first[0]}, when x₁ = ${b.first[1].toFixed(3)}` : 'x₂ is still exactly 0'}`);
      },
    },
    {
      id: 'bounds', name: 'The bounds', mode: '2d', title: 'ε^{−(p+1)/p} queries are necessary, and known methods achieve it',
      caption: 'Left: gradient descent on the T-dimensional hard instance needs a number of steps that grows linearly with T before ‖∇f‖ ≤ 1 — one new coordinate per stage. Scaling f̄_T to have gap Δ and Lipschitz p-th derivative turns T into ∝ ε^{−(p+1)/p}. Right: the resulting lower bound for any (even randomized, higher-order) algorithm matches gradient descent (p = 1, ε⁻²), cubic-regularized Newton (p = 2, ε^{−3/2}) and p-th order regularization.',
      note: 'Left: our GD runs (step 0.05) from 0. Right: orders only, constants Δ, L_p set to 1. Theorem 1 (deterministic), Theorem 2 (randomized, via random rotations).',
      hint: '', legend: [[HEX.ball, 'GD steps until ‖∇f̄_T‖ ≤ 1'], [HEX.opt2, 'p = 1'], [HEX.opt3, 'p = 2'], [HEX.flat, 'p = 3']],
      plot: { xlabel: 'dimension T of the chain', ylabel: 'GD steps', xlim: [0, 21], legend: false },
      charts: [{ id: 'lb', title: 'lower bound ε^{−(p+1)/p}', xlabel: 'accuracy ε', ylabel: 'queries', logx: true, logy: true, xlim: [1e-4, .1], h: 240 }],
      build(ctx) {
        const D = ctx.data;
        ctx.plot.clear().line(D.counts, { color: HEX.ball, width: 2.4, dots: 4 }).draw();
        ctx.charts.lb.clear().fn(e => e ** -2, { color: HEX.opt2, width: 2, label: 'p=1: ε⁻² (GD)' }).fn(e => e ** -1.5, { color: HEX.opt3, width: 2, label: 'p=2: ε^{-3/2} (cubic)' }).fn(e => e ** (-4 / 3), { color: HEX.flat, width: 2, label: 'p=3: ε^{-4/3}' }).draw();
        ctx.stat(`GD steps until ‖∇f̄_T‖ ≤ 1: ${D.counts.map(([t, k]) => `T=${t}: ${k}`).join(', ')} — about ${D.slope} steps per coordinate`);
      },
    },
  ],
};
