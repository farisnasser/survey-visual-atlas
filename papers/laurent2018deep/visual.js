// Laurent & von Brecht 2018 — convex differentiable losses vs the paper's nondifferentiable counterexample.
// Checked in scripts/check-math/laurent2018deep.py
import { def, opt, C, HEX, logH, hyperbola, segPts, statLabels } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const T = 2;
const LOSSES = {
  sq: { name: 'squared', f: A => (A - T) ** 2, df: A => 2 * (A - T) },
  lc: { name: 'log-cosh', f: A => Math.log(Math.cosh(A - T)), df: A => Math.tanh(A - T) },
};
const chain = k => { const l = LOSSES[k]; return def({ L: (x, y) => l.f(x * y), g: (x, y) => { const d = l.df(x * y); return [d * y, d * x]; }, h: L => logH(L) * 1.25, dom: [-2, 2, -2, 2] }); };
// counterexample slice: W1 = a, W2 = (1-b, b); F = |a(1-b)| + (1-ab)_+ - 1
const kink = def({
  L: (a, b) => Math.abs(a * (1 - b)) + Math.max(1 - a * b, 0) - 1,
  g: (a, b) => { const s = Math.sign(a * (1 - b)), act = 1 - a * b > 0 ? 1 : 0; return [s * (1 - b) - b * act, -s * a - a * act]; },
  h: F => (F + 1) * .62, dom: [-1.2, 2, -1.1, 1.6], N: 160,
});
kink._analytic = false; // subgradient: finite differences straddle the kinks, so skip the self-check
const subgrad = c => (p, b, d) => { const k = b.steps + 1, [ga, gb] = d.g(p[0], p[1]); return [p[0] - c / Math.sqrt(k) * ga, p[1] - c / Math.sqrt(k) * gb]; };

export default {
  tabs: [
    {
      id: 'convex', name: 'Any convex loss', title: 'Depth adds saddles, not bad minima',
      caption: 'Two scalar layers, A = w₁w₂, any convex differentiable loss ℓ(A). Every ball reaches the global valley; the origin is a saddle.',
      note: 'Scalar weights stand in for matrices; the theorem needs every hidden layer at least as wide as the input or output.',
      legend: [[HEX.ball, 'gradient descent'], [HEX.good, 'w₁w₂ = 2 (global minima)'], [HEX.neg, 'λ < 0'], [HEX.pos, 'λ > 0']],
      action: 'Drop 8 balls', cam: [0, 4.4, 5.4], tgt: [0, .35, 0],
      controls: [{ id: 'loss', type: 'seg', label: 'outer loss ℓ', value: 'sq', options: [['sq', 'squared'], ['lc', 'log-cosh']] }],
      charts: [{ id: 'f', title: 'Outer loss ℓ(A) — convex in the product A', xlabel: 'A = w₂w₁', ylabel: 'ℓ(A)', xlim: [-2.5, 4.5], legend: false }],
      build(ctx) {
        const s = ctx.stage, S = ctx.S = s.surface(chain(ctx.vals.loss));
        s.tube(S, hyperbola(T, 2, 1), C.good); s.tube(S, hyperbola(T, 2, -1), C.good);
        s.marker(S, 0, 0, C.white); s.eigenLines(S, 0, 0, { len: .9 });
        this.drop(ctx);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      drop(ctx) {
        const s = ctx.stage, r = rng(Date.now() % 1e6); s.clearBalls();
        for (let i = 0; i < 8; i++) s.ball(ctx.S, { start: [r.uniform(-1.8, 1.8), r.uniform(-1.8, 1.8)], step: opt.gd(.02), speed: 3, stop: b => b.loss() < 1e-8 });
      },
      frame(ctx) {
        const l = LOSSES[ctx.vals.loss], ch = ctx.charts.f, As = ctx.stage.balls.map(b => b.p[0] * b.p[1]);
        ch.clear().fn(l.f, { color: HEX.pos, width: 2 }).vline(T, { color: HEX.good, dash: [4, 4], text: "ℓ′(A) = 0" });
        ch.scatter(As.map(A => [A, l.f(A)]), { color: HEX.ball, r: 4 }); ch.set({ ylim: [0, l.f(-2.5)] }); ch.draw();
        const done = ctx.stage.balls.filter(b => b.done).length;
        ctx.stat(`${done} / ${As.length} balls at a global minimum · products A = ${As.slice(0, 4).map(a => a.toFixed(2)).join(', ')}…`);
      },
    },
    {
      id: 'kink', name: 'Kinked loss', title: 'Drop differentiability and a bad minimum appears',
      caption: 'The paper’s counterexample: a convex, Lipschitz but non-differentiable ℓ. One ball gets stuck at loss 0 even though −1 is reachable.',
      note: 'Slice W₁ = α, W₂ = (1 − β, β) of the paper’s 3-weight example. Balls run subgradient descent, step 0.05/√k.',
      legend: [[HEX.ball, 'subgradient descent'], [HEX.good, 'global minima · F = −1'], ['#ffffff', 'bad local minimum · F = 0']],
      actions: [{ label: 'Release both balls', fn: ctx => ctx.shell.tab.drop(ctx) }],
      cam: [3.6, 3.4, 4.2], tgt: [0, .55, 0],
      build(ctx) {
        const s = ctx.stage; s.surface(kink);
        s.tube(kink, segPts(1, 1, 2, 1), C.good, .022);
        s.tube(kink, segPts(0, -.5, 0, .5), C.white, .014);
        s.label('bad local min · F = 0', kink.W(0, -.35, .35), '', HEX.white);
        s.label('global · F = −1', kink.W(1.6, 1, .3), '', HEX.good);
        ctx.L = statLabels(s, [['', kink.W(.5, -.9, 0).setY(1.6)]]);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        s.ball(kink, { start: [.15, .1], step: subgrad(.05), speed: 2, maxSteps: 4000 });
        s.ball(kink, { start: [.6, .85], step: subgrad(.05), speed: 2, maxSteps: 4000, color: C.opt2 });
      },
      frame(ctx) {
        const [a, b] = ctx.stage.balls;
        if (a && b) ctx.stat(`coral: F = ${a.loss().toFixed(3)} at (α, β) = (${a.p[0].toFixed(2)}, ${a.p[1].toFixed(2)}) · violet: F = ${b.loss().toFixed(3)}`);
      },
    },
  ],
};
