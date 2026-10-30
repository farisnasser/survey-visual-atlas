// Haeffele & Vidal 2017 — positively homogeneous network + regularizer of the same degree: a local minimum with an
// all-zero subnetwork is global (Theorem 1). Toy factorizations and the ReLU run: scripts/check-math/haeffele2017global.py.
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const Y = 1;
/** degree-2 map uv with degree-2 regularizer (λ/2)(u² + v²): the convex problem behind it is the soft threshold. */
const matched = lam => def({
  L: (u, v) => .5 * (Y - u * v) ** 2 + lam / 2 * (u * u + v * v),
  g: (u, v) => [-(Y - u * v) * v + lam * u, -(Y - u * v) * u + lam * v],
  dom: [-1.6, 1.6, -1.6, 1.6],
});
/** degree-2 map uv with a degree-1 regularizer λ(|u| + |v|): Prop. 19 of the long version says 0 is a local minimum. */
const mismatch = lam => def({ L: (u, v) => .5 * (Y - u * v) ** 2 + lam * (Math.abs(u) + Math.abs(v)), dom: [-1.6, 1.6, -1.6, 1.6] });
const soft = (z, t) => Math.sign(z) * Math.max(Math.abs(z) - t, 0);
/** proximal gradient (ISTA): gradient step on the smooth part, soft threshold for the l1 part. */
const ista = (eta, lam) => p => { const r = Y - p[0] * p[1]; return [soft(p[0] + eta * r * p[1], eta * lam), soft(p[1] + eta * r * p[0], eta * lam)]; };
function misGlobal(lam) {   // u = v = a, minimise ½(1 − a²)² + 2λa on a fine grid (as the check script)
  let best = [0, .5]; for (let i = 0; i <= 150000; i++) { const a = 1.5 * i / 150000, f = .5 * (1 - a * a) ** 2 + 2 * lam * a; if (f < best[1]) best = [a, f]; }
  return best;
}

export default {
  tabs: [
    {
      id: 'matched', name: 'Matched degree', title: 'Same degree of homogeneity: every local minimum is global',
      caption: 'One "subnetwork" φ(u, v) = uv (degree 2) with the regularizer (λ/2)(u² + v²), also degree 2. Below λ = 1 the origin, where the subnetwork is zero, is a strict saddle and every ball reaches the global value; from λ = 1 on the origin is a local minimum, and Theorem 1 says it is then global, which it is.',
      note: 'f(u, v) = ½(1 − uv)² + (λ/2)(u² + v²); minimum over X = uv of ½(1 − X)² + λ|X| (the convex problem the paper builds) is the soft threshold X* = max(1 − λ, 0). GD, step 0.1.',
      legend: [[HEX.ball, 'GD from random starts'], [HEX.good, 'global minima'], [HEX.neg, 'λ < 0 at the zero subnetwork']],
      action: 'Release balls', cam: [0, 4.4, 4.4], tgt: [0, .3, 0],
      controls: [{ id: 'lam', label: 'regularization λ', min: .05, max: 1.5, step: .05, value: .3, live: false, fmt: v => (+v).toFixed(2) }],
      build(ctx) {
        const lam = ctx.vals.lam, d = ctx.d = matched(lam), s = ctx.stage; s.surface(d);
        ctx.fstar = lam < Y ? lam * Y - lam * lam / 2 : .5 * Y * Y;
        if (lam < Y) { const a = Math.sqrt(Y - lam); s.marker(d, a, a, C.good); s.marker(d, -a, -a, C.good); } else s.marker(d, 0, 0, C.good);
        s.eigenLines(d, 0, 0, { len: .5 });
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(5); s.clearBalls(); ctx.bs = [];
        for (let k = 0; k < 14; k++) ctx.bs.push(s.ball(ctx.d, { start: [R.uniform(-1.4, 1.4), R.uniform(-1.4, 1.4)], step: (p, b, d) => { const g = d.g(...p); return [p[0] - .1 * g[0], p[1] - .1 * g[1]]; }, speed: 3, maxSteps: 3000, r: .05 }));
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.bs) return;
        const done = ctx.bs.filter(b => b.done || b.steps > 2000), at = done.filter(b => Math.abs(b.loss() - ctx.fstar) < 1e-6).length;
        ctx.stat(`λ = ${ctx.vals.lam.toFixed(2)} · global value f* = ${ctx.fstar.toFixed(4)} · balls at f*: ${at}/${ctx.bs.length}${done.length < ctx.bs.length ? ` · still moving ${ctx.bs.length - done.length}` : ''} · origin is ${ctx.vals.lam < Y ? 'a strict saddle (λmin = λ − 1)' : 'the global minimum'}`);
      },
    },
    {
      id: 'mismatch', name: 'Degree mismatch', title: 'Regularize with a lower degree and the zero subnetwork becomes a trap',
      caption: 'Same map uv (degree 2), now with the degree-1 penalty λ(|u| + |v|). Near the origin the loss can only fall like uv while the penalty rises like |u| + |v|, so the origin is a local minimum for every λ > 0, even when a much better minimum exists. Proximal gradient balls that start near the axes get stuck there.',
      note: 'f(u, v) = ½(1 − uv)² + λ(|u| + |v|). Optimizer: proximal gradient (gradient step on the square loss, soft threshold for the penalty), step 0.05. The general statement is Prop. 19 of the long version (arXiv 1506.07540).',
      legend: [[HEX.ball, 'proximal gradient from random starts'], [HEX.good, 'global minima'], [HEX.white, 'spurious minimum at 0']],
      action: 'Release balls', cam: [0, 4.4, 4.4], tgt: [0, .3, 0],
      controls: [{ id: 'lam', label: 'regularization λ', min: .05, max: .35, step: .01, value: .15, live: false, fmt: v => (+v).toFixed(2) }],
      build(ctx) {
        const lam = ctx.vals.lam, d = ctx.d = mismatch(lam), s = ctx.stage; s.surface(d);
        ctx.G = misGlobal(lam);
        if (ctx.G[0] > 0) { s.marker(d, ctx.G[0], ctx.G[0], C.good); s.marker(d, -ctx.G[0], -ctx.G[0], C.good); }
        s.marker(d, 0, 0, C.white, .12);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(1); s.clearBalls(); ctx.bs = [];
        for (let k = 0; k < 40; k++) ctx.bs.push(s.ball(ctx.d, { start: [R.uniform(-1.4, 1.4), R.uniform(-1.4, 1.4)], step: ista(.05, ctx.vals.lam), speed: 6, maxSteps: 6000, r: .045, stop: b => Math.hypot(...b.p) < 1e-12 && b.steps > 10 }));
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.bs) return;
        const z = ctx.bs.filter(b => Math.hypot(...b.p) < 1e-9).length, g = ctx.bs.filter(b => Math.abs(b.loss() - ctx.G[1]) < 1e-5).length;
        ctx.stat(`λ = ${ctx.vals.lam.toFixed(2)} · f(0) = 0.5 · global value ${ctx.G[1].toFixed(4)}${ctx.G[0] ? '' : ' (the origin itself)'} · stuck at the origin: ${z}/40 · at the global minimum: ${g}/40`);
      },
    },
    {
      id: 'relu', name: 'ReLU network', mode: '2d', title: 'A zero neuron certifies a global minimum, but only at a true local minimum',
      caption: 'One-hidden-layer ReLU networks with r neurons and weight decay (a degree-2 network with a degree-2 penalty) trained by GD from 16 random starts each. Runs that end with an all-zero neuron and pass the local-minimum test (no new neuron can lower the objective) all reach exactly the same value, the global one (this happened for r = 6, 8 and 12). Runs whose zero neurons are merely dead, with a descent direction left, stall higher.',
      note: 'g(x) = Σᵢ aᵢ ReLU(wᵢ·[x, 1]), 10 points of y = sin 2.5x + 0.3x, objective (1/2n)Σ(g − y)² + (λ/2)Σ(aᵢ² + ‖wᵢ‖²), λ = 0.002, 200 000 GD steps of size 0.2 (offline). Test: max over unit w of |⟨residual, ReLU(Xw)⟩|/n ≤ λ (Corollary 1).',
      hint: '', legend: [[HEX.good, 'zero neuron and local-min test passed'], [HEX.flat, 'zero (dead) neuron, test fails'], [HEX.ball, 'no zero neuron']],
      plot: { xlabel: 'neurons r', ylabel: 'final objective', logy: true, xlim: [0, 13], ylim: [.01, .6], legend: false },
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear(), cnt = {};
        P.hline(D.fstar, { color: HEX.good, dash: [5, 4], text: `global value ${D.fstar.toFixed(5)}` });
        const pts = D.relu.map(([r, f, z, ok]) => { cnt[r] = (cnt[r] || 0) + 1; return [r + (cnt[r] - 8.5) * .035, f, ok ? HEX.good : z ? HEX.flat : HEX.ball]; });
        P.scatter(pts, { r: 3.4 }).draw();
        const ok = D.relu.filter(q => q[3]), dead = D.relu.filter(q => q[2] && !q[3]), nz = D.relu.filter(q => !q[2]);
        ctx.stat(`${D.relu.length} runs · zero neuron + test passed: ${ok.length}, all at ${D.fstar.toFixed(5)} · dead neuron, test failed: ${dead.length}, lowest ${Math.min(...dead.map(q => q[1])).toFixed(5)} · no zero neuron: ${nz.length}`);
      },
    },
  ],
};
