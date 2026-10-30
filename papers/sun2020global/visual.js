// Sun, Li, Liang, Ding, Srikant 2020 — overview of neural-network landscapes. Its single-neuron toy (Fig. 2, Prop. IV.1):
// F(v, w) = (1 − v·φ(w))² has spurious local minima iff some zero of φ is a local extremum. scripts/check-math/sun2020global.py
import { def, C, HEX, segPts } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const ACT = {
  quad: [z => -((z - 1) ** 2) / 8, z => -(z - 1) / 4],
  tanh: [Math.tanh, z => 1 / Math.cosh(z) ** 2],
};
const mk = k => { const [p, dp] = ACT[k]; return def({ L: (v, w) => (1 - v * p(w)) ** 2, g: (v, w) => { const e = 1 - v * p(w); return [-2 * e * p(w), -2 * e * v * dp(w)]; }, dom: [-3, 3, -3, 3], h: v => Math.log1p(Math.min(v, 1e4)) }); };

export default {
  tabs: [
    {
      id: 'neuron', name: 'One neuron', title: 'Whether a single neuron has bad minima depends on its activation',
      caption: 'One sample (x, y) = (1, 1) and one neuron, F(v, w) = (1 − v·φ(w))². With the paper’s activation φ(z) = −(z − 1)²/8 (its Fig. 2), φ has a zero at z = 1 that is also its maximum, so every point (v, 1) with v > 0 is a local minimum with loss 1 (white line), flat along v; GD from about a third of the starts ends there. With tanh, whose only zero is not an extremum, there are no such minima (Prop. IV.1) and GD reaches zero loss from every start except the saddle at the origin.',
      note: 'GD step 0.05, 30 random starts in [−3, 3]². Height log(1 + F). Green: zero-loss curves v·φ(w) = 1.',
      legend: [[HEX.ball, 'GD'], [HEX.good, 'global minima (F = 0)'], ['#ffffff', 'spurious minima (F = 1)']],
      action: 'Release balls', cam: [0, 4.6, 4.6], tgt: [0, .2, 0],
      controls: [{ id: 'act', type: 'seg', label: 'activation φ', value: 'quad', options: [['quad', '−(z − 1)²/8 (Fig. 2)'], ['tanh', 'tanh']] }],
      build(ctx) {
        const k = ctx.vals.act, d = ctx.d = mk(k), s = ctx.stage, phi = ACT[k][0]; s.surface(d);
        const pos = [], neg = []; for (let w = -3; w <= 3; w += .01) { const v = 1 / phi(w); if (isFinite(v) && Math.abs(v) <= 3) (v > 0 ? pos : neg).push([v, w]); }
        const split = a => { const out = [[]]; a.forEach((p, i) => { if (i && Math.abs(p[1] - a[i - 1][1]) > .02) out.push([]); out[out.length - 1].push(p); }); return out; };
        [...split(pos), ...split(neg)].forEach(seg => seg.length > 1 && s.tube(d, seg, C.good, .013));
        if (k === 'quad') s.tube(d, segPts(0, 1, 3, 1), C.white, .016);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(9); s.clearBalls();
        ctx.bs = Array.from({ length: 30 }, () => s.ball(ctx.d, { start: [R.uniform(-3, 3), R.uniform(-3, 3)], step: (p, b, d) => { const g = d.g(...p); return [p[0] - .05 * g[0], p[1] - .05 * g[1]]; }, speed: 6, maxSteps: 6000, r: .045 }));
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.bs) return;
        const glob = ctx.bs.filter(b => b.loss() < 1e-6).length, sp = ctx.bs.filter(b => Math.abs(b.loss() - 1) < 1e-3).length, R = ctx.data.res[ctx.vals.act];
        ctx.stat(`step ${ctx.bs[0].steps} · zero loss ${glob}/30 · stuck at F = 1: ${sp}/30 · offline, 41 × 41 grid of starts: ${Math.round(100 * R[0])}% global, ${(100 * R[1]).toFixed(1)}% at F = 1`);
      },
    },
  ],
};
