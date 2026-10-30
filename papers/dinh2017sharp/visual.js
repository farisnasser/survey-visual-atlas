// Dinh, Pascanu, Bengio, Bengio 2017 — rescaling a ReLU layer by a and the next by 1/a leaves the function unchanged but makes
// the minimum as sharp (or flat) as we like. Toy: y = θ₂ ReLU(θ₁x), one datum. scripts/check-math/dinh2017sharp.py
import { def, C, HEX, hyperbola } from '../../src/lib/atlas3d.js';

const F = def({ L: (a, b) => (b * Math.max(a, 0) - 1) ** 2, g: (a, b) => a > 0 ? [2 * (a * b - 1) * b, 2 * (a * b - 1) * a] : [0, 0], dom: [.05, 4.2, .05, 4.2], checkDom: [.2, 4, .2, 4] });
const lam = a => 2 * (a * a + 1 / (a * a));
const epsSharp = (a, e = .1) => { let m = 0; for (let i = 0; i < 3600; i++) { const t = 2 * Math.PI * i / 3600; m = Math.max(m, F.L(a + e * Math.cos(t), 1 / a + e * Math.sin(t))); } return m; };

export default {
  tabs: [
    {
      id: 'rescale', name: 'Same function, any sharpness', title: 'One network function, a whole valley of minima with very different sharpness',
      caption: 'A one-neuron ReLU network y = θ₂·ReLU(θ₁x) fit to the single point (1, 1). Every point of the green valley θ₁θ₂ = 1 is a minimum and computes exactly the same function, because ReLU(aθ₁x)·θ₂/a = ReLU(θ₁x)·θ₂. Slide a to move along the valley: the curvature across it, 2(a² + 1/a²), grows without bound, so “sharp” and “flat” describe the parameterization, not the function.',
      note: 'L(θ₁, θ₂) = (θ₂ ReLU(θ₁) − 1)². Ring: the minimum (a, 1/a); blue: the curved direction, amber: the flat one along the valley. ε-sharpness: max of L on the circle of radius ε = 0.1 (Def. 2).',
      legend: [[HEX.good, 'minima θ₁θ₂ = 1 (one function)'], [HEX.pos, 'λ > 0 across the valley'], [HEX.flat, 'λ = 0 along it']],
      cam: [0, 4.2, 4.6], tgt: [0, .2, 0],
      controls: [{ id: 'la', label: 'a (log scale)', min: -1.4, max: 1.4, step: .01, value: 0, fmt: v => `a = ${Math.exp(v).toFixed(2)}` }],
      build(ctx) { const s = ctx.stage; s.surface(F); s.tube(F, hyperbola(1, 4.2).filter(p => p[0] > .05 && p[1] > .05), C.good, .014); this.mark(ctx); },
      mark(ctx) {
        const s = ctx.stage, a = Math.exp(ctx.vals.la); (ctx.objs || []).forEach(o => s.remove(o)); s.labels.forEach(L => L.e.remove()); s.labels = [];
        const n = s.world.children.length; s.marker(F, a, 1 / a, C.white, .09); s.eigenLines(F, a, 1 / a, { len: .5, labels: false });
        ctx.objs = s.world.children.slice(n);
        ctx.stat(`a = ${a.toFixed(2)} · minimum (${a.toFixed(2)}, ${(1 / a).toFixed(2)}) · output for x = 1: ${((1 / a) * Math.max(a, 0)).toFixed(6)} · Hessian eigenvalues 0 and 2(a² + 1/a²) = ${lam(a).toFixed(2)} · ε-sharpness ${epsSharp(a).toFixed(4)}`);
      },
      onControl(ctx) { this.mark(ctx); },
    },
    {
      id: 'curve', name: 'Sharpness vs a', mode: '2d', title: 'Theorems 3–4: rescaling moves the Hessian, not the function',
      caption: 'Largest Hessian eigenvalue and ε-sharpness of the minimum (a, 1/a), all of which compute the same function. Both are smallest at a = 1 and grow without bound in either direction (Theorem 4), and the ε-flat set contains the whole valley, which is infinitely long (Theorem 2). A flatness measure that is not invariant to such reparameterizations cannot by itself explain generalization.',
      note: 'Dots: values from the check script; lines: 2(a² + 1/a²) and the live ε-sharpness (ε = 0.1).',
      hint: '', legend: [[HEX.pos, 'λ_max = 2(a² + 1/a²)'], [HEX.ball, 'ε-sharpness × 100']],
      plot: { xlabel: 'a', ylabel: 'value', logx: true, logy: true, xlim: [.2, 5], ylim: [1, 100], legend: false },
      build(ctx) {
        const D = ctx.data.rows, P = ctx.plot.clear();
        P.fn(lam, { color: HEX.pos, width: 2.2 }).fn(a => 100 * epsSharp(a), { color: HEX.ball, width: 2.2 })
          .scatter(D.map(r => [r[0], r[1]]), { color: HEX.pos, r: 4 }).scatter(D.map(r => [r[0], 100 * r[2]]), { color: HEX.ball, r: 4 }).draw();
        ctx.stat(D.map(r => `a = ${r[0]}: λ_max ${r[1]}, ε-sharpness ${r[2]}`).join(' · '));
      },
    },
  ],
};
