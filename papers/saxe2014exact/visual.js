// Saxe, McClelland, Ganguli 2014 — exact dynamics of deep linear nets. Checked in scripts/check-math/saxe2014exact.py
import { def, opt, C, HEX, hyperbola } from '../../src/lib/atlas3d.js';
import { rng, eigSym, linspace, logspace } from '../../src/lib/math.js';

const mode = s => def({ L: (a, b) => .5 * (s - a * b) ** 2, g: (a, b) => { const r = s - a * b; return [-r * b, -r * a]; }, h: L => Math.log1p(L) * 1.1 });
const uAnalytic = (t, s, u0) => { const e = Math.exp(2 * s * t); return s * e / (e - 1 + s / u0); };   // Eq. (12), tau = 1

// ---- random matrices (tab 3) ----
function gauss(N, r) { return Array.from({ length: N }, () => Array.from({ length: N }, () => r.normal() / Math.sqrt(N))); }
function orth(N, r) { // Gram-Schmidt on a Gaussian matrix -> Haar-ish orthogonal
  const A = gauss(N, r), Q = [];
  for (let j = 0; j < N; j++) {
    const v = A.map(row => row[j]);
    for (const q of Q) { const d = v.reduce((s, x, i) => s + x * q[i], 0); for (let i = 0; i < N; i++) v[i] -= d * q[i]; }
    const n = Math.hypot(...v); Q.push(v.map(x => x / n));
  }
  return Array.from({ length: N }, (_, i) => Q.map(q => q[i]));
}
const mul = (A, B) => A.map(row => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
function singulars(W) { const N = W.length, G = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => { let s = 0; for (let k = 0; k < N; k++) s += W[k][i] * W[k][j]; return s; })); return eigSym(G, { sweeps: 40 }).values.map(v => Math.sqrt(Math.max(v, 1e-300))); }

export default {
  tabs: [
    {
      id: 'hyperbolas', name: 'Hyperbolas', title: 'Learning follows hyperbolas',
      caption: 'One input–output mode: loss ½(s − ab)². Gradient flow conserves a² − b², so every ball slides along a hyperbola (faint) into the valley ab = s.',
      note: 'Scalars a, b are the projections of the two weight layers on one mode (paper Eq. 8). s = 1.',
      legend: [[HEX.ball, 'gradient descent'], [HEX.good, 'ab = s (all global minima)'], ['#9aa6cc', 'a² − b² = const']],
      action: 'Drop from small init', cam: [0, 4.6, 5.2], tgt: [0, .2, 0],
      controls: [{ id: 'init', label: 'init scale', min: .02, max: .8, step: .01, value: .12 }],
      build(ctx) {
        const s = ctx.stage, S = ctx.S = s.surface(mode(1));
        s.tube(S, hyperbola(1, 1.6, 1), C.good); s.tube(S, hyperbola(1, 1.6, -1), C.good);
        for (const c of [-1.6, -.8, -.3, .3, .8, 1.6]) for (const sg of [1, -1]) {
          const pts = []; for (let t = -1.6; t <= 1.6; t += .02) { if (c > 0) pts.push([sg * Math.sqrt(c + t * t), t]); else pts.push([t, sg * Math.sqrt(-c + t * t)]); }
          s.line(S, pts, 0x9aa6cc, .35);
        }
        s.marker(S, 0, 0, C.white); s.label('unstable fixed point', S.W(0, 0, .45), '', HEX.muted);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, r = rng(Date.now() % 1e6), k = ctx.vals.init; s.clearBalls();
        ctx.c0 = [];
        for (let i = 0; i < 8; i++) { const p = [r.uniform(-k, k), r.uniform(-k, k)]; ctx.c0.push(p[0] ** 2 - p[1] ** 2); s.ball(ctx.S, { start: p, step: opt.gd(.015), speed: 5, stop: b => b.loss() < 1e-7 }); }
      },
      frame(ctx) {
        const b = ctx.stage.balls[0]; if (!b) return;
        ctx.stat(`ball 1: a² − b² = ${(b.p[0] ** 2 - b.p[1] ** 2).toFixed(4)} (start ${ctx.c0[0].toFixed(4)}) · ab = ${(b.p[0] * b.p[1]).toFixed(3)}`);
      },
    },
    {
      id: 'plateaus', name: 'Plateaus', mode: '2d', title: 'Long plateau, then a sudden jump',
      caption: 'Mode strength u(t) from Eq. 12 for four modes. Stronger modes are learned first; smaller initial weights mean longer plateaus.',
      note: 'Lines: exact solution. Dots: discrete gradient descent on ½(s − ab)² from a = b = √u₀ (step 0.01, t = k·0.01).',
      legend: [], hint: 'move u₀ to change the plateau length',
      controls: [{ id: 'lu0', label: 'initial strength u₀', min: -6, max: -1, step: .1, value: -4, fmt: v => `1e${(+v).toFixed(1)}` }],
      plot: { xlabel: 'time t (units of τ)', ylabel: 'mode strength u = ab', xlim: [0, 8], ylim: [0, 3.3] },
      build(ctx) { this.draw(ctx); },
      onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const P = ctx.plot, u0 = 10 ** ctx.vals.lu0, cols = [HEX.ball, HEX.flat, HEX.pos, HEX.opt2];
        P.clear();
        [3, 2, 1, .5].forEach((s, i) => {
          P.hline(s, { color: cols[i], dash: [3, 5], alpha: .5, noRange: true });
          P.fn(t => uAnalytic(t, s, u0), { color: cols[i], width: 2.2, label: `s = ${s}` });
          const lr = .01, pts = []; let a = Math.sqrt(u0), b = a;
          for (let k = 0; k <= 800; k++) { if (k % 25 === 0) pts.push([k * lr, a * b]); const r = s - a * b; [a, b] = [a + lr * r * b, b + lr * r * a]; }
          P.scatter(pts, { color: cols[i], r: 2.6 });
          const tl = Math.log(s / u0) / (2 * s);
          P.vline(tl, { color: cols[i], alpha: .35, dash: [2, 3] });
        });
        P.draw();
        ctx.stat(`learning time ≈ (1/2s)·ln(s/u₀): s=3 → ${(Math.log(3 / u0) / 6).toFixed(2)}, s=0.5 → ${(Math.log(.5 / u0) / 1).toFixed(2)}`);
      },
    },
    {
      id: 'orthogonal', name: 'Orthogonal init', mode: '2d', title: 'Why orthogonal initialization helps depth',
      caption: 'Singular values of W_D ⋯ W_1 for random N×N layers. Scaled Gaussian layers keep the norm on average but squash most directions; orthogonal layers keep every singular value at 1.',
      note: 'Live: N = 60, 6 independent products per setting (the paper uses N = 1000, Fig. 6B).',
      legend: [[HEX.ball, 'Gaussian, std 1/√N'], [HEX.good, 'orthogonal']], hint: '',
      controls: [{ id: 'D', label: 'depth D', min: 1, max: 40, step: 1, value: 8, live: false, fmt: v => (+v).toFixed(0) }],
      plot: { xlabel: 'log₁₀ singular value', ylabel: 'fraction of singular values', xlim: [-8, 1.5], ylim: [0, .35] },
      build(ctx) { this.compute(ctx); },
      onControl(ctx) { this.compute(ctx); },
      compute(ctx) {
        const N = 60, D = ctx.vals.D, r = rng(D * 7 + 1), g = [], o = [];
        for (let rep = 0; rep < 6; rep++) {
          let Wg = gauss(N, r), Wo = orth(N, r);
          for (let k = 1; k < D; k++) { Wg = mul(gauss(N, r), Wg); Wo = mul(orth(N, r), Wo); }
          g.push(...singulars(Wg)); o.push(...singulars(Wo));
        }
        const P = ctx.plot, bins = 48, lo = -8, hi = 1.5, hist = vals => { const c = new Array(bins).fill(0); vals.forEach(v => { const k = Math.floor((Math.log10(v) - lo) / (hi - lo) * bins); c[Math.max(0, Math.min(bins - 1, k))]++; }); return c.map((n, k) => [lo + (k + .5) * (hi - lo) / bins, n / vals.length]); };
        P.clear();
        P.line(hist(g), { color: HEX.ball, width: 2, step: true });
        P.line(hist(o), { color: HEX.good, width: 2, step: true });
        P.vline(0, { color: HEX.muted, dash: [3, 4], text: 'σ = 1' });
        P.draw();
        const sg = [...g].sort((a, b) => a - b);
        ctx.stat(`depth ${D}: Gaussian median σ = ${sg[sg.length >> 1].toExponential(1)}, max ${sg[sg.length - 1].toFixed(2)} · orthogonal all σ = ${Math.min(...o).toFixed(4)}…${Math.max(...o).toFixed(4)}`);
      },
    },
  ],
};
