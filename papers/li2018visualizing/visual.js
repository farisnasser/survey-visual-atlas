// Li et al. 2018 — filter-normalized loss slices of a real (tiny) ReLU network.
// Network trained offline: scripts/check-math/li2018visualizing.py (weights in data.json). Slices are computed live.
import { def, opt, C, HEX, statLabels } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

function makeNet(D) {
  const X = D.X, y = D.y, H = D.b1.length;
  const loss = (W1, b1, W2, b2) => {
    let s = 0;
    for (let n = 0; n < X.length; n++) {
      let out = b2;
      for (let i = 0; i < H; i++) { const z = W1[i][0] * X[n][0] + W1[i][1] * X[n][1] + b1[i]; if (z > 0) out += W2[i] * z; }
      s += (out - y[n]) ** 2;
    }
    return .5 * s / X.length;
  };
  return { H, loss };
}
function directions(seed, W1, W2, normalize) {
  const r = rng(seed), H = W1.length;
  let dW1 = W1.map(() => [r.normal(), r.normal()]), dW2 = W2.map(() => r.normal());
  if (normalize) {
    dW1 = dW1.map((d, i) => { const nd = Math.hypot(...d), nw = Math.hypot(...W1[i]); return d.map(v => v / nd * nw); });
    const nd = Math.hypot(...dW2), nw = Math.hypot(...W2); dW2 = dW2.map(v => v / nd * nw);
  }
  return { dW1, dW2 };   // biases are left unperturbed
}
function sliceDef(net, P, normalize, ox) {
  const d1 = directions(1, P.W1, P.W2, normalize), d2 = directions(2, P.W1, P.W2, normalize);
  const L = (a, b) => net.loss(P.W1.map((w, i) => [w[0] + a * d1.dW1[i][0] + b * d2.dW1[i][0], w[1] + a * d1.dW1[i][1] + b * d2.dW1[i][1]]), P.b1,
    P.W2.map((w, i) => w + a * d1.dW2[i] + b * d2.dW2[i]), P.b2);
  return def({ L, h: v => Math.log1p(v * 4) * .28, dom: [-.3, .3, -.3, .3], N: 64, ox, size: 3.2, fit: null });
}

export default {
  tabs: [
    {
      id: 'scale', name: 'Scale illusion', title: 'Same network, different-looking sharpness',
      caption: 'Left: a trained 2-16-1 ReLU net. Right: the same function with layer 1 scaled by c and layer 2 by 1/c. Raw random directions make the copies look different; filter-normalized directions make them identical.',
      note: 'Slice f(α, β) = L(θ + αδ + βη); filter = incoming weights of one neuron; same random seed on both sides; biases unperturbed.',
      legend: [[HEX.white, 'minimizer θ (centre)']], cam: [0, 5.2, 4.2], tgt: [0, .2, 0],
      controls: [
        { id: 'c', label: 'rescale c', min: 1, max: 6, step: .5, value: 4, live: false, digits: 1 },
        { id: 'norm', type: 'seg', label: 'directions', value: 'raw', options: [['raw', 'raw Gaussian'], ['filter', 'filter-normalized']] },
      ],
      build(ctx) {
        const D = ctx.data, net = makeNet(D), c = ctx.vals.c, nz = ctx.vals.norm === 'filter';
        const P = { W1: D.W1, b1: D.b1, W2: D.W2, b2: D.b2 }, Pc = { W1: D.W1.map(w => w.map(v => v * c)), b1: D.b1.map(v => v * c), W2: D.W2.map(v => v / c), b2: D.b2 };
        const s = ctx.stage, A = s.surface(sliceDef(net, P, nz, -1.8)), B = s.surface(sliceDef(net, Pc, nz, 1.8));
        s.marker(A, 0, 0, C.white, .07); s.marker(B, 0, 0, C.white, .07);
        const curv = d => (d.L(.05, 0) + d.L(-.05, 0) + d.L(0, .05) + d.L(0, -.05) - 4 * d.L(0, 0)) / (.05 * .05);
        const L = statLabels(s, [['original θ', A.W(0, -.38, 0).setY(0)], [`rescaled · c = ${c}`, B.W(0, -.38, 0).setY(0)]]);
        L[0].stat(`sharpness proxy ${curv(A).toFixed(2)}`); L[1].stat(`sharpness proxy ${curv(B).toFixed(2)}`);
        ctx.stat(`train loss ${D.train_loss.toFixed(4)} on both (identical functions) · sharpness proxy: left ${curv(A).toFixed(2)}, right ${curv(B).toFixed(2)}${nz ? ' — equal, as filter normalization promises' : ''}`);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
    },
    {
      id: 'hidden', name: 'Hidden saddle', title: 'A bowl in 2D can hide a saddle',
      caption: 'Slice through L(x, y, z) = ½(x² + y² − 2z²) spanned by x and a direction tilted from y toward z. Untilted it is a perfect bowl; tilt it and the negative curvature shows up.',
      note: 'Teaching example from our notes: a 2D slice cannot certify convexity (paper Sec. 6). Hessian = diag(1, 1, −2).',
      legend: [[HEX.neg, 'λ < 0 in the slice'], [HEX.pos, 'λ > 0'], [HEX.flat, 'λ ≈ 0']], cam: [3.0, 3.0, 4.0], tgt: [0, .3, 0],
      controls: [{ id: 'phi', label: 'tilt φ', min: 0, max: 90, step: 1, value: 0, fmt: v => `${(+v).toFixed(0)}°` }],
      build(ctx) {
        const p = ctx.vals.phi * Math.PI / 180, cb = Math.cos(p) ** 2 - 2 * Math.sin(p) ** 2;
        const S = def({ L: (a, b) => .5 * (a * a + cb * b * b), g: (a, b) => [a, cb * b], hess: () => [1, 0, cb], h: v => v * .9 + 1.1, dom: [-1.4, 1.4, -1.4, 1.4] });
        const s = ctx.stage; s.surface(S); s.eigenLines(S, 0, 0, { len: 1.1, eps: .02 }); s.marker(S, 0, 0, C.white);
        ctx.stat(`curvature along the tilted direction: cos²φ − 2 sin²φ = ${cb.toFixed(3)}${cb < 0 ? ' < 0 → saddle' : ' ≥ 0 → looks convex'} (turns negative at φ = 35.3°)`);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
    },
    {
      id: 'evidence', name: 'Paper’s evidence', mode: '2d', title: 'Skip connections and width (reported results)',
      caption: 'CIFAR-10 test error reported in the paper. Without skips, deeper ResNets get worse and their filter-normalized landscapes look chaotic; with skips and more width, landscapes look smooth and error drops.',
      note: 'Numbers copied from Figs. 5 and 6 of the paper, not computed here.',
      hint: '', legend: [[HEX.pos, 'with skip connections'], [HEX.ball, 'without skips']],
      controls: [{ id: 'which', type: 'seg', label: 'comparison', value: 'depth', options: [['depth', 'depth (Fig. 5)'], ['width', 'width (Fig. 6)']] }],
      plot: { ylabel: 'test error (%)', ylim: [0, 18], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const R = ctx.data.reported[ctx.vals.which], items = [];
        R.labels.forEach((l, i) => { items.push({ y: R.skip[i], color: HEX.pos, off: -.5, w: .48 }); });
        const P = ctx.plot.clear().set({ xlim: [-.6, R.labels.length - .4], xfmt: v => R.labels[Math.round(v)] ?? '' });
        P.bars(R.skip.map(v => ({ y: v, color: HEX.pos, off: 0, w: .49 })), { width: .7 });
        P.bars(R.noskip.map(v => ({ y: v, color: HEX.ball, off: .51, w: .49 })), { width: .7 });
        R.labels.forEach((l, i) => { P.text(i - .19, R.skip[i] + .5, `${R.skip[i]}`, { color: HEX.pos, align: 'center' }); P.text(i + .19, R.noskip[i] + .5, `${R.noskip[i]}`, { color: HEX.ball, align: 'center' }); });
        P.draw();
        ctx.stat(ctx.vals.which === 'depth' ? 'ResNet-110: 5.79% with skips vs 16.44% without' : 'Wide-ResNet-56, k = 8: 3.93% with skips vs 8.70% without');
      },
    },
  ],
};
