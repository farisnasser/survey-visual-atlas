// Lu, Ma, Lu, Lu, Ying 2020 — a ResNet as a continuous-depth flow whose residual blocks are particles of a measure ρ(θ, t).
// Forward trajectories, adjoint norms and loss curves come from scripts/check-math/lu2020mean.py (data.json):
// X_{n+1} = X_n + (1/L)(1/M) Σ_j a_nj tanh(w_nj·X_n + b_nj), X_0 = (x, 0), output X_L,1, target sin(πx), L = 16, M = 32.
import { C, HEX } from '../../src/lib/atlas3d.js';

const THREE = window.THREE;
const SX = 1.1, SY = 2.4, SZ = 1.1;          // world scale: X₁ → x, depth t → y, X₂ → z
const W = (x1, x2, t) => new THREE.Vector3(x1 * SX, t * SY - 1.0, -x2 * SZ);
const mix = (u) => { const a = [0xff, 0x7a, 0x59], b = [0xb2, 0x8c, 0xff]; return new THREE.Color(...a.map((v, k) => (v + (b[k] - v) * u) / 255)); };
const LCOL = { 4: HEX.opt3, 16: HEX.ball, 64: HEX.opt2 };

export default {
  tabs: [
    {
      id: 'flow', name: 'Data flowing through depth', title: 'A deep ResNet transports the data along a continuous flow; training reshapes the flow',
      caption: 'Each curve is one training input x ∈ [−1, 1] carried through 16 residual layers (depth runs upward), starting at (x, 0). The network reads out the first coordinate at the top and should output sin(πx); green dots mark where each curve must end in that coordinate. Slide through training: the flow, made of 16 × 32 small residual blocks, bends until every curve lands on its target.',
      note: 'Every block contributes 1/(L·M) of a tanh feature, the mean-field scaling of the paper’s model (3); all 1 536 block parameters are trained by GD (step 0.6·L·M), w₂ and w₁ fixed. Curve colour encodes the target value.',
      legend: [[HEX.ball, 'target −1 … '], [HEX.opt2, '… target +1'], [HEX.good, 'required output']],
      cam: [3.6, 2.4, 3.2], tgt: [0, .2, 0],
      controls: [{ id: 'k', label: 'training step', min: 0, max: 3000, step: 300, value: 3000, fmt: v => String(Math.round(v)), live: true }],
      actions: [{ label: 'Play training', fn: ctx => { ctx.play = true; ctx.acc = 0; ctx.set('k', 0); ctx.shell.tab.draw(ctx); } }],
      build(ctx) {
        const s = ctx.stage; s.clear(); ctx.objs = [];
        const grid = []; for (let i = -1; i <= 1.0001; i += .5) { grid.push(W(i, -1, 0), W(i, 1, 0)); grid.push(W(-1, i, 0), W(1, i, 0)); }
        s.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(grid), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: .12 })));
        s.segment(W(-1.3, 0, 1), W(1.3, 0, 1), C.muted, .5);
        s.label('depth t = 0', W(1.25, -1, 0), '', HEX.muted); s.label('t = 1 (read out X₁)', W(1.25, -1, 1), '', HEX.muted);
        this.draw(ctx);
      },
      draw(ctx) {
        const D = ctx.data, s = ctx.stage, fr = D.frames[Math.round(ctx.vals.k / 300)];
        (ctx.objs || []).forEach(o => s.remove(o)); ctx.objs = [];
        const L = fr.length - 1;
        for (let i = 0; i < D.xs.length; i++) {
          const pts = fr.map((X, n) => W(X[i][0], X[i][1], n / L));
          const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: mix((D.ys[i] + 1) / 2) }));
          ctx.objs.push(s.add(line));
          ctx.objs.push(s.dot(W(D.ys[i], fr[L][i][1], 1), C.good, .025));
        }
        const E = D.H16[Math.round(ctx.vals.k / 10)];
        const err = Math.max(...D.xs.map((_, i) => Math.abs(fr[L][i][0] - D.ys[i])));
        ctx.stat(`training step ${Math.round(ctx.vals.k)} · loss E = ${E.toExponential(2)} · largest output error ${err.toFixed(3)}`);
      },
      frame(ctx, dt) {
        if (!ctx.play) return; ctx.acc += dt * 2.5; if (ctx.acc < 1) return; ctx.acc = 0;
        const k = Math.min(3000, ctx.vals.k + 300); ctx.set('k', k); this.draw(ctx); if (k >= 3000) ctx.play = false;
      },
      onControl(ctx) { ctx.play = false; this.draw(ctx); },
    },
    {
      id: 'depth', name: 'Depth limit & gradients', mode: '2d', title: 'Deeper nets train the same way, and the gradient signal survives depth',
      caption: 'Left: training loss for depth L = 4, 16 and 64 with the same 1/L scaling (32 blocks per layer): the curves approach a depth-independent limit, the continuous model the paper analyses. Right: the adjoint ‖p(·, t)‖, the backpropagated sensitivity of the loss to the features at depth t, relative to its value at the output, at several moments of training. It never collapses — the content of Lemma 1, ‖p(·, t)‖ ≥ e^{−(C₁ + C₂r)}E(ρ), which is why a deep ResNet’s gradient behaves like a two-layer network’s.',
      note: 'Same data and readout as the first tab. Theorem 3: whenever E(ρ) > 0 a descent direction exists, so every local minimizer of the continuous model is global; Theorem 4 (gradient flow reaches it) needs homogeneity and support assumptions.',
      hint: '', legend: [[HEX.opt3, 'L = 4'], [HEX.ball, 'L = 16'], [HEX.opt2, 'L = 64']],
      plot: { xlabel: 'GD step', ylabel: 'loss E', logy: true, xlim: [0, 1500], legend: false },
      charts: [{ id: 'adj', title: '‖p(·, t)‖ / ‖p(·, 1)‖ (L = 16)', xlabel: 'depth t', ylabel: 'ratio', ylim: [0, 1.6], h: 240 }],
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear();
        for (const L of [4, 16, 64]) P.line(D.curves[L].map((v, i) => [i * 10, v]), { color: LCOL[L], width: L === 16 ? 2.4 : 1.8 });
        P.draw();
        const ch = ctx.charts.adj.clear().hline(.3, { color: HEX.neg, dash: [4, 4], text: '0.3' });
        D.adj.slice(0, -1).forEach((a, j) => ch.line(a.map((v, n) => [n / (a.length - 1), v]), { color: j ? HEX.ball : HEX.opt3, width: 1.4, alpha: .4 + .06 * j }));
        ch.draw();
        ctx.stat(`loss after 1 500 steps: L = 4: ${D.curves[4].at(-1).toExponential(1)} · L = 16: ${D.curves[16].at(-1).toExponential(1)} · L = 64: ${D.curves[64].at(-1).toExponential(1)} · smallest adjoint ratio over depth and training: ${Math.min(...D.adj.slice(0, -1).flat()).toFixed(2)}`);
      },
    },
  ],
};
