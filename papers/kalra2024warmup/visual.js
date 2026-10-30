// Kalra & Barkeshli 2024 — warmup lets training tolerate a larger target learning rate: catapults during warmup pull the
// sharpness down to 2/η_t. Adam's preconditioned sharpness starts huge; GI-Adam lowers it by √(1 − β2).
// Toy, success map and Adam numbers: scripts/check-math/kalra2024warmup.py (data.json).
import { def, C, HEX, segPts } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const SIG = .02, U0 = [2, .35], LAM0 = U0[0] ** 2 + U0[1] ** 2, C0 = 6, ET = C0 / LAM0;   // eta_trgt * lambda0 = 6 > 4
const uv = def({ L: (u, v) => .5 * (u * v) ** 2 + .5 * SIG * SIG, g: (u, v) => [u * v * v, v * u * u], dom: [-2.6, 2.6, -2.6, 2.6], h: L => Math.log1p(L) });

export default {
  tabs: [
    {
      id: 'warm', name: 'Warmup', title: 'Ramp the step up and the sharpness gets out of the way',
      caption: 'SGD on the smallest linear network (two examples with labels ±0.02) at a target step 6/λ₀, far above the 2/λ₀ the starting point tolerates. Without warmup the first steps throw it off the surface. With a linear warmup, each time η_t·λ crosses 2 a small catapult lowers the sharpness, so λ tracks 2/η_t down and training continues at the full target step.',
      note: 'f = uv on x = 1, labels +0.02 and −0.02 (one per step). Minima: the axes, sharpness λ = u² + v². Label noise keeps the residual from vanishing; without it GD would sit at zero loss and could not catapult.',
      legend: [[HEX.ball, 'SGD with warmup'], [HEX.good, 'minima (uv = 0)']],
      action: 'Restart', cam: [3.6, 3.8, 4.2], tgt: [0, .2, 0],
      controls: [{ id: 'tw', type: 'seg', label: 'warmup steps', value: 200, options: [[1, 'none'], [20, '20'], [50, '50'], [200, '200'], [400, '400']] }],
      charts: [
        { id: 'lam', title: 'sharpness λ_t and 2/η_t', xlabel: 'step', ylabel: 'λ', ylim: [0, 6], xlim: [0, 600], h: 200, legend: true },
        { id: 'loss', title: 'loss (log)', xlabel: 'step', ylabel: 'L', logy: true, ylim: [1e-4, 1e3], xlim: [0, 600], h: 160, legend: false },
      ],
      build(ctx) {
        const s = ctx.stage; s.surface(uv);
        s.tube(uv, segPts(-2.6, 0, 2.6, 0), C.good); s.tube(uv, segPts(0, -2.6, 0, 2.6), C.good);
        s.marker(uv, U0[0], U0[1], C.white, .08); s.label('λ₀ = 4.12', uv.W(U0[0], U0[1], .35), '', HEX.white);
        this.drop(ctx);
      },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, Tw = +ctx.vals.tw, R = rng(7); s.clearBalls(); ctx.hist = []; ctx.div = false;
        ctx.b = s.ball(uv, {
          start: U0, speed: 1.2, maxSteps: 600,
          step: (p, b) => {
            const eta = ET * Math.min(1, (b.steps + 1) / Tw), yi = R() < .5 ? SIG : -SIG, r = p[0] * p[1] - yi, q = [p[0] - eta * r * p[1], p[1] - eta * r * p[0]];
            if (!isFinite(q[0]) || Math.abs(q[0]) > 2.6 || Math.abs(q[1]) > 2.6) { ctx.div = true; b.done = true; return p; }
            ctx.hist.push([b.steps + 1, eta, q[0] ** 2 + q[1] ** 2, uv.L(q[0], q[1])]); return q;
          },
        });
      },
      frame(ctx) {
        const H = ctx.hist, Tw = +ctx.vals.tw;
        ctx.charts.lam.clear().fn(t => 2 / (ET * Math.min(1, Math.max(t, 1) / Tw)), { color: HEX.neg, dash: [5, 4], label: '2/η_t' })
          .line(H.map(h => [h[0], h[2]]), { color: HEX.ball, width: 1.6, label: 'λ_t' }).draw();
        ctx.charts.loss.clear().line(H.map(h => [h[0], h[3]]), { color: HEX.pos, width: 1.4 }).draw();
        const last = H[H.length - 1];
        ctx.stat(ctx.div ? `η_trgt·λ₀ = 6, warmup ${Tw === 1 ? 'none' : Tw + ' steps'}: diverged at step ${H.length + 1} (left the surface)` : `step ${last?.[0] ?? 0} · η_t = ${(last?.[1] ?? 0).toFixed(3)} (target ${ET.toFixed(3)}) · λ = ${(last?.[2] ?? LAM0).toFixed(3)} vs 2/η_t = ${(2 / (last?.[1] || 1e-9)).toFixed(3)}`);
      },
    },
    {
      id: 'sweep', name: 'Larger targets', mode: '2d', title: 'Longer warmup, larger trainable target step',
      caption: 'For each target step (vertical) and warmup length (horizontal), how many of 10 SGD runs of the same toy survive 1000 steps. Without warmup only η_trgt·λ₀ < 4 works (the catapult limit). Warmup extends the trainable range far beyond it, which the paper identifies as the main benefit of warmup in real networks.',
      note: 'Same two-example linear model and label noise as the first tab; 10 seeds per cell, precomputed in the check script. The paper’s Figs. 3–4 show the analogous maps for FCNs, ResNets and Transformers (reported, not reproduced).',
      hint: '', legend: [[HEX.good, 'all runs train'], [HEX.neg, 'all diverge']],
      plot: { xlabel: 'warmup steps (log)', ylabel: 'η_trgt · λ₀', legend: false, xlim: [-.5, 7.5], ylim: [-.5, 7.5], xfmt: v => [1, 5, 10, 20, 50, 100, 200, 400][Math.round(v)] ?? '', yfmt: v => [2, 3, 4, 5, 6, 8, 10, 12][Math.round(v)] ?? '' },
      build(ctx) {
        const D = ctx.data, z = D.grid.map(row => row.map(v => v));
        const col = v => `rgba(${Math.round(255 - (255 - 95) * v / 10)},${Math.round(93 + (224 - 93) * v / 10)},${Math.round(115 + (176 - 115) * v / 10)},.85)`;
        const P = ctx.plot.clear().heat(z, [-.5, 7.5, -.5, 7.5], { cell: col });
        D.grid.forEach((row, j) => row.forEach((v, i) => P.text(i, j, String(v), { color: '#0b0e1a', align: 'center' })));
        P.hline(2.5, { color: HEX.white, dash: [4, 4], text: 'η_trgt·λ₀ = 4 (no-warmup limit)' }).draw();
        ctx.stat(`η_trgt·λ₀ = 12: ${D.grid[7].map((v, i) => `${D.tws[i]}→${v}`).join(' · ')} (successes of 10)`);
      },
    },
    {
      id: 'adam', name: 'Adam & GI-Adam', mode: '2d', title: 'Adam starts with a huge preconditioned sharpness',
      caption: 'For Adam the stability-relevant curvature is λ_max(P⁻¹H) with P_t = (1 − β1ᵗ)·diag(√v̂_t) + ε. At the first step v̂ is just g₀², so tiny gradient components make P⁻¹H enormous, far above Adam’s threshold (2 + 2β1)/((1 − β1)η). Initializing v₀ = g₀² (GI-Adam) divides it by 1/√(1 − β2) ≈ 32 and acts as a built-in warmup.',
      note: 'Quadratic L = ½Σhᵢθᵢ² in 200 dimensions, curvatures 0.1–10, initial coordinates spread over four decades; η = 10⁻³, β1 = 0.9, β2 = 0.999. Precomputed.',
      hint: '', legend: [[HEX.ball, 'Adam'], [HEX.opt2, 'GI-Adam (v₀ = g₀²)'], [HEX.neg, '(2 + 2β1)/((1 − β1)η)'], [HEX.pos, 'λ_max(H)']],
      plot: { xlabel: 'step', ylabel: 'λ_max(P⁻¹H)', logy: true, ylim: [1, 1e6], xlim: [1, 300], legend: false },
      build(ctx) {
        const D = ctx.data;
        ctx.plot.clear().hline(D.thr, { color: HEX.neg, dash: [5, 4], text: `threshold ${D.thr}` }).hline(9.87, { color: HEX.pos, dash: [3, 4], text: 'λ_max(H) = 9.87' })
          .line(D.adam.map((v, i) => [i + 1, v]), { color: HEX.ball, width: 2 }).line(D.gi.map((v, i) => [i + 1, v]), { color: HEX.opt2, width: 2 }).draw();
        ctx.stat(`step 1: Adam ${D.adam[0].toExponential(2)}, GI-Adam ${D.gi[0].toExponential(2)} · ratio ${(D.gi[0] / D.adam[0]).toFixed(4)} vs √(1 − β2) = ${Math.sqrt(.001).toFixed(4)}`);
      },
    },
  ],
};
