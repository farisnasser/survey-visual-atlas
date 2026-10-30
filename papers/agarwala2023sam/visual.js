// Agarwala & Dauphin 2023 — unnormalized SAM has its own edge of stability, η(λ + ρλ²) = 2, below GD's 2/η, and it
// regularizes the top eigenvalue during training. Checked in scripts/check-math/agarwala2023sam.py (MLP replication in data.json).
import { def, C, HEX, hyperbola } from '../../src/lib/atlas3d.js';

const lamSAM = (eta, rho) => rho <= 0 ? 2 / eta : (-1 + Math.sqrt(1 + 8 * rho / eta)) / (2 * rho);
const uv = def({ L: (u, v) => .5 * (u * v - 1) ** 2, g: (u, v) => { const r = u * v - 1; return [r * v, r * u]; }, hess: (u, v) => [v * v, 2 * u * v - 1, u * u], dom: [-2.6, 2.6, -2.6, 2.6], h: v => Math.log1p(v) });
const top = (u, v) => { const [a, b, c] = uv.hess(u, v); return (a + c) / 2 + Math.hypot((a - c) / 2, b); };
const ETA = .3, Z0 = [2.2, .1];
const samStep = rho => p => { const g = uv.g(p[0], p[1]), q = [p[0] + rho * g[0], p[1] + rho * g[1]], h = uv.g(q[0], q[1]); return [p[0] - ETA * h[0], p[1] - ETA * h[1]]; };
const COL = { '0': HEX.opt2, '0.05': HEX.ball, '0.1': HEX.opt3 };

export default {
  tabs: [
    {
      id: 'uv', name: 'Flatter landing', title: 'SAM cannot settle where GD can',
      caption: 'Loss ½(uv − 1)²: every point of the green hyperbola is a minimum, with sharpness u² + v². From the same unbalanced start and the same step, GD (violet) lands on a fairly sharp minimum, which it may since that sharpness is below 2/η. Unnormalized SAM (coral) is only stable where η(λ + ρλ²) < 2, a stricter edge, and it lands on a flatter part of the valley; a larger radius ρ pushes it further towards the flattest point u = v.',
      note: 'Unnormalized SAM θ ← θ − η∇L(θ + ρ∇L(θ)) (the variant the paper analyses), η = 0.3, start (2.2, 0.1). At ρ = 0.2 it crosses to the other branch of the hyperbola.',
      legend: [[HEX.opt2, 'GD'], [HEX.ball, 'unnormalized SAM'], [HEX.good, 'minima uv = 1']],
      action: 'Restart', cam: [3.2, 4.2, 4.2], tgt: [0, .1, 0],
      controls: [{ id: 'rho', label: 'SAM radius ρ', min: 0, max: .25, step: .01, value: .1 }],
      charts: [{ id: 'S', title: 'sharpness along the runs', xlabel: 'step', ylabel: 'λ_max', ylim: [0, 8], xlim: [0, 200], h: 230, legend: false }],
      build(ctx) {
        const s = ctx.stage; s.surface(uv); s.tube(uv, hyperbola(1, 2.6, 1), C.good); s.tube(uv, hyperbola(1, 2.6, -1), C.good);
        s.marker(uv, 1, 1, C.white, .07); this.drop(ctx);
      },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, rho = ctx.vals.rho; s.clearBalls(); ctx.H = { g: [[0, top(...Z0)]], s: [[0, top(...Z0)]] };
        ctx.bg = s.ball(uv, { start: Z0, speed: .5, maxSteps: 2000, step: samStep(0), color: C.opt2 }); ctx.bg.onStep = b => ctx.H.g.push([b.steps, top(...b.p)]);
        ctx.bs = s.ball(uv, { start: Z0, speed: .5, maxSteps: 2000, step: samStep(rho) }); ctx.bs.onStep = b => ctx.H.s.push([b.steps, top(...b.p)]);
      },
      frame(ctx) {
        const rho = ctx.vals.rho, ls = lamSAM(ETA, rho);
        ctx.charts.S.clear().hline(2 / ETA, { color: HEX.opt2, dash: [5, 4], text: '2/η' }).hline(ls, { color: HEX.ball, dash: [5, 4], text: 'SAM edge' })
          .line(ctx.H.g, { color: HEX.opt2, width: 1.6 }).line(ctx.H.s, { color: HEX.ball, width: 1.6 }).draw();
        const sg = ctx.bg.p[0] ** 2 + ctx.bg.p[1] ** 2, ss = ctx.bs.p[0] ** 2 + ctx.bs.p[1] ** 2;
        ctx.stat(`step ${ctx.bg.steps} · GD at sharpness u² + v² = ${sg.toFixed(3)} (edge 2/η = ${(2 / ETA).toFixed(2)}) · SAM ρ = ${rho.toFixed(2)} at ${ss.toFixed(3)} (edge ${ls.toFixed(3)})`);
      },
    },
    {
      id: 'mlp', name: 'SAM edge of stability', mode: '2d', title: 'In a trained network the top eigenvalue sits on the SAM edge',
      caption: 'Our offline replication: the same tanh network as on the Cohen et al. page, trained by full-batch GD and by unnormalized SAM at η = 0.3. GD’s sharpness climbs to 2/η = 6.67 and stays; with SAM it stops lower, at the value λ that solves η(λ + ρλ²) = 2 (dashed): 5.27 for ρ = 0.05 and 4.57 for ρ = 0.1. The regularization acts during training, far from any minimum.',
      note: 'tanh MLP 10–48–48–1, 128 synthetic points, MSE, 3000 steps, sharpness every 10 steps by power iteration on exact-gradient HVPs. The paper shows the same for a fully connected net on 2-class CIFAR and a WideResNet 28-10 on CIFAR-10 (reported).',
      hint: '', legend: [[HEX.opt2, 'GD'], [HEX.ball, 'SAM ρ = 0.05'], [HEX.opt3, 'SAM ρ = 0.1']],
      plot: { xlabel: 'iteration', ylabel: 'λ_max(∇²L)', xlim: [0, 3000], ylim: [3.5, 7.6], legend: false },
      charts: [{ id: 'loss', title: 'train loss (log)', xlabel: 'iteration', ylabel: 'loss', logy: true, ylim: [1e-7, 1], h: 240, legend: false }],
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear(), lc = ctx.charts.loss.clear();
        for (const [k, rows] of Object.entries(D.curves)) {
          P.hline(lamSAM(D.eta, +k), { color: COL[k], dash: [5, 4], alpha: .8, text: +k ? `η(λ + ${k}λ²) = 2: ${lamSAM(D.eta, +k).toFixed(2)}` : `2/η = ${(2 / D.eta).toFixed(2)}` });
          P.line(rows.map(r => [r[0], r[2]]), { color: COL[k], width: 1.8 }); lc.line(rows.map(r => [r[0], r[1]]), { color: COL[k], width: 1.4 });
        }
        P.draw(); lc.draw();
        const late = k => { const r = D.curves[k].slice(-100); return r.reduce((a, x) => a + x[2], 0) / r.length; };
        ctx.stat(`late sharpness: GD ${late('0').toFixed(3)} (2/η = 6.667) · SAM ρ = 0.05: ${late('0.05').toFixed(3)} (edge 5.275) · SAM ρ = 0.1: ${late('0.1').toFixed(3)} (edge 4.574)`);
      },
    },
    {
      id: 'edge', name: 'Where the edge sits', mode: '2d', title: 'SAM’s edge falls with the radius ρ',
      caption: 'On a quadratic, unnormalized SAM multiplies each eigen-component by 1 − η(λ + ρλ²) (Eq. 12), so it is stable only for λ below (−1 + √(1 + 8ρ/η))/(2ρ). At ρ = 0 this is GD’s 2/η; any ρ > 0 lowers it, more so for small η.',
      note: 'Exact formula; the dot marks the chosen η and ρ.',
      hint: '', legend: [[HEX.ball, 'SAM edge λ_SAM(ρ)'], [HEX.opt2, 'GD edge 2/η']],
      controls: [{ id: 'eta', label: 'learning rate η', min: .05, max: 1, step: .01, value: .3 }, { id: 'rho', label: 'radius ρ', min: 0, max: .5, step: .01, value: .1 }],
      plot: { xlabel: 'radius ρ', ylabel: 'largest stable eigenvalue', xlim: [0, .5], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const { eta, rho } = ctx.vals, P = ctx.plot.clear().set({ ylim: [0, 2 / eta * 1.08] });
        P.hline(2 / eta, { color: HEX.opt2, dash: [5, 4], text: `2/η = ${(2 / eta).toFixed(2)}` }).fn(r => lamSAM(eta, r), { color: HEX.ball, width: 2.4 }).scatter([[rho, lamSAM(eta, rho)]], { color: HEX.white, r: 5 }).draw();
        ctx.stat(`η = ${eta.toFixed(2)}, ρ = ${rho.toFixed(2)}: λ_SAM = ${lamSAM(eta, rho).toFixed(3)} vs 2/η = ${(2 / eta).toFixed(3)} · notes’ example η = 0.5, ρ = 0.2 → ${lamSAM(.5, .2).toFixed(2)}`);
      },
    },
  ],
};
