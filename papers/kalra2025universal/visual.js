// Kalra, He, Barkeshli 2025 — UV model: sharpness reduction, progressive sharpening, edge of stability, chaos.
// Eqs. (1)-(2) and the eta_c threshold are checked in scripts/check-math/kalra2025universal.py
import { def, C, HEX, hyperbola, statLabels } from '../../src/lib/atlas3d.js';

const Y = 2;   // target, x = 1, n_eff = 1  =>  eta_c = sqrt(n_eff)/(x y) = 0.5
const uv = def({ L: (u, v) => .5 * (u * v - Y) ** 2, g: (u, v) => { const r = u * v - Y; return [r * v, r * u]; }, h: L => Math.log1p(L) * .9, dom: [-2.6, 2.6, -2.6, 2.6] });
// Eqs. (1)-(2): residual f~ and lambda = Tr H
const step = (r, l, eta, ne, x2 = 1, y = Y) => [r * (1 - eta * l + eta * eta * x2 / ne * r * (r + y)), l + eta * x2 * r / ne * (eta * l * r - 4 * (r + y))];

export default {
  tabs: [
    {
      id: 'bounce', name: 'Edge of stability', title: 'Above η_c the ball never settles',
      caption: 'Loss ½(uv − 2)² of the smallest UV model. Every point of the green valley is a minimum with sharpness u² + v². GD can only settle where sharpness < 2/η; the flattest point has 4, so for η > η_c = 0.5 it bounces forever.',
      note: 'n = 1, x = 1, y = 2: this is the UV model of the paper in its smallest form. Start (0.1, 0.12) near the saddle at the origin.',
      legend: [[HEX.ball, 'gradient descent'], [HEX.good, 'uv = 2 (minima)'], ['#ffffff', 'flattest minimum (√2, √2)']],
      action: 'Restart', cam: [3.4, 3.6, 4.4], tgt: [0, .3, 0],
      controls: [{ id: 'eta', label: 'step size η', min: .3, max: .85, step: .01, value: .55 }],
      charts: [
        { id: 'lam', title: 'sharpness λ = u² + v² (= Tr H on the valley)', xlabel: 'step', ylabel: 'λ', ylim: [0, 10], h: 210, legend: true },
        { id: 'loss', title: 'loss', xlabel: 'step', ylabel: 'L', logy: true, ylim: [1e-12, 10], h: 170, legend: false },
      ],
      build(ctx) {
        const s = ctx.stage; s.surface(uv);
        s.tube(uv, hyperbola(Y, 2.6, 1), C.good); s.tube(uv, hyperbola(Y, 2.6, -1), C.good);
        s.marker(uv, Math.SQRT2, Math.SQRT2, C.white, .08); s.marker(uv, 0, 0, C.white, .06);
        s.label('saddle', uv.W(0, 0, .35), '', HEX.muted); s.label('flattest: λ = 4', uv.W(Math.SQRT2, Math.SQRT2, .3), '', HEX.white);
        this.drop(ctx);
      },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, eta = ctx.vals.eta; s.clearBalls(); ctx.hist = [];
        ctx.b = s.ball(uv, { start: [.1, .12], speed: .35, maxSteps: 1e6, step: (p, b) => { const [gu, gv] = uv.g(p[0], p[1]); const q = [p[0] - eta * gu, p[1] - eta * gv]; ctx.hist.push([b.steps, q[0] ** 2 + q[1] ** 2, uv.L(q[0], q[1])]); return q; } });
      },
      frame(ctx) {
        const eta = ctx.vals.eta, H = ctx.hist.slice(-400);
        ctx.charts.lam.clear().set({ xlim: [Math.max(0, (H[0]?.[0] ?? 0)), Math.max(40, H[H.length - 1]?.[0] ?? 40)] })
          .hline(2 / eta, { color: HEX.neg, dash: [5, 4], text: `2/η = ${(2 / eta).toFixed(2)}` }).hline(4, { color: HEX.white, dash: [2, 4], alpha: .6 })
          .line(H.map(h => [h[0], h[1]]), { color: HEX.ball, width: 1.6, label: 'λ_t' }).draw();
        ctx.charts.loss.clear().set({ xlim: [Math.max(0, (H[0]?.[0] ?? 0)), Math.max(40, H[H.length - 1]?.[0] ?? 40)] }).line(H.map(h => [h[0], Math.max(h[2], 1e-12)]), { color: HEX.pos, width: 1.6 }).draw();
        const last = ctx.hist.slice(-32).map(h => h[1].toFixed(3)), uniq = new Set(last).size;
        ctx.stat(`η = ${eta.toFixed(2)} ${eta < .5 ? '< η_c' : '> η_c'} · ${ctx.hist.length < 300 ? 'transient…' : uniq === 1 ? `settled at λ = ${last[0]}` : uniq <= 16 ? `period-${uniq} oscillation` : 'chaotic oscillation'}`);
      },
    },
    {
      id: 'flow', name: 'Function-space map', mode: '2d', title: 'Two numbers describe the whole UV model',
      caption: 'Eqs. (1)–(2) in the (residual f̃, sharpness λ) plane. Warm cells: λ grows (progressive sharpening); cool cells: λ shrinks. Click to start a trajectory. The zero-loss line f̃ = 0 is stable only below λ = 2/η (green).',
      note: 'x = 1, y = 2. Fixed point II at (−y, 0) is the parameter-space origin (a saddle). Each dot is one GD step of the full model.',
      hint: 'click the plane to launch a trajectory', legend: [],
      controls: [{ id: 'eta', label: 'η / η_c', min: .4, max: 1.7, step: .02, value: .9 }, { id: 'lne', label: 'effective width n_eff', min: 0, max: 3, step: .1, value: 0, fmt: v => (10 ** v).toFixed(v < 1 ? 1 : 0) }],
      plot: { xlim: [-3.2, 3.2], ylim: [0, 12], xlabel: 'residual f̃ = f − y', ylabel: 'λ = Tr H', legend: false },
      build(ctx) {
        ctx.starts = [[-1.96, .4], [-1.9, 7]];
        ctx.listen(document.querySelector('#plot2d'), 'pointerdown', ev => { ctx.starts.push(ctx.plot.toData(ev)); if (ctx.starts.length > 6) ctx.starts.shift(); this.draw(ctx); });
        this.draw(ctx);
      },
      onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const ne = 10 ** ctx.vals.lne, etac = Math.sqrt(ne) / Y, eta = ctx.vals.eta * etac, P = ctx.plot.clear(), cells = [], nx = 64, ny = 48;
        P.set({ ylim: [0, Math.max(12, 3 / eta)] });
        const [x0, x1] = [-3.2, 3.2], y1 = Math.max(12, 3 / eta);
        for (let j = 0; j < ny; j++) { const row = []; for (let i = 0; i < nx; i++) { const r = x0 + (x1 - x0) * (i + .5) / nx, l = y1 * (j + .5) / ny; row.push(step(r, l, eta, ne)[1] - l); } cells.push(row); }
        P.heat(cells, [x0, x1, 0, y1], { cell: d => d > 0 ? 'rgba(255,200,87,.13)' : 'rgba(124,196,255,.08)' });
        const ar = []; for (let i = 0; i < 15; i++) for (let j = 0; j < 11; j++) { const r = x0 + (x1 - x0) * (i + .5) / 15, l = y1 * (j + .5) / 11, [r1, l1] = step(r, l, eta, ne), n = Math.hypot((r1 - r) / 6.4, (l1 - l) / y1) || 1; ar.push([r, l, (r1 - r) / n * .16, (l1 - l) / n * .16 * y1 / 6.4]); }
        P.arrows(ar, { color: 'rgba(142,153,189,.55)' });
        P.line([[0, 0], [0, Math.min(2 / eta, y1)]], { color: HEX.good, width: 3 }).line([[0, 2 / eta], [0, y1]], { color: HEX.flat, width: 3, dash: [4, 3] });
        P.hline(2 / eta, { color: HEX.neg, dash: [5, 4], text: `λ = 2/η` });
        P.scatter([[-Y, 0]], { color: HEX.white, r: 5 }).text(-Y, .5, 'II · saddle', { color: HEX.white, align: 'center' });
        const cols = [HEX.ball, HEX.opt2, HEX.opt3, HEX.pos, HEX.flat, HEX.good];
        ctx.starts.forEach(([r0, l0], k) => {
          let r = r0, l = Math.max(l0, 1e-3); const pts = [[r, l]];
          for (let t = 0; t < 400; t++) { [r, l] = step(r, l, eta, ne); if (!isFinite(r) || Math.abs(r) > 50 || l > 200) break; pts.push([r, l]); }
          P.line(pts, { color: cols[k % cols.length], width: 1.2, alpha: .8, dots: 1.8 }).scatter([[r0, l0]], { color: cols[k % cols.length], r: 4.5 });
        });
        P.draw();
        ctx.stat(`n_eff = ${ne.toFixed(ne < 10 ? 1 : 0)} · η_c = √n_eff / (x y) = ${etac.toFixed(3)} · η = ${eta.toFixed(3)} (${ctx.vals.eta < 1 ? 'below: settles on the zero-loss line' : 'above: EoS possible'})`);
      },
    },
    {
      id: 'chaos', name: 'Route to chaos', mode: '2d', title: 'Period doubling as η grows',
      caption: 'Late-time sharpness values of the UV model for each step size (Eqs. 1–2, n_eff = 1). Below η_c = 0.5 training converges; above it the sharpness oscillates around 2/η, splits 1 → 2 → 4 → … and turns chaotic, until training diverges.',
      note: 'Each column: 3000 steps from (f̃, λ) = (−1.9, 1), last 120 values plotted. Paper Fig. 3b.',
      hint: '', legend: [[HEX.ball, 'late-time λ'], [HEX.neg, '2/η']],
      plot: { xlim: [.3, .95], ylim: [0, 9], xlabel: 'step size η', ylabel: 'late-time sharpness λ', legend: false },
      build(ctx) {
        const pts = []; let diverge = null;
        for (let k = 0; k <= 520; k++) {
          const eta = .3 + .65 * k / 520; let r = -1.9, l = 1, ok = true;
          for (let t = 0; t < 3000; t++) { [r, l] = step(r, l, eta, 1); if (!isFinite(r) || Math.abs(r) > 1e6) { ok = false; break; } if (t >= 2880) pts.push([eta, l]); }
          if (!ok && diverge === null) diverge = eta;
        }
        ctx.plot.clear().fn(e => 2 / e, { color: HEX.neg, dash: [5, 4] }).vline(.5, { color: HEX.muted, dash: [3, 4], text: 'η_c = 0.5' })
          .scatter(pts, { color: 'rgba(255,122,89,.75)', r: .9 }).vline(diverge ?? .95, { color: HEX.flat, dash: [2, 3], text: 'diverges' }).draw();
        ctx.stat(`first divergence at η ≈ ${diverge?.toFixed(3) ?? '> 0.95'} (paper: ≈ 0.8 for its initialization)`);
      },
    },
  ],
};
