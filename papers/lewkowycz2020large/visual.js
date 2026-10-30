// Lewkowycz, Bahri, Dyer, Sohl-Dickstein, Gur-Ari 2020 — the catapult phase: for 2/λ0 < η < 4/λ0 the loss spikes,
// the curvature drops below 2/η and training converges in a flatter minimum. Eqs. (6)-(7) checked in
// scripts/check-math/lewkowycz2020large.py (they reproduce GD on u, v to 1e-15).
import { def, C, HEX, segPts } from '../../src/lib/atlas3d.js';

const U0 = [2, .35], LAM0 = U0[0] ** 2 + U0[1] ** 2;   // n = 1: f = uv, lambda = u^2 + v^2
const uv = def({ L: (u, v) => .5 * (u * v) ** 2, g: (u, v) => [u * v * v, v * u * u], dom: [-2.6, 2.6, -2.6, 2.6], h: L => Math.log1p(L) });
/** Eqs. (6)-(7): exact function-space dynamics of the width-n warmup model (x = 1, y = 0). */
const fsStep = (f, l, eta, n) => [(1 - eta * l + eta * eta * f * f / n) * f, l + eta * f * f / n * (eta * l - 4)];
function runFs(c, n, T = 400) {
  let f = 1, l = 1; const out = [[0, .5, 1]];   // f0 = 1, lambda0 = 1, eta = c / lambda0
  for (let t = 1; t <= T; t++) { [f, l] = fsStep(f, l, c, n); if (!isFinite(f) || Math.abs(f) > 1e8) return { out, div: true }; out.push([t, f * f / 2, l]); }
  return { out, div: false };
}
const phase = c => c < 2 ? 'lazy' : c < 4 ? 'catapult' : 'divergent';

export default {
  tabs: [
    {
      id: 'spike', name: 'Catapult', title: 'Too large for the start, fine for where it lands',
      caption: 'Loss ½(uv)² of the smallest two-layer linear model. Both axes are minima (green); the curvature there is λ = u² + v². The ball starts near a sharp point with curvature λ₀. For η between 2/λ₀ and 4/λ₀ the loss first grows, the ball is thrown towards the origin, the curvature falls below 2/η and GD converges on a flatter part of the axes.',
      note: 'Width n = 1 (u, v scalars), one example x = 1 with label 0 (the paper’s Sec. 2.1 model). At n = 1 the lazy phase is not sharp: even below 2/λ₀ the curvature moves.',
      legend: [[HEX.ball, 'gradient descent'], [HEX.good, 'minima (uv = 0)']],
      action: 'Restart', cam: [3.6, 3.8, 4.2], tgt: [0, .2, 0],
      controls: [{ id: 'c', label: 'η · λ₀', min: 1, max: 4.4, step: .05, value: 3 }],
      charts: [
        { id: 'loss', title: 'loss (log)', xlabel: 'step', ylabel: 'L', logy: true, ylim: [1e-10, 1e3], h: 170, legend: false },
        { id: 'lam', title: 'curvature λ = u² + v²', xlabel: 'step', ylabel: 'λ', ylim: [0, 6], h: 190, legend: false },
      ],
      build(ctx) {
        const s = ctx.stage; s.surface(uv);
        s.tube(uv, segPts(-2.6, 0, 2.6, 0), C.good); s.tube(uv, segPts(0, -2.6, 0, 2.6), C.good);
        s.marker(uv, U0[0], U0[1], C.white, .08); s.label('start: λ₀ = 4.12', uv.W(U0[0], U0[1], .35), '', HEX.white);
        this.drop(ctx);
      },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, eta = ctx.vals.c / LAM0; s.clearBalls(); ctx.hist = [[0, uv.L(...U0), LAM0]];
        ctx.b = s.ball(uv, { start: U0, speed: .2, maxSteps: 400, step: p => { const [gu, gv] = uv.g(p[0], p[1]), q = [p[0] - eta * gu, p[1] - eta * gv]; ctx.hist.push([ctx.hist.length, uv.L(q[0], q[1]), q[0] ** 2 + q[1] ** 2]); return q; }, stop: b => Math.abs(b.p[0]) >= 2.6 || Math.abs(b.p[1]) >= 2.6 });
      },
      frame(ctx) {
        const eta = ctx.vals.c / LAM0, H = ctx.hist, xl = [0, Math.max(30, H.length)];
        ctx.charts.loss.clear().set({ xlim: xl }).line(H.map(h => [h[0], Math.max(h[1], 1e-10)]), { color: HEX.ball, width: 1.6 }).draw();
        ctx.charts.lam.clear().set({ xlim: xl }).hline(2 / eta, { color: HEX.neg, dash: [5, 4], text: `2/η = ${(2 / eta).toFixed(2)}` })
          .line(H.map(h => [h[0], h[2]]), { color: HEX.pos, width: 1.6 }).draw();
        const last = H[H.length - 1], peak = Math.max(...H.map(h => h[1])), out = Math.abs(ctx.b.p[0]) >= 2.6 || Math.abs(ctx.b.p[1]) >= 2.6;
        ctx.stat(`η·λ₀ = ${ctx.vals.c.toFixed(2)} (${phase(ctx.vals.c)}) · peak loss ${peak.toFixed(3)} vs start ${H[0][1].toFixed(3)} · ${out ? 'diverged (left the box)' : `λ now ${last[2].toFixed(2)} vs 2/η = ${(2 / eta).toFixed(2)}`}`);
      },
    },
    {
      id: 'width', name: 'Width n', mode: '2d', title: 'Wide networks catapult after ~log n quiet steps',
      caption: 'The exact function-space map of the width-n model, Eqs. (6)–(7). While f²/n is tiny the kernel λ barely moves and the loss grows geometrically, as a linear (NTK) model would; once f² reaches order n the kernel drops below 2/η and the loss collapses. Wider networks wait longer, about log n steps.',
      note: 'f₀ = 1, λ₀ = 1, so η = η·λ₀. Loss on the main axes, λ on the side with the stability line 2/η.',
      hint: '', legend: [[HEX.ball, 'loss f²/2'], [HEX.pos, 'kernel λ'], [HEX.neg, '2/η']],
      controls: [{ id: 'c', label: 'η · λ₀', min: 1, max: 4.3, step: .05, value: 3 }, { id: 'ln', label: 'width n', min: 0, max: 8, step: 1, value: 4, fmt: v => `10^${v}` }],
      plot: { xlabel: 'step', ylabel: 'loss', logy: true, ylim: [1e-12, 1e8], xlim: [0, 60], legend: false },
      charts: [{ id: 'lam', title: 'kernel λ_t', xlabel: 'step', ylabel: 'λ', ylim: [0, 1.4], xlim: [0, 60], h: 240, legend: false }],
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const c = ctx.vals.c, n = 10 ** ctx.vals.ln, { out, div } = runFs(c, n, 60);
        ctx.plot.clear().line(out.map(r => [r[0], Math.max(r[1], 1e-12)]), { color: HEX.ball, width: 2, dots: 1.6 }).draw();
        ctx.charts.lam.clear().hline(2 / c, { color: HEX.neg, dash: [5, 4], text: '2/η' }).line(out.map(r => [r[0], r[2]]), { color: HEX.pos, width: 2 }).draw();
        const pk = out.reduce((a, r) => r[1] > a[1] ? r : a, out[0]);
        ctx.stat(`n = 10^${ctx.vals.ln} · η·λ₀ = ${c.toFixed(2)} → ${div ? 'diverges (λ grows when ηλ > 4)' : `${phase(c)}: loss peaks at step ${pk[0]} (${pk[1].toExponential(1)}), final λ = ${out[out.length - 1][2].toFixed(3)}${c > 2 ? ` < 2/η = ${(2 / c).toFixed(3)}` : ''}`}`);
      },
    },
    {
      id: 'phases', name: 'Three phases', mode: '2d', title: 'Lazy, catapult, divergent',
      caption: 'Final curvature λ_∞ (relative to λ₀) of the width-n model after training, for every learning rate. Below 2/λ₀ it stays at λ₀ (lazy, NTK-like); between 2/λ₀ and 4/λ₀ it ends below 2/η, a flatter minimum; above 4/λ₀ training diverges. The paper finds the same 2/λ₀ transition in real networks, with the upper edge near 12/λ₀ for ReLU nets (reported).',
      note: 'Eqs. (6)–(7), f₀ = λ₀ = 1, 4000 steps per point. Dashed: 2/(ηλ₀), the largest stable curvature.',
      hint: '', legend: [[HEX.ball, 'λ_∞ / λ₀'], [HEX.neg, '2/(ηλ₀)']],
      controls: [{ id: 'ln', label: 'width n', min: 1, max: 8, step: 1, value: 3, fmt: v => `10^${v}` }],
      plot: { xlabel: 'η · λ₀', ylabel: 'λ_∞ / λ₀', xlim: [.5, 4.6], ylim: [0, 1.15], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const n = 10 ** ctx.vals.ln, pts = []; let divAt = null;
        for (let c = .5; c <= 4.6; c += .02) { const { out, div } = runFs(c, n, 4000); if (div) { divAt ??= c; continue; } pts.push([c, out[out.length - 1][2]]); }
        const P = ctx.plot.clear().band(.5, 2, { color: 'rgba(124,196,255,.06)' }).band(2, 4, { color: 'rgba(255,200,87,.07)' }).band(4, 4.6, { color: 'rgba(255,93,115,.07)' });
        P.fn(c => 2 / c, { color: HEX.neg, dash: [5, 4] }).scatter(pts, { color: HEX.ball, r: 2.2 })
          .text(1.25, 1.08, 'lazy', { color: HEX.pos, align: 'center' }).text(3, 1.08, 'catapult', { color: HEX.flat, align: 'center' }).text(4.3, 1.08, 'diverge', { color: HEX.neg, align: 'center' }).draw();
        ctx.stat(`n = 10^${ctx.vals.ln}: first divergence at η·λ₀ ≈ ${divAt?.toFixed(2) ?? '> 4.6'} (theory: 4) · at η·λ₀ = 3 the final curvature is ${(pts.find(p => p[0] > 2.99)?.[1] ?? NaN).toFixed(3)}·λ₀`);
      },
    },
  ],
};
