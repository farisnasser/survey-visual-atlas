// Goodfellow, Vinyals, Saxe 2015 — the loss along the straight line from initialization to the SGD solution usually falls
// smoothly; between two solutions it can rise. Factored model of Fig. 14 + an offline MLP run: scripts/check-math/goodfellow2015qualitatively.py
import { def, C, HEX, segPts, hyperbola } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const F = def({ L: (a, b) => (1 - a * b) ** 2, g: (a, b) => [-2 * b * (1 - a * b), -2 * a * (1 - a * b)], dom: [-1.6, 1.6, -1.6, 1.6] });
const interp = (p, q, a) => [(1 - a) * p[0] + a * q[0], (1 - a) * p[1] + a * q[1]];

export default {
  tabs: [
    {
      id: 'factored', name: 'Factored model', title: 'A saddle at the start, a straight downhill line to the solution',
      caption: 'The paper’s Fig. 14 model: one-unit, two-layer linear net with loss (1 − w₁w₂)². SGD starts near the saddle at the origin and ends on the hyperbola of global solutions. Afterwards the white line joins start and end: the loss along it only goes down, even though the path SGD took bends. The violet line joins two equally good solutions, (1, 1) and (−1, −1): it climbs over the saddle.',
      note: 'SGD = GD with Gaussian gradient noise (σ = 0.05), step 0.02, start (0.12, 0.05). Side chart: loss along each straight line, α from 0 to 1.',
      legend: [[HEX.ball, 'SGD'], ['#ffffff', 'start → solution line'], [HEX.opt2, 'solution → solution line'], [HEX.good, 'global minima w₁w₂ = 1']],
      action: 'Run SGD again', cam: [0, 4.4, 4.4], tgt: [0, .3, 0],
      charts: [{ id: 'l', title: 'loss along the straight line', xlabel: 'α', ylabel: 'loss', xlim: [0, 1], ylim: [0, 1.05], h: 220, legend: false }],
      build(ctx) {
        const s = ctx.stage; s.surface(F); s.tube(F, hyperbola(1, 1.6), C.good, .014); s.tube(F, hyperbola(1, 1.6, -1), C.good, .014); s.eigenLines(F, 0, 0, { len: .45 });
        s.tube(F, segPts(1, 1, -1, -1), C.opt2, .012);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(4); s.clearBalls(); if (ctx.ln) { s.remove(ctx.ln); ctx.ln = null; }
        ctx.b = s.ball(F, { start: [.12, .05], speed: 4, maxSteps: 3000, step: (p, b, d) => { const g = d.g(...p); return [p[0] - .02 * (g[0] + .05 * R.normal()), p[1] - .02 * (g[1] + .05 * R.normal())]; } });
        ctx.drawn = false; this.chart(ctx, null);
      },
      chart(ctx, end) {
        const A = Array.from({ length: 101 }, (_, i) => i / 100), P = ctx.charts.l.clear();
        P.line(A.map(a => [a, F.L(...interp([1, 1], [-1, -1], a))]), { color: HEX.opt2, width: 2 });
        if (end) { const v = A.map(a => F.L(...interp([.12, .05], end, a))); ctx.mono = v.every((x, i) => !i || x <= v[i - 1] + 1e-12); P.line(A.map((a, i) => [a, v[i]]), { color: '#ffffff', width: 2 }); }
        P.draw();
      },
      frame(ctx) {
        const b = ctx.b; if (!b) return;
        if (b.done && !ctx.drawn) { ctx.drawn = true; ctx.ln = ctx.stage.tube(F, segPts(.12, .05, b.p[0], b.p[1]), C.white, .012); this.chart(ctx, b.p); }
        ctx.stat(`SGD step ${b.steps} · loss ${b.loss().toExponential(2)} · w₁w₂ = ${(b.p[0] * b.p[1]).toFixed(3)}${ctx.drawn ? ` · along the white line the loss ${ctx.mono ? 'falls monotonically' : 'is not monotone'} from 0.988 · along the violet line it rises to 1 at the saddle` : ''}`);
      },
    },
    {
      id: 'mlp', name: 'MLP interpolation', mode: '2d', title: 'A small tanh net: usually a smooth slide, sometimes a bump',
      caption: 'Training loss J((1 − α)θ₀ + αθ_f) for two 2-16-16-1 tanh networks trained by SGD on a two-spirals task (both reach 100% training accuracy). Run 2 slides down without obstacles, as the paper reports for its seven models; run 1 has a bump at α ≈ 0.25. Between the two solutions (dashed) the loss rises far above both ends. Right: SGD’s path does not follow the straight line.',
      note: '200 points, logistic loss, batch 20, step 0.3, 20 000 steps (offline). Across 24 more runs (12 seeds × steps 0.05 and 0.3) the line was monotone, within 1% of the drop, in 19.',
      hint: '', legend: [[HEX.ball, 'run 1: θ₀ → θ_f'], [HEX.opt2, 'run 2: θ₀ → θ_f'], [HEX.muted, 'θ_f(run 1) → θ_f(run 2)']],
      plot: { xlabel: 'α', ylabel: 'training loss J', logy: true, xlim: [-.25, 1.25], ylim: [1e-4, 20], legend: false },
      charts: [{ id: 'off', title: 'run 1: SGD path vs the straight line', xlabel: 'projection on θ_f − θ₀', ylabel: 'distance off the line', h: 220, legend: false }],
      build(ctx) {
        const D = ctx.data, al = D.alphas, P = ctx.plot.clear();
        P.vline(0, { color: HEX.muted, dash: [2, 4], text: 'α = 0' }).vline(1, { color: HEX.muted, dash: [2, 4], text: 'α = 1' });
        P.line(al.map((a, i) => [a, D.between[i]]), { color: HEX.muted, dash: [5, 4], width: 1.8 });
        P.line(al.map((a, i) => [a, D.line[0][i]]), { color: HEX.ball, width: 2.2 }).line(al.map((a, i) => [a, D.line[1][i]]), { color: HEX.opt2, width: 2.2 }).draw();
        ctx.charts.off.clear().line(D.proj.map((p, i) => [p, D.resid[i]]), { color: HEX.ball, width: 2, dots: 1.5 }).draw();
        const sw = Object.values(D.sweep).flat(), mono = sw.filter(q => q < .01).length;
        ctx.stat(`largest rise above the running minimum on [0, 1]: run 1 ${D.bumps[0][0].toFixed(3)} at α = ${D.bumps[0][1].toFixed(2)}, run 2 ${D.bumps[1][0].toFixed(4)} · barrier between the two solutions ${D.barrier.toFixed(2)} · sweep: monotone in ${mono}/${sw.length} runs · ‖θ_f − θ₀‖ = ${D.dist}, max distance off the line ${Math.max(...D.resid).toFixed(2)}`);
      },
    },
  ],
};
