// Lee, Simchowitz, Jordan, Recht 2016 — the paper's Sec. 3 example f = x²/2 + y⁴/4 − y²/2.
// Checked in scripts/check-math/lee2016gradient.py
import { def, opt, C, HEX, segPts } from '../../src/lib/atlas3d.js';
import { rng, logspace } from '../../src/lib/math.js';

const F = def({
  L: (x, y) => .5 * x * x + .25 * y ** 4 - .5 * y * y,
  g: (x, y) => [x, y ** 3 - y], hess: (x, y) => [1, 0, 3 * y * y - 1],
  h: v => (v + .25) * .9,
});
const LR = .1; // < 1/L = 0.15 on |y| <= 1.6

export default {
  tabs: [
    {
      id: 'stable', name: 'Stable set', title: 'Only a measure-zero set of starts reaches the saddle',
      caption: 'f = x²/2 + y⁴/4 − y²/2. Gradient descent converges to the saddle only from the x-axis (white), a line with zero area. Random starts never land on it.',
      legend: [[HEX.ball, 'GD from random starts'], ['#ffffff', 'stable set of the saddle (x-axis)'], [HEX.good, 'minima (0, ±1)'], [HEX.neg, 'λ = −1'], [HEX.pos, 'λ = +1']],
      actions: [{ label: 'Drop 40 random balls', fn: ctx => ctx.shell.tab.drop(ctx) }, { label: 'Start on the stable set', fn: ctx => ctx.shell.tab.onAxis(ctx) }],
      cam: [3.4, 3.5, 4.4], tgt: [0, .35, 0],
      build(ctx) {
        const s = ctx.stage; s.surface(F);
        s.tube(F, segPts(-1.6, 0, 1.6, 0), C.white, .011);
        s.eigenLines(F, 0, 0, { len: .6 });
        s.marker(F, 0, 1, C.good); s.marker(F, 0, -1, C.good);
        s.label('stable set', F.W(1.25, 0, .3), '', HEX.white);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, r = rng(Date.now() % 1e6); s.clearBalls();
        for (let i = 0; i < 40; i++) s.ball(F, { start: [r.uniform(-1.5, 1.5), r.uniform(-1.5, 1.5)], step: opt.gd(LR), speed: 1, r: .05, maxTrail: 800, stop: b => Math.hypot(...F.g(...b.p)) < 1e-6 });
      },
      onAxis(ctx) { const s = ctx.stage; s.clearBalls(); s.ball(F, { start: [1.45, 0], step: opt.gd(LR), stop: b => Math.hypot(...b.p) < 1e-6, color: C.opt2 }); },
      frame(ctx) {
        const bs = ctx.stage.balls, near = q => Math.hypot(q[0], q[1] - 1) < 1e-3 || Math.hypot(q[0], q[1] + 1) < 1e-3;
        ctx.stat(`${bs.length} ball${bs.length > 1 ? 's' : ''}: ${bs.filter(b => near(b.p)).length} at a minimum · ${bs.filter(b => Math.hypot(...b.p) < 1e-3).length} at the saddle`);
      },
    },
    {
      id: 'escape', name: 'Escape time', mode: '2d', title: 'Avoiding the saddle can still take forever',
      caption: 'Iterations for GD to reach |y| > 0.5 from (1, y₀). The theorem says the saddle is avoided almost surely, but the time grows like log(1/y₀) without bound.',
      note: 'Near the saddle y grows by a factor (1 + α) per step, so t ≈ ln(0.5/y₀) / ln(1 + α).',
      hint: '', controls: [{ id: 'a', label: 'step α', min: .02, max: .14, step: .01, value: .1 }],
      plot: { logx: true, xlabel: 'initial distance y₀ from the stable set', ylabel: 'iterations to escape', xlim: [1e-14, .3] },
      build(ctx) { this.draw(ctx); },
      onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const a = ctx.vals.a, pts = [];
        for (const y0 of logspace(1e-14, .2, 28)) { let x = 1, y = y0, k = 0; while (Math.abs(y) <= .5 && k < 1e6) { [x, y] = [x - a * x, y - a * (y ** 3 - y)]; k++; } pts.push([y0, k]); }
        const P = ctx.plot.clear();
        P.fn(y0 => Math.log(.5 / y0) / Math.log(1 + a), { color: HEX.muted, dash: [5, 5], label: 'ln(0.5/y₀) / ln(1+α)' });
        P.line(pts, { color: HEX.ball, width: 2, dots: 3, label: 'gradient descent' });
        P.draw();
        const esc = y0 => { let x = 1, y = y0, k = 0; while (Math.abs(y) <= .5) { [x, y] = [x - a * x, y - a * (y ** 3 - y)]; k++; } return k; };
        ctx.stat(`α = ${a}: y₀ = 10⁻³ → ${esc(1e-3)} steps · y₀ = 10⁻¹² → ${esc(1e-12)} steps · y₀ = 0 → never`);
      },
    },
    {
      id: 'map', name: 'GD as a map', mode: '2d', title: 'For small steps, GD never folds space',
      caption: 'One GD step g(x) = x − α∇f(x) applied to a grid. For α < 1/L the map is a diffeomorphism (Prop. 8), so the thin set that flows into the saddle stays thin when pulled back.',
      note: 'det Dg = (1 − α)((1 + α) − 3αy²). Red cells: det < 0, where the map folds (only possible when α exceeds 1/L on the region).',
      legend: [['#5d6890', 'original grid'], [HEX.pos, 'grid after one step'], [HEX.neg, 'folded (det Dg < 0)']], hint: '',
      controls: [{ id: 'a', label: 'step α', min: .02, max: .7, step: .01, value: .1 }],
      plot: { xlim: [-1.9, 1.9], ylim: [-2.6, 2.6], xlabel: 'x', ylabel: 'y', legend: false },
      build(ctx) { this.draw(ctx); },
      onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const a = ctx.vals.a, P = ctx.plot.clear(), gmap = (x, y) => [x - a * x, y - a * (y ** 3 - y)], n = 24, D = 1.6;
        P.custom((g, p) => {
          for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
            const x0 = -D + 2 * D * i / n, y0 = -D + 2 * D * j / n, x1 = x0 + 2 * D / n, y1 = y0 + 2 * D / n, ym = (y0 + y1) / 2;
            const det = (1 - a) * ((1 + a) - 3 * a * ym * ym);
            const q = [gmap(x0, y0), gmap(x1, y0), gmap(x1, y1), gmap(x0, y1)];
            g.beginPath(); q.forEach(([x, y], k) => k ? g.lineTo(p.sx(x), p.sy(y)) : g.moveTo(p.sx(x), p.sy(y))); g.closePath();
            g.fillStyle = det < 0 ? 'rgba(255,93,115,.55)' : 'rgba(124,196,255,.10)'; g.fill();
            g.strokeStyle = det < 0 ? 'rgba(255,93,115,.9)' : 'rgba(124,196,255,.75)'; g.lineWidth = 1; g.stroke();
          }
          g.strokeStyle = 'rgba(93,104,144,.45)';
          for (let i = 0; i <= n; i++) { const v = -D + 2 * D * i / n; g.beginPath(); g.moveTo(p.sx(v), p.sy(-D)); g.lineTo(p.sx(v), p.sy(D)); g.moveTo(p.sx(-D), p.sy(v)); g.lineTo(p.sx(D), p.sy(v)); g.stroke(); }
        });
        P.draw();
        const Lr = Math.max(1, 3 * D * D - 1), fold = Math.sqrt((1 + a) / (3 * a));
        ctx.stat(`1/L on this grid = ${(1 / Lr).toFixed(3)} · α = ${a.toFixed(2)} ${a < 1 / Lr ? '< 1/L: invertible' : '≥ 1/L'} · fold at |y| = ${fold.toFixed(2)}${fold > D ? ' (outside the grid)' : ''}`);
      },
    },
  ],
};
