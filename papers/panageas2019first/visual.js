// Panageas, Piliouras, Wang 2019 — GD with vanishing steps αₖ = Ω(1/k) still avoids strict saddles for almost every start;
// summable steps can stop at a non-critical point (Remark 3.1, Fig. 1). Checked in scripts/check-math/panageas2019first.py.
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const F = def({ L: (x, y) => x * x / 2 - y * y / 2 + y ** 4 / 4, g: (x, y) => [x, -y + y ** 3], dom: [-2, 2, -1.6, 1.6], h: v => v, fit: 1.4 });
const C0 = .8, alpha = (k, s) => C0 / (k + 1) ** s;
const SCHED = [[.5, '0.8/√(k+1)', C.ball, HEX.ball], [1, '0.8/(k+1)', C.opt2, HEX.opt2], [2, '0.8/(k+1)²', C.opt3, HEX.opt3]];
const gdStep = s => (p, b) => { const a = alpha(b.steps, s); return [p[0] - a * p[0], p[1] - a * (-p[1] + p[1] ** 3)]; };

export default {
  tabs: [
    {
      id: 'schedules', name: 'Step schedules', title: 'Vanishing steps still escape the saddle, unless they vanish too fast',
      caption: 'f = x²/2 − y²/2 + y⁴/4: a strict saddle at the origin (Hessian diag(1, −1), invertible) and minima at (0, ±1). Three GD runs start at the same point (1.5, 0.05), close to the saddle’s stable manifold y = 0, with decaying steps. 1/√k and 1/k are not summable, so the unstable direction keeps growing and both reach the minimum (1/k slowly). 1/k² is summable: the total distance GD can travel is finite, and it stops at a point that is not even critical.',
      note: 'αₖ = 0.8/(k + 1)^s, s = ½, 1, 2. The paper’s own Fig. 1 does the same on x² − y² with 1/√k, 1/k and 1/k⁴.',
      legend: [[HEX.ball, 'αₖ = 0.8/√(k+1)'], [HEX.opt2, 'αₖ = 0.8/(k+1)'], [HEX.opt3, 'αₖ = 0.8/(k+1)² (summable)'], [HEX.good, 'minima (0, ±1)'], [HEX.neg, 'λ < 0 at the saddle']],
      action: 'Release balls', cam: [3.6, 3.4, 2.4], tgt: [0, .2, 0],
      build(ctx) { const s = ctx.stage; s.surface(F); s.marker(F, 0, 1, C.good); s.marker(F, 0, -1, C.good); s.eigenLines(F, 0, 0, { len: .6 }); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        ctx.bs = SCHED.map(([e, , col]) => s.ball(F, { start: [1.5, .05], step: gdStep(e), color: col, speed: 4, maxSteps: 4000 }));
      },
      frame(ctx) {
        if (!ctx.bs) return;
        ctx.stat(`step ${ctx.bs[0].steps} · ` + ctx.bs.map((b, i) => { const g = F.g(...b.p); return `${SCHED[i][1]}: (${b.p[0].toFixed(3)}, ${b.p[1].toFixed(3)}), ‖∇f‖ = ${Math.hypot(...g).toExponential(1)}`; }).join(' · '));
      },
    },
    {
      id: 'stable', name: 'Where starts end', mode: '2d', title: 'Ω(1/k) steps: the starts that reach the saddle form a set of measure zero',
      caption: 'Every start on a 121 × 97 grid runs 3000 GD steps with the chosen schedule; colour shows where it ends. With 1/√k or 1/k only the stable manifold y = 0 and the two rows y = ±1.5 (which the first step, α₀ = 0.8, maps exactly onto y = 0) end at the saddle: lines, measure zero (Theorem 5.1). With the summable 1/k² a whole open band of starts never reaches any critical point.',
      note: 'End classes: within 0.05 of a minimum (green), of the saddle (red), elsewhere (amber, not critical). Computed live; grid as in the check script.',
      hint: '', legend: [[HEX.good, 'ends at a minimum'], [HEX.neg, 'ends at the saddle'], [HEX.flat, 'stops at a non-critical point']],
      controls: [{ id: 's', type: 'seg', label: 'schedule αₖ', value: 1, options: SCHED.map(q => [q[0], q[1]]) }],
      plot: { xlabel: 'x₀', ylabel: 'y₀', xlim: [-2, 2], ylim: [-1.6, 1.6], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const s = +ctx.vals.s, nx = 121, ny = 97, z = [], cnt = [0, 0, 0];
        for (let j = 0; j < ny; j++) {
          const row = [];
          for (let i = 0; i < nx; i++) {
            let x = i * (4 / (nx - 1)) - 2, y = j * (3.2 / (ny - 1)) - 1.6;   // numpy linspace rounding
            for (let k = 0; k < 3000; k++) { const a = alpha(k, s); x -= a * x; y -= a * (-y + y * y * y); }
            const c = Math.min(Math.hypot(x, y - 1), Math.hypot(x, y + 1)) < .05 ? 0 : Math.hypot(x, y) < .05 ? 1 : 2; cnt[c]++; row.push(c);
          }
          z.push(row);
        }
        const colr = [HEX.good, HEX.neg, HEX.flat];
        ctx.plot.clear().heat(z, [-2, 2, -1.6, 1.6], { cell: v => colr[v], alpha: .85 }).draw();
        const N = nx * ny, D = ctx.data.frac[String(s)];
        ctx.stat(`αₖ = ${SCHED.find(q => q[0] === s)[1]} · at a minimum ${(100 * cnt[0] / N).toFixed(2)}% · at the saddle ${cnt[1]} starts (${(100 * cnt[1] / N).toFixed(2)}%) · non-critical ${(100 * cnt[2] / N).toFixed(2)}% · check script: ${(100 * D[0]).toFixed(2)}% / ${D[3]} / ${(100 * D[2]).toFixed(2)}%`);
      },
    },
  ],
};
