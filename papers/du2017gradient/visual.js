// Du, Jin, Lee, Jordan, Póczos, Singh 2017 — GD can need exponential time to pass a chain of saddles; perturbed GD does not.
// Our smooth two- and six-saddle chains (same mechanism as the paper's construction): scripts/check-math/du2017gradient.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const G = 1, CC = .5;
const S = u => u <= 0 ? 0 : u >= 1 ? 1 : 6 * u ** 5 - 15 * u ** 4 + 10 * u ** 3, dS = u => u <= 0 || u >= 1 ? 0 : 30 * u ** 4 - 60 * u ** 3 + 30 * u * u;
/** Two-saddle chain f = −x²/2 + q(x)y²/2 + x⁴/4 + y⁴/4, q(x) = L − (L + γ)S(x²/c²): y has curvature +L until x escapes, then −γ. */
const chain = L => {
  const q = t => L - (L + G) * S(t * t / (CC * CC)), dq = t => -(L + G) * dS(t * t / (CC * CC)) * 2 * t / (CC * CC);
  return def({
    L: (x, y) => -G * x * x / 2 + q(x) * y * y / 2 + x ** 4 / 4 + y ** 4 / 4,
    g: (x, y) => [-G * x + x ** 3 + dq(x) * y * y / 2, q(x) * y + y ** 3],
    dom: [-1.4, 1.4, -1.4, 1.4],
  });
};
const eta = L => 1 / (4 * (L + 3));
const gdStep = L => (p, b, d) => { const g = d.g(p[0], p[1]), e = eta(L); return [p[0] - e * g[0], p[1] - e * g[1]]; };
/** perturbed GD: when ‖∇f‖ < r, add a uniform kick from the ball of radius r, then take the GD step */
const pgdStep = (L, R, r = .01) => (p, b, d) => {
  let q = p; const g0 = d.g(p[0], p[1]);
  if (Math.hypot(...g0) < r) { const a = 2 * Math.PI * R(), s = r * Math.sqrt(R()); q = [p[0] + s * Math.cos(a), p[1] + s * Math.sin(a)]; b.kicks = (b.kicks || 0) + 1; }
  const g = d.g(q[0], q[1]), e = eta(L); return [q[0] - e * g[0], q[1] - e * g[1]];
};

export default {
  tabs: [
    {
      id: 'chain', name: 'Saddle chain', title: 'Each saddle hands GD a worse start for the next one',
      caption: 'Two saddles in a row: the origin (GD must first grow x) and (±1, 0) (then it must grow y). While x escapes the first saddle, the y direction still curves upward, so y shrinks; by the time GD reaches the second saddle, y is tiny and escaping takes longer. Raise L, the upward curvature, and the second wait grows. Perturbed GD (violet) adds a small random kick whenever the gradient is small, so it never starts the second escape from an absurdly small y.',
      note: 'f = −x²/2 + q(x)y²/2 + x⁴/4 + y⁴/4, q(x) = L − (L + 1)·S(x²/c²) with S a smoothstep, c = 0.5. Step 1/(4(L + 3)), start (0.001, 0.05). PGD kick radius 0.01. Our chain, not the paper’s spline construction.',
      legend: [[HEX.ball, 'GD'], [HEX.opt2, 'perturbed GD'], [HEX.neg, 'λ < 0 (escape direction)'], [HEX.good, 'minima (±1, ±1)']],
      action: 'Release balls', cam: [-.6, 4.4, 4.4], tgt: [0, .3, 0],
      controls: [{ id: 'L', type: 'seg', label: 'upward curvature L', value: 2, options: [[1, '1'], [2, '2'], [4, '4']] }],
      build(ctx) {
        const L = +ctx.vals.L, d = ctx.d = chain(L), s = ctx.stage; s.surface(d);
        for (const a of [1, -1]) for (const b of [1, -1]) s.marker(d, a, b, C.good);
        s.eigenLines(d, 0, 0, { len: .4 }); s.eigenLines(d, 1, 0, { len: .4 }); this.drop(ctx);
      },
      drop(ctx) {
        const L = +ctx.vals.L, s = ctx.stage; s.clearBalls(); ctx.esc = [[null, null], [null, null]];
        const mk = (step, color, i) => { const b = s.ball(ctx.d, { start: [.001, .05], step, color, speed: 4, maxSteps: 8000 }); b.onStep = q => { const e = ctx.esc[i]; if (e[0] == null && Math.abs(q.p[0]) > CC) e[0] = q.steps; if (e[0] != null && e[1] == null && Math.abs(q.p[1]) > CC) e[1] = q.steps; }; return b; };
        ctx.bs = [mk(gdStep(L), C.ball, 0), mk(pgdStep(L, rng(3)), C.opt2, 1)];
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.bs) return;
        const f = (e, b) => `first saddle left at step ${e[0] ?? '…'}, second at ${e[1] ?? '…'}, |y| min so far ${(b.ymin = Math.min(b.ymin ?? 1, Math.abs(b.p[1]))).toExponential(1)}`;
        ctx.stat(`L = ${ctx.vals.L} · step ${ctx.bs[0].steps} · GD: ${f(ctx.esc[0], ctx.bs[0])} · PGD: ${f(ctx.esc[1], ctx.bs[1])}`);
      },
    },
    {
      id: 'exp', name: 'Six saddles', mode: '2d', title: 'Escape times grow geometrically for GD and linearly for perturbed GD',
      caption: 'The same construction in six dimensions: GD must leave six saddles one after another. Step at which each saddle is left, for five random starts, on a log scale. Each GD wait is about (L + γ)/γ times the previous one, the factor the paper derives, so the total is exponential in the number of saddles; perturbed GD spends roughly the same time at each. With L = 2 the sixth coordinate of two GD runs shrank below 10⁻³⁰⁰, so in floating point GD never leaves the last saddle (marked ×).',
      note: 'Offline. x₀ ~ U[−0.1, 0.1]⁶, step 1/(4(L + 3)), 400 000 steps max, PGD kick radius 0.01. The paper’s Theorem 4.1 uses a different (“octopus”) function and U[−1, 1]^d starts.',
      hint: '', legend: [[HEX.ball, 'GD'], [HEX.opt2, 'perturbed GD']],
      controls: [{ id: 'L', type: 'seg', label: 'upward curvature L', value: 2, options: [[1, '1'], [2, '2']] }],
      plot: { xlabel: 'saddle number', ylabel: 'step when left', logy: true, xlim: [.5, 6.5], ylim: [20, 1e6], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const D = ctx.data.esc[(+ctx.vals.L).toFixed(1)], P = ctx.plot.clear(), L = +ctx.vals.L;
        D.pgd.forEach(r => P.line(r.map((v, i) => [i + 1, v]), { color: HEX.opt2, width: 1.6, dots: 3 }));
        const stuck = [];
        D.gd.forEach(r => { P.line(r.map((v, i) => [i + 1, v ?? NaN]), { color: HEX.ball, width: 1.6, dots: 3 }); r.forEach((v, i) => { if (v == null) stuck.push([i + 1, 4e5]); }); });
        if (stuck.length) P.scatter(stuck, { color: HEX.neg, r: 5 }).text(6.3, 4e5, '× never left (400 000 steps)', { color: HEX.neg, align: 'right', dy: -12 });
        P.draw();
        const last = a => a.map(r => r[5] ?? '—').join(', ');
        ctx.stat(`L = ${L}, factor (L + γ)/γ = ${L + 1} · GD left saddle 6 at steps ${last(D.gd)} · PGD at ${last(D.pgd)}`);
      },
    },
  ],
};
