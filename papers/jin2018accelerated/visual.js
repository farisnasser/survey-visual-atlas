// Jin, Netrapalli, Jordan 2018 — perturbed accelerated gradient descent (Alg. 2) with negative curvature exploitation (Alg. 3).
// Checked in scripts/check-math/jin2018accelerated.py (same toy, parameters, escape counts and Hamiltonian monotonicity).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const ETA = .25;   // eta = 1/(4 ell), ell = 1
const mk = lam => {
  const f = (x, y) => .5 * x * x - .5 * lam * y * y + .25 * lam * y ** 4;
  const g = (x, y) => [x, -lam * y + lam * y ** 3];
  return { f, g, d: def({ L: f, g, hess: (x, y) => [1, 0, -lam + 3 * lam * y * y], h: v => v, dom: [-.3, .3, -1.4, 1.4], square: true, N: 90 }) };
};
const theta = lam => Math.sqrt(lam) / 4;   // theta = 1/(4 sqrt(kappa)), kappa = ell / sqrt(rho eps) with sqrt(rho eps) = lam
const gdStep = (p, b, d) => { const [gx, gy] = d.g(...p); return [p[0] - ETA * gx, p[1] - ETA * gy]; };
const agdStep = th => (p, b, d) => {
  b.v ??= [0, 0];
  const y = [p[0] + (1 - th) * b.v[0], p[1] + (1 - th) * b.v[1]], [gx, gy] = d.g(...y), q = [y[0] - ETA * gx, y[1] - ETA * gy];
  b.v = [q[0] - p[0], q[1] - p[1]]; return q;
};

/** Algorithm 2 (perturbation + AGD + NCE); (2) is tested only for v != 0, where it is not trivially an equality. */
function pagd(lam, T = 2600, seed = 0) {
  const { f, g } = mk(lam), R = rng(seed), th = theta(lam), gam = th * th / ETA, s = gam / (4 * 6 * lam), eps = 1e-3, r = 1e-3, Tp = 60;
  let x = [1, 0], v = [0, 0], last = -1e9; const out = [];
  for (let t = 0; t < T; t++) {
    let kick = false;
    if (Math.hypot(...g(...x)) <= eps && t - last > Tp) { const a = R() * 2 * Math.PI, rr = r * Math.sqrt(R()); x = [x[0] + rr * Math.cos(a), x[1] + rr * Math.sin(a)]; last = t; kick = true; }
    const y = [x[0] + (1 - th) * v[0], x[1] + (1 - th) * v[1]], gy = g(...y);
    let xn = [y[0] - ETA * gy[0], y[1] - ETA * gy[1]], vn = [xn[0] - x[0], xn[1] - x[1]];
    const dx = [x[0] - y[0], x[1] - y[1]];
    const nce = (v[0] || v[1]) && f(...x) <= f(...y) + gy[0] * dx[0] + gy[1] * dx[1] - gam / 2 * (dx[0] ** 2 + dx[1] ** 2);
    if (nce) {
      const nv = Math.hypot(...v);
      if (nv >= s) xn = x.slice(); else { const dl = [s * v[0] / nv, s * v[1] / nv], A = [x[0] + dl[0], x[1] + dl[1]], B = [x[0] - dl[0], x[1] - dl[1]]; xn = f(...A) < f(...B) ? A : B; }
      vn = [0, 0];
    }
    x = xn; v = vn;
    out.push([t, f(...x), f(...x) + (v[0] ** 2 + v[1] ** 2) / (2 * ETA), nce ? 1 : 0, kick ? 1 : 0]);
  }
  return out;
}

export default {
  tabs: [
    {
      id: 'escape', name: 'Faster escape', title: 'Momentum leaves a weak saddle in √ of the time',
      caption: 'Both balls start 10⁻³ off a saddle with weak negative curvature −λ. Gradient descent (white) grows the unstable coordinate by (1 + ηλ) per step; accelerated GD (coral) grows it by roughly 1 + c√(ηλ), so it escapes in about 1/√λ instead of 1/λ steps.',
      note: 'f = x²/2 − λy²/2 + λy⁴/4 on |x| ≤ 0.3. η = 1/(4ℓ) = 0.25, θ = 1/(4√κ) with κ = ℓ/λ (Eq. 3). Height is display-scaled.',
      legend: [[HEX.opt3, 'GD'], [HEX.ball, 'AGD (Alg. 1)'], [HEX.good, 'minima y = ±1'], [HEX.neg, 'λ < 0 direction']],
      action: 'Release balls', cam: [3.4, 2.6, 3.2], tgt: [0, .4, 0],
      controls: [{ id: 'lam', label: 'curvature λ', min: .01, max: .2, step: .005, value: .05, digits: 3 }],
      charts: [{ id: 'y', title: '|y_t| (distance from the stable set)', xlabel: 'step t', ylabel: '|y|', logy: true, ylim: [5e-4, 2], h: 220 }],
      build(ctx) {
        ctx.M = mk(ctx.vals.lam); const s = ctx.stage, d = ctx.M.d; s.surface(d);
        s.eigenLines(d, 0, 0, { len: .5 }); s.marker(d, 0, 1, C.good); s.marker(d, 0, -1, C.good);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, d = ctx.M.d; s.clearBalls();
        const stop = b => Math.abs(b.p[1]) > .98 && Math.hypot(...d.g(...b.p)) < 1e-4;
        ctx.a = s.ball(d, { start: [.3, 1e-3], step: gdStep, color: C.opt3, speed: 6, maxSteps: 6000, stop });
        ctx.b = s.ball(d, { start: [.3, 1e-3], step: agdStep(theta(ctx.vals.lam)), speed: 6, maxSteps: 6000, stop });
        for (const b of [ctx.a, ctx.b]) { b.hy = []; b.onStep = q => { q.hy.push([q.steps, Math.max(Math.abs(q.p[1]), 5e-4)]); if (q.esc == null && Math.abs(q.p[1]) > .5) q.esc = q.steps; }; }
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        const { a, b } = ctx; if (!a) return;
        const xm = Math.max(50, a.steps, b.steps);
        ctx.charts.y.clear().set({ xlim: [0, xm] }).hline(.5, { color: HEX.muted, dash: [4, 4], text: 'escaped' })
          .line(a.hy, { color: HEX.opt3, width: 1.8, label: 'GD' }).line(b.hy, { color: HEX.ball, width: 1.8, label: 'AGD' }).draw();
        const e = q => q.esc != null ? `${q.esc} steps` : '…';
        ctx.stat(`λ = ${ctx.vals.lam.toFixed(3)}, θ = ${theta(ctx.vals.lam).toFixed(3)} · escape (|y| > 0.5): GD ${e(a)}, AGD ${e(b)}${a.esc && b.esc ? ` · ratio ${(a.esc / b.esc).toFixed(1)}×` : ''}`);
      },
    },
    {
      id: 'scaling', name: 'Scaling', mode: '2d', title: '1/λ versus 1/√λ',
      caption: 'Escape steps for five curvatures, from the check script. Slopes on log–log axes: GD ≈ −1, AGD ≈ −½. This is the mechanism behind the paper’s Õ(ε^{−7/4}) rate versus Õ(ε^{−2}) for perturbed GD.',
      note: 'Same toy and start as Faster escape. Theorem 3 counts iterations to an ε-second-order stationary point; this chart isolates the escape phase only.',
      hint: '', legend: [[HEX.opt3, 'GD'], [HEX.ball, 'AGD']],
      plot: { xlabel: 'curvature λ', ylabel: 'steps to |y| > 0.5', logx: true, logy: true, xlim: [.008, .25], ylim: [20, 6000] },
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear();
        const fit = (ys, k) => x => ys[0] * (x / D.lams[0]) ** k;
        P.fn(fit(D.gd, -1), { color: HEX.opt3, dash: [4, 4], width: 1 }).fn(fit(D.agd, -.5), { color: HEX.ball, dash: [4, 4], width: 1 });
        P.line(D.lams.map((l, i) => [l, D.gd[i]]), { color: HEX.opt3, width: 2, dots: 4, label: `GD (slope ${D.slopes[0]})` });
        P.line(D.lams.map((l, i) => [l, D.agd[i]]), { color: HEX.ball, width: 2, dots: 4, label: `AGD (slope ${D.slopes[1]})` }).draw();
        ctx.stat(`λ = 0.01: GD ${D.gd[4]} steps, AGD ${D.agd[4]} steps · dashed: exact slopes −1 and −½`);
      },
    },
    {
      id: 'energy', name: 'Hamiltonian', mode: '2d', title: 'f can rise; the Hamiltonian cannot',
      caption: 'Full PAGD from the saddle’s stable set (y = 0): a tiny kick, accelerated steps, and Negative Curvature Exploitation when the function is “too non-convex” between x_t and y_t. f(x_t) goes up on many steps; E_t = f(x_t) + ‖v_t‖²/(2η) never does (Lemmas 4 and 5). Only the kick adds energy.',
      note: 'λ = 0.1, γ = θ²/η, s = γ/(4ρ), ρ = 6λ. We test condition (2) only when v_t ≠ 0: at v_t = 0 it holds with equality and Alg. 3 would divide by ‖v_t‖ = 0.',
      hint: '', legend: [[HEX.ball, 'f(x_t)'], [HEX.pos, 'Hamiltonian E_t'], [HEX.flat, 'NCE fired'], [HEX.neg, 'perturbation']],
      plot: { xlabel: 'iteration t', ylabel: 'value', xlim: [0, 2600] },
      build(ctx) {
        const H = pagd(.1), P = ctx.plot.clear();
        const lo = Math.min(...H.map(h => h[1])), hi = Math.max(...H.map(h => h[2]));
        P.set({ ylim: [lo - .002, Math.min(hi, .6) + .01] });
        H.filter(h => h[3]).forEach(h => P.vline(h[0], { color: 'rgba(255,200,87,.35)', width: 1 }));
        H.filter(h => h[4]).forEach(h => P.vline(h[0], { color: HEX.neg, width: 1.5 }));
        P.line(H.map(h => [h[0], h[2]]), { color: HEX.pos, width: 2, label: 'E_t' }).line(H.map(h => [h[0], h[1]]), { color: HEX.ball, width: 1.4, label: 'f(x_t)' }).draw();
        let upF = 0, upE = 0; for (let i = 1; i < H.length; i++) { if (H[i][1] > H[i - 1][1] + 1e-15) upF++; if (!H[i][4] && H[i][2] > H[i - 1][2] + 1e-13) upE++; }
        ctx.stat(`f increased on ${upF} of ${H.length - 1} steps · E increased on ${upE} steps (excluding kicks) · NCE fired ${H.filter(h => h[3]).length}× · final f = ${H[H.length - 1][1].toFixed(5)} (minimum −0.025)`);
      },
    },
  ],
};
