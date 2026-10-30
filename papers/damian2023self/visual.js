// Damian, Nichani, Lee 2023 — self-stabilization: the cubic term of the loss pushes GD back below 2/η.
// The paper's App. B toy (x, y part): GD on it is exactly the simplified dynamics Eq. (3).
// Checked in scripts/check-math/damian2023self.py (Lemma 3 conservation, mean sharpness = 2/η, outward drift of the discrete map).
import { def, C, HEX, segPts } from '../../src/lib/atlas3d.js';

const ETA = 1, AL = .002, BE = 1, DELTA = Math.sqrt(2 * AL / BE), T = 450;
// S(x, y) = 2/eta + y along u = e_x;  alpha = -grad L . grad S;  beta = |grad S|^2 = 1
const toy = def({ L: (x, y) => (2 / ETA + y) * x * x / 2 - AL * y, g: (x, y) => [(2 / ETA + y) * x, x * x / 2 - AL], dom: [-.15, .15, -.09, .09], square: true, h: v => v, N: 90 });
const gfun = (X, Y) => { const z = BE * X * X / (2 * AL); return z - Math.log(z) - 1 + Y * Y / AL; };   // Lemma 3
const stage = (x, y) => Math.abs(x) < DELTA ? (y < 0 ? '1 · progressive sharpening' : '2 · blow-up') : (y > 0 ? '3 · self-stabilization' : '4 · return to stability');
/** ODE (4) orbit through (X0, Y0), RK4, one revolution. */
function orbit(X0, Y0) {
  let X = X0, Y = Y0; const out = [[X, Y]], dt = .05, f = (X, Y) => [X * Y, AL - BE * X * X / 2];
  for (let i = 0; i < 20000; i++) {
    const k1 = f(X, Y), k2 = f(X + dt / 2 * k1[0], Y + dt / 2 * k1[1]), k3 = f(X + dt / 2 * k2[0], Y + dt / 2 * k2[1]), k4 = f(X + dt * k3[0], Y + dt * k3[1]);
    const nY = Y + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); X += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    if (i > 50 && Y < Y0 && nY >= Y0 && X < DELTA) { out.push([X, nY]); break; }
    Y = nY; out.push([X, Y]);
  }
  return out;
}

export default {
  tabs: [
    {
      id: 'stages', name: 'Four stages', title: 'Oscillation along the sharp direction pushes the sharpness back down',
      caption: 'The paper’s toy loss: across the trough (x) the curvature is S = 2/η + y, and the loss slopes gently towards larger y, so GD keeps sharpening. Once S > 2/η (beyond the amber line) the zig-zag across the trough grows; its x² term then pulls y back, S drops below 2/η and the zig-zag dies out. Then the cycle repeats.',
      note: 'L(x, y) = (2/η + y)x²/2 − αy with η = 1, α = 0.002, β = 1 (App. B without its extra z coordinate). GD here is exactly Eq. (3). Run for 450 steps (~4 cycles): the discrete map slowly spirals outward, so longer runs leave the edge.',
      legend: [[HEX.ball, 'gradient descent'], [HEX.flat, 'S = 2/η'], [HEX.white, 'period-2 orbit x = ±δ']],
      action: 'Restart', cam: [0, 3.4, 3.4], tgt: [0, 0, 0],
      controls: [{ id: 'x0', label: 'initial |x| / δ', min: .2, max: .95, step: .05, value: .6 }],
      charts: [
        { id: 'x', title: '|x_t| (oscillation along u)', xlabel: 'step', ylabel: '|x|', ylim: [0, .16], xlim: [0, T], h: 170, legend: false },
        { id: 's', title: 'S_t − 2/η', xlabel: 'step', ylabel: 'S − 2/η', ylim: [-.16, .16], xlim: [0, T], h: 170, legend: false },
      ],
      build(ctx) {
        const s = ctx.stage; s.surface(toy);
        s.tube(toy, segPts(-.15, 0, .15, 0), C.flat, .012);
        s.marker(toy, DELTA, 0, C.white, .06); s.marker(toy, -DELTA, 0, C.white, .06);
        s.label('S > 2/η', toy.W(0, .07, .25), '', HEX.neg); s.label('S < 2/η', toy.W(0, -.07, .25), '', HEX.pos);
        this.drop(ctx);
      },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls(); ctx.hist = [[0, ctx.vals.x0 * DELTA, 0]];
        ctx.b = s.ball(toy, { start: [ctx.vals.x0 * DELTA, 0], speed: .9, maxSteps: T, step: p => { const [gx, gy] = toy.g(p[0], p[1]), q = [p[0] - ETA * gx, p[1] - ETA * gy]; ctx.hist.push([ctx.hist.length, Math.abs(q[0]), q[1]]); return q; } });
      },
      frame(ctx) {
        const H = ctx.hist;
        ctx.charts.x.clear().hline(DELTA, { color: HEX.white, dash: [3, 4], text: 'δ' }).line(H.map(h => [h[0], h[1]]), { color: HEX.ball, width: 1.4 }).draw();
        ctx.charts.s.clear().hline(0, { color: HEX.flat, dash: [5, 4], text: '2/η' }).line(H.map(h => [h[0], h[2]]), { color: HEX.pos, width: 1.4 }).draw();
        const [, ax, y] = H[H.length - 1], mean = H.slice(1).reduce((a, h) => a + h[2], 0) / Math.max(1, H.length - 1);
        ctx.stat(`step ${H.length - 1} · stage ${stage(ax, y)} · S − 2/η = ${y.toFixed(4)} · mean so far ${mean.toFixed(4)} · g = ${gfun(Math.max(ax, 1e-9), y).toFixed(3)}`);
      },
    },
    {
      id: 'orbit', name: 'Predator–prey', mode: '2d', title: 'Oscillation and sharpness chase each other around a cycle',
      caption: 'Eq. (4) in the plane of oscillation size |x| and sharpness excess S − 2/η: a Lotka–Volterra system whose orbits are closed curves of the conserved quantity g (Lemma 3), around the fixed point (δ, 0) with δ = √(2α/β). GD’s iterates (dots) follow these orbits for a few cycles, slowly spiralling outward. A smaller starting oscillation means a bigger loop (Corollary 1). Click to start elsewhere.',
      note: 'α = 0.002, β = 1, η = 1. Grey: ODE orbits through |x| = 0.3δ, 0.6δ, 0.9δ. Dots: 450 steps of GD on the toy (one dot per step, |x|).',
      hint: 'click the plane to start GD there', legend: [[HEX.ball, 'GD iterates'], [HEX.muted, 'ODE orbits (g = const)'], [HEX.white, 'fixed point (δ, 0)']],
      plot: { xlabel: 'oscillation |x|', ylabel: 'S − 2/η', xlim: [0, .16], ylim: [-.12, .12], legend: false },
      build(ctx) {
        ctx.start = [.6 * DELTA, 0];
        ctx.listen(document.querySelector('#plot2d'), 'pointerdown', ev => { const [X, Y] = ctx.plot.toData(ev); if (X > 1e-3) { ctx.start = [X, Y]; this.draw(ctx); } });
        this.draw(ctx);
      },
      draw(ctx) {
        const P = ctx.plot.clear();
        for (const f of [.3, .6, .9]) P.line(orbit(f * DELTA, 0), { color: HEX.muted, width: 1, alpha: .7 });
        let [x, y] = ctx.start; const pts = [[x, y]];
        for (let t = 0; t < T; t++) { [x, y] = [-(1 + ETA * y) * x, y + ETA * (AL - BE * x * x / 2)]; if (!isFinite(x) || Math.abs(x) > 1) break; pts.push([Math.abs(x), y]); }
        P.line(orbit(...ctx.start), { color: HEX.ball, width: 1, alpha: .5, dash: [3, 3] }).scatter(pts, { color: HEX.ball, r: 1.6 }).scatter([ctx.start], { color: HEX.ball, r: 4.5 })
          .scatter([[DELTA, 0]], { color: HEX.white, r: 5 }).hline(0, { color: HEX.flat, dash: [5, 4], alpha: .6 })
          .text(.004, .105, '2 blow-up', { color: HEX.muted }).text(.11, .105, '3 self-stabilization', { color: HEX.muted }).text(.11, -.105, '4 return', { color: HEX.muted }).text(.004, -.105, '1 sharpening', { color: HEX.muted }).draw();
        const g0 = gfun(ctx.start[0], ctx.start[1]), last = pts[pts.length - 1];
        ctx.stat(`start |x| = ${ctx.start[0].toFixed(3)}, S − 2/η = ${ctx.start[1].toFixed(3)} · g = ${g0.toFixed(3)} at start, ${gfun(last[0], last[1]).toFixed(3)} after ${pts.length - 1} GD steps · mean S − 2/η = ${(pts.reduce((a, p) => a + p[1], 0) / pts.length).toFixed(4)}`);
      },
    },
  ],
};
