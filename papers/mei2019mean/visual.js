// Mei, Misiakiewicz, Montanari 2019 — mean-field SGD on the anisotropic-Gaussian problem of Sec. 3.2 (fixed coefficients).
// Neuron snapshots, risk curves and deviations come from scripts/check-math/mei2019mean.py (data.json);
// the surface R∞(r₁, r₂) is the closed form verified there.
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const erf = x => { const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + .3275911 * a); return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-a * a)); };
const Phi = z => .5 * (1 + erf(z / Math.SQRT2)), phi = z => Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);
/** R∞ for the paper's model when every neuron has ‖P_V w‖ = r₁, ‖P_⊥ w‖ = r₂ (uniform directions, d → ∞). */
const makeRinf = ([DEL, T1, T2, S1, S2]) => {
  const Er = (s, a) => s * phi(a / s) - a * (1 - Phi(a / s));
  const Es = s => s < 1e-9 ? (0 <= T1 ? S1 : S2) : S1 + (S2 - S1) / (T2 - T1) * (Er(s, T1) - Er(s, T2));
  return (r1, r2) => .5 * [-1, 1].reduce((acc, y) => acc + (y - Es(Math.sqrt((1 + DEL * y) ** 2 * r1 * r1 + r2 * r2))) ** 2, 0);
};
const NCOL = { 50: HEX.opt3, 200: HEX.opt2, 800: HEX.ball };
const rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);

export default {
  tabs: [
    {
      id: 'neurons', name: 'Neurons as a population', title: 'SGD moves a cloud of neurons; the cloud, not each neuron, is what the theory follows',
      caption: 'Each coral dot is one hidden neuron of a 3 200-neuron network trained by one-pass SGD on the paper’s problem (two classes of centred Gaussians that differ only in variance on a hidden subspace V of half the dimensions). Its position is (‖P_V w‖, ‖P_⊥ w‖). The surface is R∞(r₁, r₂), the risk if every neuron had that profile (d → ∞). The cloud starts uninformative near (0.7, 0.7); SGD grows the neurons along V while their off-subspace norm stays near 0.7, and the cloud ends around the minimum of R∞ (1.55, 0.70).',
      note: 'Truncated ReLU with t₁ = 0.5, t₂ = 1.5, s₁ = −2.5, s₂ = 7.5 and Δ = 0.6 (our choice); step ε = 0.05/d; every 8th neuron shown. The real network is a mixture of profiles, so its risk is not the surface height under any one dot.',
      legend: [[HEX.ball, 'neurons (r₁, r₂)'], [HEX.good, 'minimum of R∞']],
      cam: [3.4, 3.4, 3.6], tgt: [0, .2, 0],
      controls: [{ id: 'd', type: 'seg', label: 'dimension d', value: 80, options: [[20, '20'], [80, '80']] }],
      actions: [{ label: 'Replay', fn: ctx => { ctx.k = 0; ctx.acc = 0; } }],
      build(ctx) {
        const D = ctx.data, s = ctx.stage; s.clear();
        const R = makeRinf(D.consts);
        ctx.F = def({ L: R, h: v => Math.log1p(v * 6), dom: [0, 3, 0, 3], N: 80 });
        s.surface(ctx.F);
        let best = [0, 0, Infinity]; for (let i = 0; i <= 60; i++) for (let j = 0; j <= 60; j++) { const v = R(3 * i / 60, 3 * j / 60); if (v < best[2]) best = [3 * i / 60, 3 * j / 60, v]; }
        ctx.best = best; s.marker(ctx.F, best[0], best[1], C.good, .1);
        ctx.k = 0; ctx.acc = 0; ctx.pts = null; this.place(ctx);
      },
      place(ctx) {
        const D = ctx.data, s = ctx.stage, fr = D[`snap${ctx.vals.d}`], P = fr[Math.min(ctx.k, fr.length - 1)];
        if (ctx.pts) s.remove(ctx.pts);
        ctx.pts = s.points(P.map(([a, b]) => ctx.F.W(Math.min(a, 3), Math.min(b, 3), .05).toArray()), P.map(() => rgb(HEX.ball)), .06);
        const t = D[`t${ctx.vals.d}`], ti = t[Math.min(ctx.k * 4, t.length - 1)], m1 = P.reduce((a, p) => a + p[0], 0) / P.length, m2 = P.reduce((a, p) => a + p[1], 0) / P.length;
        const Rk = D[`ref${ctx.vals.d}`][Math.min(ctx.k * 4, t.length - 1)];
        ctx.stat(`d = ${ctx.vals.d} · time t = ${ti.toFixed(2)} · mean profile (‖P_V w‖, ‖P_⊥ w‖) = (${m1.toFixed(2)}, ${m2.toFixed(2)}) · network risk ${Rk.toFixed(3)} · R∞ minimum ${ctx.best[2].toFixed(4)} at (${ctx.best[0].toFixed(2)}, ${ctx.best[1].toFixed(2)})`);
      },
      frame(ctx, dt) { const fr = ctx.data[`snap${ctx.vals.d}`]; if (!fr || ctx.k >= fr.length - 1) return; ctx.acc += dt * 2; if (ctx.acc < 1) return; ctx.acc = 0; ctx.k++; this.place(ctx); },
      onControl(ctx) { this.build(ctx); },
    },
    {
      id: 'dimfree', name: 'Dimension-free error', mode: '2d', title: 'Finite-width SGD follows the mean-field curve with an error that does not grow with d',
      caption: 'Risk of one-pass SGD with N = 50, 200, 800 neurons against the N = 3 200 run (white dashed), our proxy for the mean-field PDE. Right: the largest gap over time, averaged over 3 seeds, for d = 20 and d = 80. Theorem 1(A) bounds it by K e^{KT}(√log N + z)/√N plus a √ε term — no dimension factor on the 1/√N term; here the gap at d = 80 is no larger than at d = 20.',
      note: 'The mean-field risk curve itself depends on d (here s₀ = d/2 informative directions), so the two dimensions are compared through the gap, not the curves. The gap stops shrinking past N ≈ 200 at d = 20: the √ε term and the proxy’s own fluctuations take over.',
      hint: '', legend: [[HEX.opt3, 'N = 50'], [HEX.opt2, 'N = 200'], [HEX.ball, 'N = 800'], [HEX.white, 'N = 3 200 (proxy)']],
      controls: [{ id: 'd', type: 'seg', label: 'dimension d', value: 80, options: [[20, '20'], [80, '80']] }],
      plot: { xlabel: 'time t = kε', ylabel: 'population risk', legend: false },
      charts: [{ id: 'dev', title: 'sup_t |R_N − R_3200|', xlabel: 'neurons N', ylabel: 'gap', logx: true, logy: true, xlim: [30, 1500], ylim: [.01, 1], h: 240 }],
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const D = ctx.data, d = ctx.vals.d, t = D[`t${d}`], P = ctx.plot.clear();
        for (const N of D.Ns) P.line(D[`R${d}_${N}`].map((v, i) => [t[i], v]), { color: NCOL[N], width: 1.8 });
        P.line(D[`ref${d}`].map((v, i) => [t[i], v]), { color: HEX.white, dash: [5, 4], width: 2 }).draw();
        const ch = ctx.charts.dev.clear().fn(N => D.dev['20_50'] * Math.sqrt(50 / N), { color: HEX.muted, dash: [3, 3], width: 1, label: '∝ 1/√N' });
        for (const dd of D.ds) ch.line(D.Ns.map(N => [N, D.dev[`${dd}_${N}`]]), { color: dd === 20 ? HEX.opt2 : HEX.ball, width: 2, dots: 3, label: `d = ${dd}` });
        ch.draw();
        ctx.stat(D.Ns.map(N => `N = ${N}: gap ${D.dev[`20_${N}`].toFixed(3)} (d = 20) vs ${D.dev[`80_${N}`].toFixed(3)} (d = 80)`).join(' · '));
      },
    },
  ],
};
