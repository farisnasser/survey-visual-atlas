// Chizat, Oyallon, Bach 2019 — lazy training: scaling a model by α makes training follow its linearization at init.
// Toy model and the paper's Fig. 1 replication from scripts/check-math/chizat2019lazy.py (data.json).
import { def, opt, C, HEX, statLabels } from '../../src/lib/atlas3d.js';

const h = (a, b) => a + .5 * b + a * b;                 // h(0) = 0, Dh(0) = (1, 0.5), D²h ≠ 0
const hb = (a, b) => a + .5 * b;                         // linearization at w₀ = 0
const DOMS = a => [-1.6 / a, 1.6 / a, -1.6 / a, 1.6 / a];

export default {
  tabs: [
    {
      id: 'scale', name: 'Scale makes it linear', title: 'Scale the output by α and the loss around the start looks like its linearization',
      caption: 'Left: F_α(w) = (αh(w) − 1)²/(2α²) for the non-linear model h(w) = w₁ + ½w₂ + w₁w₂. Right: the same for the linearized model h̄(w) = w₁ + ½w₂ (a straight valley). Both are drawn on a window of width ∝ 1/α around w₀ = 0, and both balls run gradient descent from w₀. At α = 1 the true landscape is curved and its path differs; as α grows the two pictures and paths become the same: the parameters move only O(1/α) and stay O(1/α²) from the linearized path (Theorem 2.2).',
      note: 'Square loss, target 1. The window shrinks with α so the motion stays visible; heights are fitted per surface.',
      legend: [[HEX.ball, 'GD on the model'], [HEX.opt3, 'GD on its linearization'], [HEX.good, 'start w₀ = 0']],
      cam: [0, 4.0, 4.6], tgt: [0, .2, 0], action: 'Run again',
      controls: [{ id: 'a', type: 'seg', label: 'scale α', value: 1, options: [[1, '1'], [4, '4'], [16, '16'], [64, '64']] }],
      charts: [{ id: 'g', title: 'distance between the two paths × α²', xlabel: 'time t', ylabel: '‖w − w̄‖·α²', h: 210 }],
      build(ctx) {
        const al = ctx.vals.a, s = ctx.stage; s.clear();
        ctx.A = def({ L: (x, y) => .5 * (al * h(x, y) - 1) ** 2 / al ** 2, g: (x, y) => { const r = (al * h(x, y) - 1) / al; return [r * (1 + y), r * (.5 + x)]; }, dom: DOMS(al), ox: -1.8, size: 3, h: v => Math.sqrt(v), checkDom: DOMS(al) });
        ctx.B = def({ L: (x, y) => .5 * (al * hb(x, y) - 1) ** 2 / al ** 2, g: (x, y) => { const r = (al * hb(x, y) - 1) / al; return [r, .5 * r]; }, dom: DOMS(al), ox: 1.8, size: 3, h: v => Math.sqrt(v), checkDom: DOMS(al) });
        s.surface(ctx.A); s.surface(ctx.B); s.marker(ctx.A, 0, 0, C.good, .08); s.marker(ctx.B, 0, 0, C.good, .08);
        ctx.lab = statLabels(s, [['model h', ctx.A.W(0, 1.6 / al, .4)], ['linearization h̄', ctx.B.W(0, 1.6 / al, .4)]]);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls(); const dt = .02;
        ctx.p = s.ball(ctx.A, { start: [0, 0], step: opt.gd(dt), speed: 2, maxSteps: 300 });
        ctx.q = s.ball(ctx.B, { start: [0, 0], step: opt.gd(dt), color: C.opt3, speed: 2, maxSteps: 300 });
        ctx.gap = [[0, 0]]; ctx.p.onStep = z => ctx.gap.push([z.steps * dt, Math.hypot(z.p[0] - ctx.q.p[0], z.p[1] - ctx.q.p[1]) * ctx.vals.a ** 2]);
      },
      onControl(ctx) { this.build(ctx); },
      frame(ctx) {
        const { p, q } = ctx; if (!p) return; const al = ctx.vals.a;
        ctx.charts.g.clear().set({ xlim: [0, 6] }).line(ctx.gap, { color: HEX.ball, width: 2 }).draw();
        ctx.lab[0].stat(`output αh = ${(al * h(...p.p)).toFixed(3)}`); ctx.lab[1].stat(`output αh̄ = ${(al * hb(...q.p)).toFixed(3)}`);
        ctx.stat(`α = ${al} · t = ${(p.steps * .02).toFixed(2)} · ‖w − w₀‖ = ${Math.hypot(...p.p).toExponential(2)} (×α: ${(Math.hypot(...p.p) * al).toFixed(3)}) · ‖w − w̄‖ = ${Math.hypot(p.p[0] - q.p[0], p.p[1] - q.p[1]).toExponential(2)}`);
      },
    },
    {
      id: 'neurons', name: 'Lazy vs active neurons', mode: '2d', title: 'Large initialization: the network fits the data while its neurons stay put',
      caption: 'Replication of the paper’s Fig. 1(a)–(b): a two-layer ReLU network (m = 20, symmetrized so its output starts at 0) learns a teacher with 3 neurons (dashed rays) from 15 points on the circle. Each curve is one neuron drawn as |a_j|·b_j (coral a_j > 0, violet a_j < 0), dots at the end. With a small initialization (τ = 0.1) the neurons travel to the teacher directions; with a large one (τ = 5) they barely move — lazy training — although both fit the data.',
      note: 'GD with step 1/λ_max of the tangent kernel at init (≤ 0.2). Right: relative weight change after training vs τ, falling like τ⁻² — the lazy regime.',
      hint: '', legend: [[HEX.ball, 'a_j > 0'], [HEX.opt2, 'a_j < 0'], [HEX.good, 'teacher neurons']],
      controls: [{ id: 'tau', type: 'seg', label: 'init scale τ', value: '0.1', options: [['0.1', '0.1 (active)'], ['5.0', '5 (lazy)']] }],
      plot: { xlabel: '(|a|b)₁', ylabel: '(|a|b)₂', legend: false },
      charts: [{ id: 'rel', title: 'relative weight change ‖Δw‖/‖w₀‖', xlabel: 'τ', ylabel: 'ratio', logx: true, logy: true, h: 230 }],
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const D = ctx.data, tau = ctx.vals.tau, P = D[`path${tau}`], last = P[P.length - 1], R = Math.max(1.4, ...last.map(v => Math.max(Math.abs(v[0]), Math.abs(v[1])))) * 1.1;
        const pl = ctx.plot.clear().set({ xlim: [-R * 1.4, R * 1.4], ylim: [-R, R] });
        D.TB.forEach((b, k) => pl.line([[0, 0], [b[0] * R * 2, b[1] * R * 2]], { color: HEX.good, dash: [5, 4], width: 1.2 }));
        pl.line(Array.from({ length: 101 }, (_, i) => [Math.cos(2 * Math.PI * i / 100), Math.sin(2 * Math.PI * i / 100)]), { color: HEX.muted, width: .8, alpha: .5 });
        for (let j = 0; j < last.length; j++) { const col = last[j][2] > 0 ? HEX.ball : HEX.opt2; pl.line(P.map(F => [F[j][0], F[j][1]]), { color: col, width: 1.3, alpha: .85 }).scatter([[last[j][0], last[j][1]]], { color: col, r: 3 }); }
        pl.draw();
        ctx.charts.rel.clear().line(D.scan, { color: HEX.ball, width: 2, dots: 3 }).fn(t => D.scan[3][1] * (t / D.scan[3][0]) ** -2, { color: HEX.muted, dash: [4, 4], width: 1, label: '∝ τ⁻²' }).draw();
        ctx.stat(`τ = ${tau}: training loss ${D[`loss${tau}`][0].toFixed(3)} → ${D[`loss${tau}`].at(-1).toExponential(1)}, relative weight change ${D[`rel${tau}`]}`);
      },
    },
  ],
};
