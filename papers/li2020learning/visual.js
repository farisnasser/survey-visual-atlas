// Li, Ma, Zhang 2020 — a trained two-layer ReLU net learns f*(x) = Σ aᵢ|w*ᵢ·x| by moving its neurons onto the teacher
// directions (feature learning), which fixed features cannot do. Exact population training replayed from
// scripts/check-math/li2020learning.py (data.json): d = 3, W* = I, aᵢ = 1/3, m = 60, truncated GD (Algorithm 1).
import { C, HEX } from '../../src/lib/atlas3d.js';

const THREE = window.THREE;
const S = .9;
const V = w => new THREE.Vector3(w[0] * S, w[2] * S, -w[1] * S);
const rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
const massNearAxes = W => { let num = 0, den = 0; for (const w of W) { const n2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2], n = Math.sqrt(n2), c = Math.max(...w.map(v => Math.abs(v))) / n; den += n2; if (c > Math.cos(15 * Math.PI / 180)) num += n2; } return num / den; };

export default {
  tabs: [
    {
      id: 'neurons', name: 'Neurons find the teacher', title: 'Gradient descent moves the neurons onto the six teacher directions ±eᵢ',
      caption: 'Each coral dot is one of the 60 hidden neurons wᵢ ∈ ℝ³ of the student f_W(x) = (1/60) Σ ‖wᵢ‖ ReLU(wᵢ·x); green rays are the teacher’s directions ±e₁, ±e₂, ±e₃ (f* = ⅓Σ|xᵢ| is six ReLUs). Slide through training: the neurons start as a small random cloud, grow, and gather on the six rays — the features themselves change, which a fixed-kernel (NTK) model cannot do.',
      note: 'Exact population loss (Gaussian x, arc-cosine kernel), truncated GD with η = 2 and threshold ‖w‖² ≤ 25 as in Algorithm 1; d = 3 so the picture is literal. The paper’s theorem is about large d.',
      legend: [[HEX.ball, 'student neurons wᵢ'], [HEX.good, 'teacher directions ±eᵢ']],
      cam: [2.6, 2.0, 3.0], tgt: [0, 0, 0],
      controls: [{ id: 'k', label: 'GD step', min: 0, max: 6000, step: 150, value: 6000, fmt: v => String(Math.round(v)) }],
      actions: [{ label: 'Play training', fn: ctx => { ctx.play = true; ctx.acc = 0; ctx.set('k', 0); ctx.shell.tab.draw(ctx); } }],
      build(ctx) {
        const s = ctx.stage; s.clear(); ctx.objs = [];
        for (let i = 0; i < 3; i++) for (const sg of [1, -1]) { const e = [0, 0, 0]; e[i] = 2 * sg; s.segment(V([0, 0, 0]), V(e), C.good, .9); s.dot(V(e.map(v => v * .5)), C.good, .025); }
        const circ = (f) => Array.from({ length: 97 }, (_, i) => V(f(2 * Math.PI * i / 96)));
        for (const f of [t => [Math.cos(t), Math.sin(t), 0], t => [Math.cos(t), 0, Math.sin(t)], t => [0, Math.cos(t), Math.sin(t)]])
          s.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(circ(f)), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: .12 })));
        s.label('e₁', V([2.15, 0, 0]), '', HEX.good); s.label('e₂', V([0, 2.15, 0]), '', HEX.good); s.label('e₃', V([0, 0, 2.15]), '', HEX.good);
        this.draw(ctx);
      },
      draw(ctx) {
        const D = ctx.data, s = ctx.stage, j = Math.round(ctx.vals.k / 150), W = D.frames[j];
        (ctx.objs || []).forEach(o => s.remove(o)); ctx.objs = [];
        ctx.objs.push(s.points(W.map(w => V(w).toArray()), W.map(() => rgb(HEX.ball)), .07));
        const segs = []; for (const w of W) segs.push(V([0, 0, 0]), V(w));
        ctx.objs.push(s.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segs), new THREE.LineBasicMaterial({ color: C.ball, transparent: true, opacity: .25 }))));
        const L = D.hist[Math.min(Math.round(ctx.vals.k / 20), D.hist.length - 1)];
        ctx.stat(`GD step ${Math.round(ctx.vals.k)} · population loss ${L.toExponential(2)} · neuron mass within 15° of a teacher direction: ${(100 * massNearAxes(W)).toFixed(1)}%`);
      },
      frame(ctx, dt) { if (!ctx.play) return; ctx.acc += dt * 4; if (ctx.acc < 1) return; ctx.acc = 0; const k = Math.min(6000, ctx.vals.k + 150); ctx.set('k', k); this.draw(ctx); if (k >= 6000) ctx.play = false; },
      onControl(ctx) { ctx.play = false; this.draw(ctx); },
    },
    {
      id: 'vsfixed', name: 'Learned vs fixed features', mode: '2d', title: 'Moving features beats the best fit on the same features frozen at initialization',
      caption: 'Population loss of the trained network (coral) against the best possible linear fit on its own 60 random initial features relu(wᵢ(0)·x) (dashed), a kernel/random-feature method of the same size. Training passes it after about 1 240 steps and ends 35× lower. Right: share of neuron mass aligned with a teacher direction. Theorem 1.2 makes the gap a theorem in high dimension: every polynomial-size feature map or kernel method has loss Ω(1/d), while Algorithm 1 reaches O(1/d^{1+Q}).',
      note: 'At d = 3 this is an illustration, not the asymptotic statement. In our runs at d = 8 and 16 (m = 20d, 3 000 steps) the trained loss was still on a plateau at the random-feature level — the paper’s second stage needs Θ(d^{1+10Q}/η) steps.',
      hint: '', legend: [[HEX.ball, 'trained network'], [HEX.muted, 'fixed random features, same width']],
      plot: { xlabel: 'GD step', ylabel: 'population loss', logy: true, xlim: [0, 6000], ylim: [1e-5, 1], legend: false },
      charts: [{ id: 'm', title: 'mass within 15° of ±eᵢ', xlabel: 'GD step', ylabel: 'share', ylim: [0, 1.02], h: 220 }],
      build(ctx) {
        const D = ctx.data;
        ctx.plot.clear().hline(D.rf, { color: HEX.muted, dash: [5, 4], width: 1.4, text: 'fixed features' }).line(D.hist.map((v, i) => [i * 20, v]), { color: HEX.ball, width: 2.4 }).draw();
        ctx.charts.m.clear().line(D.frames.map((W, j) => [j * 150, massNearAxes(W)]), { color: HEX.ball, width: 2 }).draw();
        ctx.stat(`final loss ${D.hist.at(-1).toExponential(2)} vs fixed-feature optimum ${D.rf.toExponential(2)} · loss below 1% of its start after ${D.t1} steps · aligned mass ${(100 * D.m0).toFixed(1)}% → ${(100 * D.mass).toFixed(1)}%`);
      },
    },
  ],
};
