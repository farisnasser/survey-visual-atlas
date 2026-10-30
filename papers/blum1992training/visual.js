// Blum & Rivest 1992 — training the 3-node network (2 hidden threshold units + 1 output) is NP-complete.
// The paper's own Set-Splitting example and its plane construction. Checked in scripts/check-math/blum1992training.py.
import { C, HEX } from '../../src/lib/atlas3d.js';

const THREE = window.THREE;
const S = 2.4, W = (x, y, z) => new THREE.Vector3((x - .5) * S, z * S * .9 + .15, -(y - .5) * S);
const POS = [[0, 0, 0], [1, 1, 0], [0, 1, 1]], NEG = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const dot = (a, x) => a[0] * x[0] + a[1] * x[1] + a[2] * x[2];

/** Polygon where the plane a.x = c cuts the box [lo, hi]^3 (world coordinates). */
function planePoly(a, c, lo = -.25, hi = 1.25) {
  const pts = [], V = [lo, hi];
  for (let ax = 0; ax < 3; ax++) for (const u of V) for (const v of V) {
    const b = [0, 1, 2].filter(i => i !== ax), p = [0, 0, 0]; p[b[0]] = u; p[b[1]] = v;
    if (Math.abs(a[ax]) < 1e-12) continue;
    p[ax] = (c - a[b[0]] * u - a[b[1]] * v) / a[ax];
    if (p[ax] >= lo - 1e-9 && p[ax] <= hi + 1e-9) pts.push(p);
  }
  if (pts.length < 3) return null;
  const cen = [0, 1, 2].map(i => pts.reduce((s, p) => s + p[i], 0) / pts.length);
  const n = a.map(v => v / Math.hypot(...a)), t = Math.abs(n[0]) < .9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = [n[1] * t[2] - n[2] * t[1], n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0]], e2 = [n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0]];
  pts.sort((p, q) => Math.atan2(dot(e2, p.map((v, i) => v - cen[i])), dot(e1, p.map((v, i) => v - cen[i]))) - Math.atan2(dot(e2, q.map((v, i) => v - cen[i])), dot(e1, q.map((v, i) => v - cen[i]))));
  return pts;
}
function addPlane(s, poly, color, opacity = .28) {
  const v = poly.map(p => W(...p)), g = new THREE.BufferGeometry(), pos = [];
  for (let i = 1; i < v.length - 1; i++) pos.push(...v[0].toArray(), ...v[i].toArray(), ...v[i + 1].toArray());
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.computeVertexNormals();
  s.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false })));
  s.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(v), new THREE.LineBasicMaterial({ color })));
}
function cube(s) {
  const c = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], e = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  e.forEach(([i, j]) => s.segment(W(...c[i]), W(...c[j]), C.muted, .35));
}

export default {
  tabs: [
    {
      id: 'split', name: 'Two planes', title: 'Training the 3-node net = splitting a set',
      caption: 'The paper’s example: S = {s₁, s₂, s₃}, subsets {s₁, s₂} and {s₂, s₃}. Each unit vector is a “−” point, each subset’s indicator a “+” point. A split (S₁, S₂) gives two hidden-unit planes; the net outputs “+” only where both units fire.',
      note: 'Plane i: aᵢ·x = −½ with aᵢⱼ = −1 if sⱼ is in Sᵢ, else n = 3. The planes classify all six points iff no subset lies inside S₁ or inside S₂ (Theorem 3, via Set-Splitting).',
      legend: [[HEX.good, '“+” example'], [HEX.neg, '“−” example'], [HEX.opt2, 'hidden unit 1 plane'], [HEX.opt3, 'hidden unit 2 plane'], [HEX.flat, 'misclassified']],
      cam: [3.6, 3.2, 4.2], tgt: [0, 1.1, 0],
      controls: [{ id: 'S1', type: 'seg', label: 'S₁ (rest is S₂)', value: '2', options: [['2', '{s₂}'], ['1', '{s₁}'], ['12', '{s₁, s₂}'], ['', '∅']] }],
      build(ctx) {
        const s = ctx.stage, S1 = new Set([...ctx.vals.S1].map(Number)), n = 3;
        const a = [1, 2, 3].map(j => S1.has(j) ? -1 : n), b = [1, 2, 3].map(j => S1.has(j) ? n : -1);
        cube(s);
        const pa = planePoly(a, -.5), pb = planePoly(b, -.5);
        if (pa) addPlane(s, pa, C.opt2); if (pb) addPlane(s, pb, C.opt3, .18);
        const cls = x => dot(a, x) > -.5 && dot(b, x) > -.5;
        let ok = 0;
        for (const [pts, want] of [[POS, true], [NEG, false]]) for (const x of pts) {
          const right = cls(x) === want; ok += right;
          s.dot(W(...x), want ? C.good : C.neg, .09);
          if (!right) s.dot(W(...x), C.flat, .13).material.wireframe = true;
          s.label(`${x.join('')} ${want ? '+' : '−'}`, W(x[0], x[1], x[2] + .12), '', right ? (want ? HEX.good : HEX.neg) : HEX.flat);
        }
        const C2 = [[1, 2], [2, 3]], split = C2.every(c => !c.every(j => S1.has(j)) && !c.every(j => !S1.has(j)));
        ctx.stat(`a = (${a.join(', ')}), b = (${b.join(', ')}) · ${ok}/6 examples correct · ${split ? 'valid set-splitting: every subset is cut' : 'not a set-splitting: some subset sits inside one side'}`);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
    },
    {
      id: 'xor', name: 'Lifted XOR', title: 'Add pairwise products and the problem becomes linear',
      caption: 'XOR is not linearly separable in (x₁, x₂). Lift each input by the product z = x₁x₂ and one plane, x₁ + x₂ − 2z = ½, separates it. The paper’s positive result: what the XOR-output 3-node net can classify is linearly separable after adding all pairwise products (O(n²) inputs), and a perceptron on those is trainable by linear programming.',
      note: 'Changing the representation changes the complexity: the hard part is choosing the hidden units, not fitting a linear rule.',
      legend: [[HEX.good, 'XOR = 1'], [HEX.neg, 'XOR = 0'], [HEX.opt2, 'separating plane (lifted)']],
      cam: [3.4, 2.8, 4.0], tgt: [0, .8, 0],
      controls: [{ id: 'lift', type: 'seg', label: 'features', value: 'raw', options: [['raw', 'x₁, x₂'], ['lift', 'x₁, x₂, x₁x₂']] }],
      build(ctx) {
        const s = ctx.stage, L = ctx.vals.lift === 'lift'; cube(s);
        const X = [[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]];
        X.forEach(([x1, x2, y]) => { const z = L ? x1 * x2 : 0; s.dot(W(x1, x2, z), y ? C.good : C.neg, .1); s.label(`(${x1},${x2}) → ${y}`, W(x1, x2, z + .14), '', y ? HEX.good : HEX.neg); });
        if (L) addPlane(s, planePoly([1, 1, -2], .5), C.opt2);
        ctx.stat(L ? 'lifted: x₁ + x₂ − 2x₁x₂ > ½ exactly on the XOR = 1 points · one linear unit suffices' : 'raw inputs: no line separates (0,1),(1,0) from (0,0),(1,1)');
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
    },
  ],
};
