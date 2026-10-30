// atlas3d — the shared 3D layer for every paper visual. Ported from reference/saddles.html and generalised.
// Uses the global THREE (three@0.128.0 + OrbitControls, loaded by the page shell).
//
// Honesty contract: a surface is always drawn from a real function L(x, y); balls run a real optimizer on that
// same L (display height may be rescaled with def.h, the dynamics never are).
import { eig2, hess2, clamp } from './math.js';

const THREE = window.THREE;
export const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches;
export const C = { ball: 0xff7a59, opt2: 0xb28cff, opt3: 0xf2f4ff, neg: 0xff5d73, pos: 0x7cc4ff, flat: 0xffc857, good: 0x5fe0b0, white: 0xffffff, muted: 0x8e99bd };
export const HEX = { ball: '#ff7a59', opt2: '#b28cff', opt3: '#f2f4ff', neg: '#ff5d73', pos: '#7cc4ff', flat: '#ffc857', good: '#5fe0b0', white: '#ffffff', muted: '#8e99bd' };
export const logH = L => Math.log1p(Math.max(L, 0)) * 0.7;
const STOPS = [[0x1b, 0x1f, 0x4b], [0x24, 0x4f, 0x86], [0x2a, 0x9c, 0x96], [0x96, 0xd0, 0x78], [0xf4, 0xe8, 0xa4]];
export const ramp = t => {
  t = clamp(t, 0, 1) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t)), u = t - i, a = STOPS[i], b = STOPS[i + 1];
  return [0, 1, 2].map(k => (a[k] + (b[k] - a[k]) * u) / 255);
};

/** A surface definition. L(x,y) true loss; g(x,y) its gradient (numerical if omitted);
 *  h(L) display height; dom [x0,x1,y0,y1]; ox world x-offset (for side-by-side scenes). */
export function def(o) {
  // fit: display height range in world units (null = use h as given). square: separate x/y scales for a square footprint.
  const d = Object.assign({ h: logH, dom: [-1.6, 1.6, -1.6, 1.6], N: 110, ox: 0, size: 3.2, hClip: [-Infinity, Infinity], fit: 1.9, square: false, hScale: 1, hOff: 0 }, o);
  d._analytic = !!o.g;
  if (!d.g) d.g = (x, y) => { const e = 1e-6; return [(d.L(x + e, y) - d.L(x - e, y)) / (2 * e), (d.L(x, y + e) - d.L(x, y - e)) / (2 * e)]; };
  const [x0, x1, y0, y1] = d.dom;
  d.cx = (x0 + x1) / 2; d.cy = (y0 + y1) / 2;
  d.s = d.size / Math.max(x1 - x0, y1 - y0);
  d.sx = d.square ? d.size / (x1 - x0) : d.s; d.sy = d.square ? d.size / (y1 - y0) : d.s;
  d.raw = (x, y) => clamp(d.h(d.L(x, y)), d.hClip[0], d.hClip[1]);
  d.H = (x, y) => d.raw(x, y) * d.hScale + d.hOff;
  d.W = (x, y, lift = 0) => new THREE.Vector3((x - d.cx) * d.sx + d.ox, d.H(x, y) + lift, -(y - d.cy) * d.sy);
  d.in = (x, y) => x >= x0 && x <= x1 && y >= y0 && y <= y1;
  d.clampP = p => [clamp(p[0], x0, x1), clamp(p[1], y0, y1)];
  return d;
}

/** Surface from a precomputed grid (bilinear). grid[j][i] at x = a0 + i*(a1-a0)/(n-1), y = b0 + j*(...). */
export function gridDef(grid, dom, o = {}) {
  const ny = grid.length, nx = grid[0].length, [x0, x1, y0, y1] = dom;
  const L = (x, y) => {
    const fx = clamp((x - x0) / (x1 - x0) * (nx - 1), 0, nx - 1.000001), fy = clamp((y - y0) / (y1 - y0) * (ny - 1), 0, ny - 1.000001);
    const i = Math.floor(fx), j = Math.floor(fy), u = fx - i, v = fy - j;
    return (1 - u) * (1 - v) * grid[j][i] + u * (1 - v) * grid[j][i + 1] + (1 - u) * v * grid[j + 1][i] + u * v * grid[j + 1][i + 1];
  };
  return def(Object.assign({ L, dom, N: Math.min(140, Math.max(nx, ny) * 2) }, o));
}

// ---------- optimizer steps: (p, ball, def) -> new p ----------
export const opt = {
  gd: lr => (p, b, d) => { const [gx, gy] = d.g(p[0], p[1]); return [p[0] - lr * gx, p[1] - lr * gy]; },
  sgd: (lr, sigma, r) => (p, b, d) => { const [gx, gy] = d.g(p[0], p[1]); return [p[0] - lr * (gx + sigma * r.normal()), p[1] - lr * (gy + sigma * r.normal())]; },
  momentum: (lr, beta) => (p, b, d) => {
    const [gx, gy] = d.g(p[0], p[1]); b.v = b.v || [0, 0];
    b.v = [beta * b.v[0] - lr * gx, beta * b.v[1] - lr * gy]; return [p[0] + b.v[0], p[1] + b.v[1]];
  },
  /** Newton-type step with Hessian eigen-rescaling: kind 'newton' (signed) or 'sfn' (|lambda|); delta = damping floor. */
  curv: (lr, kind, delta = 0.1) => (p, b, d) => {
    const [gx, gy] = d.g(p[0], p[1]);
    const [a, c, e] = d.hess ? d.hess(p[0], p[1]) : hess2(d.L, p[0], p[1]);
    const ev = eig2(a, c, e);
    let sx = 0, sy = 0;
    for (const [lam, v] of ev) {
      let l = kind === 'sfn' ? Math.max(Math.abs(lam), delta) : (Math.abs(lam) < delta ? (lam >= 0 ? delta : -delta) : lam);
      const proj = (gx * v[0] + gy * v[1]) / l;
      sx += proj * v[0]; sy += proj * v[1];
    }
    return [p[0] - lr * sx, p[1] - lr * sy];
  },
};

export class Stage {
  constructor(el, labelEl) {
    this.el = el; this.labelEl = labelEl;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(2, devicePixelRatio));
    this.renderer.setClearColor(0x0b0e1a);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x0b0e1a, 9, 18);
    this.camera = new THREE.PerspectiveCamera(42, 1, .05, 60);
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: .08, enablePan: false, minDistance: 2.2, maxDistance: 14, maxPolarAngle: Math.PI * .49 });
    this.scene.add(new THREE.HemisphereLight(0xbcd2ff, 0x10122a, .75));
    const sun = new THREE.DirectionalLight(0xffffff, .75); sun.position.set(3, 6, 4); this.scene.add(sun);
    this.world = new THREE.Group(); this.scene.add(this.world);
    this.balls = []; this.labels = []; this.frameFns = []; this.running = false; this.visible = true;
    new ResizeObserver(() => this.resize()).observe(el);
    new IntersectionObserver(es => { this.visible = es[0].isIntersecting; }).observe(el);
    document.addEventListener('visibilitychange', () => { this.hidden = document.hidden; });
    this.resize();
  }
  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.fov = w < h ? 58 : 42;
    this.camera.updateProjectionMatrix();
  }
  clear() {
    this.scene.remove(this.world);
    this.world.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose?.(); });
    this.world = new THREE.Group(); this.scene.add(this.world);
    this.balls = []; this.labels.forEach(L => L.e.remove()); this.labels = [];   // frame callbacks belong to the tab, not the scene
  }
  view(cam, tgt) { this.camera.position.set(...cam); this.controls.target.set(...tgt); this.controls.update(); }
  add(o) { this.world.add(o); return o; }
  remove(o) { this.world.remove(o); o.geometry?.dispose(); }

  /** Height-coloured surface with faint grid lines. Returns the def (with .mesh). */
  surface(d, { grid = true, opacity = 1, colorFn = null } = {}) {
    const N = d.N, [x0, x1, y0, y1] = d.dom, pos = [], col = [], idx = [], hs = [], raw = [];
    let rmin = Infinity, rmax = -Infinity;
    for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) { const v = d.raw(x0 + (x1 - x0) * i / N, y0 + (y1 - y0) * j / N); raw.push(v); if (isFinite(v)) { rmin = Math.min(rmin, v); rmax = Math.max(rmax, v); } }
    if (d.fit && !d._fitted) { d.hScale = d.fit / Math.max(1e-12, rmax - rmin); d.hOff = -rmin * d.hScale; d._fitted = true; }   // display only
    let hmin = Infinity, hmax = -Infinity;
    for (let j = 0, k = 0; j <= N; j++) for (let i = 0; i <= N; i++, k++) {
      const x = x0 + (x1 - x0) * i / N, y = y0 + (y1 - y0) * j / N, h = raw[k] * d.hScale + d.hOff;
      hs.push(h); hmin = Math.min(hmin, h); hmax = Math.max(hmax, h);
      pos.push((x - d.cx) * d.sx + d.ox, h, -(y - d.cy) * d.sy);
    }
    const span = Math.max(1e-9, hmax - hmin);
    hs.forEach((h, k) => {
      if (colorFn) { const i = k % (N + 1), j = Math.floor(k / (N + 1)); col.push(...colorFn(x0 + (x1 - x0) * i / N, y0 + (y1 - y0) * j / N, (h - hmin) / span)); }
      else col.push(...ramp(Math.pow((h - hmin) / span, .6)));
    });
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const a = j * (N + 1) + i, b = a + 1, c = a + N + 1, e = c + 1; idx.push(a, c, b, b, c, e); }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx); g.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: .65, metalness: .05, side: THREE.DoubleSide, transparent: opacity < 1, opacity });
    d.mesh = this.add(new THREE.Mesh(g, mat));
    d.hmin = hmin; d.hmax = hmax;
    if (grid) {
      const lp = [], step = Math.max(1, Math.round(N / 11)), P = (i, j) => pos.slice((j * (N + 1) + i) * 3, (j * (N + 1) + i) * 3 + 3);
      for (let k = 0; k <= N; k += step) for (let m = 0; m < N; m++) lp.push(...P(k, m), ...P(k, m + 1), ...P(m, k), ...P(m + 1, k));
      const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3));
      d.gridLines = this.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: .1 })));
    }
    return d;
  }
  /** Recolour an existing surface (e.g. heat map of |grad|). fn(x, y) -> [r,g,b] in 0..1 */
  recolor(d, fn) {
    const N = d.N, [x0, x1, y0, y1] = d.dom, c = d.mesh.geometry.attributes.color;
    for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) { const rgb = fn(x0 + (x1 - x0) * i / N, y0 + (y1 - y0) * j / N); c.setXYZ(j * (N + 1) + i, rgb[0], rgb[1], rgb[2]); }
    c.needsUpdate = true;
  }
  tube(d, pts, color, r = .018, lift = .02) {
    const v = pts.filter(p => d.in(p[0], p[1])).map(([x, y]) => d.W(x, y, lift));
    if (v.length < 2) return null;
    return this.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(v), Math.max(8, v.length * 2), r, 8, false), new THREE.MeshBasicMaterial({ color })));
  }
  line(d, pts, color, opacity = .9, lift = .015) {
    const v = pts.filter(p => d.in(p[0], p[1])).map(([x, y]) => d.W(x, y, lift));
    return this.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(v), new THREE.LineBasicMaterial({ color, transparent: true, opacity })));
  }
  marker(d, x, y, color = C.white, r = .1) {
    const m = new THREE.Mesh(new THREE.TorusGeometry(r, .014, 8, 40), new THREE.MeshBasicMaterial({ color }));
    m.rotation.x = Math.PI / 2; m.position.copy(d.W(x, y, .02)); return this.add(m);
  }
  dot(v, color, r = .045) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 12), new THREE.MeshBasicMaterial({ color }));
    m.position.copy(v); return this.add(m);
  }
  segment(a, b, color, opacity = 1) {
    return this.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity })));
  }
  /** Points cloud: positions = [[x,y,z],...] in world units, colors = [[r,g,b],...] */
  points(positions, colors, size = .03, opacity = 1) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions.flat(), 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(colors.flat(), 3));
    return this.add(new THREE.Points(g, new THREE.PointsMaterial({ size, vertexColors: true, transparent: opacity < 1, opacity, sizeAttenuation: true })));
  }
  /** Hessian eigen-directions at (x,y) as tubes coloured by sign, with short lambda labels. */
  eigenLines(d, x, y, { len = 1.0, eps = 1e-3, labels = true, fmtL = l => (Math.abs(l) < 5e-3 ? '0' : l.toFixed(Math.abs(l) < 1 ? 2 : 1)) } = {}) {
    const [a, b, e] = d.hess ? d.hess(x, y) : hess2(d.L, x, y);
    const out = [];
    for (const [lam, v] of eig2(a, b, e)) {
      const col = Math.abs(lam) < eps ? C.flat : lam < 0 ? C.neg : C.pos, hex = Math.abs(lam) < eps ? HEX.flat : lam < 0 ? HEX.neg : HEX.pos;
      const pts = []; for (let t = -len; t <= len + 1e-9; t += len / 40) pts.push([x + t * v[0], y + t * v[1]]);
      this.tube(d, pts, col);
      if (labels) {
        const t = len * .8, px = x + t * v[0], py = y + t * v[1];
        if (d.in(px, py)) this.label(`λ = ${fmtL(lam)}`, d.W(px, py, .22), '', hex);
      }
      out.push([lam, v]);
    }
    return out;
  }

  // ---------- balls ----------
  ball(d, { start, step, color = C.ball, speed = 1, maxSteps = 5000, stop = null, trail = true, r = .065, maxTrail = 6000, lift = .06 } = {}) {
    const mesh = this.add(new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: .55, roughness: .3 })));
    let tr = null;
    if (trail) {
      const tg = new THREE.BufferGeometry();
      tg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxTrail * 3), 3)); tg.setDrawRange(0, 0);
      tr = this.add(new THREE.Line(tg, new THREE.LineBasicMaterial({ color, transparent: true, opacity: .85 })));
    }
    const b = { d, mesh, trail: tr, n: 0, maxTrail, p: start.slice(), steps: 0, done: false, speed, maxSteps, stop, step: step || opt.gd(.02), lift, hist: [] };
    b.loss = () => d.L(b.p[0], b.p[1]);
    this.place(b); this.balls.push(b); return b;
  }
  place(b) {
    const [x, y] = b.p, v = b.d.W(x, y, b.lift);
    b.mesh.position.copy(v);
    if (b.trail && b.n < b.maxTrail) {
      const a = b.trail.geometry.attributes.position; a.setXYZ(b.n++, v.x, v.y - .04, v.z);
      b.trail.geometry.setDrawRange(0, b.n); a.needsUpdate = true;
    }
  }
  /** Teleport a ball (e.g. a perturbation kick) — adds a straight trail segment. */
  kick(b, p) { b.p = b.d.clampP(p); this.place(b); }
  removeBall(b) { this.remove(b.mesh); if (b.trail) this.remove(b.trail); this.balls = this.balls.filter(x => x !== b); }
  clearBalls() { [...this.balls].forEach(b => this.removeBall(b)); }
  stepBalls() {
    for (const b of this.balls) {
      if (b.done) continue;
      const sp = b.speed * (REDUCE ? 40 : 1);
      b.acc = (b.acc || 0) + sp;
      const k = Math.floor(b.acc); b.acc -= k;
      if (!k) continue;
      for (let s = 0; s < k; s++) {
        const q = b.step(b.p, b, b.d);
        if (!isFinite(q[0]) || !isFinite(q[1])) { b.done = true; break; }
        b.p = b.d.clampP(q); b.steps++;
        if (b.onStep) b.onStep(b);
        if ((b.stop && b.stop(b)) || b.steps >= b.maxSteps) { b.done = true; break; }
      }
      this.place(b);
    }
  }

  // ---------- labels ----------
  label(html, v, cls = '', color = '') {
    const e = document.createElement('div'); e.className = 'lbl ' + cls; e.innerHTML = html; if (color) e.style.color = color;
    this.labelEl.appendChild(e); const L = { e, v, set: h => { e.innerHTML = h; } }; this.labels.push(L); return L;
  }
  placeLabels() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    for (const L of this.labels) {
      const p = L.v.clone().project(this.camera);
      L.e.hidden = p.z > 1;
      L.e.style.transform = `translate(${((p.x + 1) / 2 * w).toFixed(1)}px, ${((1 - p.y) / 2 * h).toFixed(1)}px) translate(-50%, -50%)`;
    }
  }
  onFrame(fn) { this.frameFns.push(fn); }
  start() {
    if (this.running) return; this.running = true;
    let last = performance.now();
    const loop = t => {
      if (!this.running) return;
      requestAnimationFrame(loop);
      const dt = Math.min(.05, (t - last) / 1000); last = t;
      if (!this.visible || this.hidden || this.paused) return;
      this.stepBalls();
      for (const f of this.frameFns) f(dt);
      this.controls.update(); this.renderer.render(this.scene, this.camera); this.placeLabels();
    };
    requestAnimationFrame(loop);
  }
  stop() { this.running = false; }
  renderOnce() { this.controls.update(); this.renderer.render(this.scene, this.camera); this.placeLabels(); }
  snapshot() { this.renderOnce(); return this.renderer.domElement.toDataURL('image/png'); }
}

/** Big floating labels with a live ".stat" line, e.g. under side-by-side surfaces. items: [[html, Vector3]] */
export function statLabels(stage, items) {
  return items.map(([html, v]) => { const L = stage.label(`${html}<span class="stat"></span>`, v, 'big'); L.stat = s => { const e = L.e.querySelector('.stat'); if (e) e.textContent = s; }; return L; });
}
export const ballStat = b => b ? `step ${b.steps} · loss ${fmtN(b.loss())}` : '';
const fmtN = v => (Math.abs(v) < 1e-3 && v !== 0) || Math.abs(v) >= 1e4 ? v.toExponential(1) : v.toFixed(3);

// small helpers used by many visuals
export const hyperbola = (k, dom = 1.6, sgn = 1, step = .01) => { const a = []; for (let x = Math.abs(k) / dom; x <= dom + 1e-9; x += step) a.push([sgn * x, sgn * k / x]); return a; };
export const segPts = (x0, y0, x1, y1, n = 60) => Array.from({ length: n + 1 }, (_, i) => [x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n]);
export const circlePts = (cx, cy, r, n = 80) => Array.from({ length: n + 1 }, (_, i) => [cx + r * Math.cos(2 * Math.PI * i / n), cy + r * Math.sin(2 * Math.PI * i / n)]);
