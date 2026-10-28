// Small numerics shared by the paper visuals: seeded RNG, Gaussian samples, symmetric eigen-solver,
// finite-difference checks. Everything here is plain JS so visuals stay dependency-free.

export function rng(seed = 1) {
  // mulberry32
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  next.normal = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0;
    while (u === 0) u = next();
    v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  next.uniform = (lo, hi) => lo + (hi - lo) * next();
  return next;
}

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
export const logspace = (a, b, n) => linspace(Math.log10(a), Math.log10(b), n).map(v => 10 ** v);
export const norm = v => Math.hypot(...v);

/** Eigen-decomposition of a symmetric matrix (array of rows) by cyclic Jacobi. Returns {values, vectors} ascending;
 *  vectors[k] is the k-th eigenvector. Fine for n up to ~150. */
export function eigSym(Ain, { vectors = false, sweeps = 60, tol = 1e-12 } = {}) {
  const n = Ain.length;
  const A = Ain.map(r => Float64Array.from(r));
  const V = vectors ? Array.from({ length: n }, (_, i) => { const r = new Float64Array(n); r[i] = 1; return r; }) : null;
  for (let s = 0; s < sweeps; s++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < tol) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const th = (A[q][q] - A[p][p]) / (2 * apq);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - sn * akq;
          A[k][q] = sn * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - sn * aqk;
          A[q][k] = sn * apk + c * aqk;
        }
        if (V) for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - sn * vkq;
          V[k][q] = sn * vkp + c * vkq;
        }
      }
    }
  }
  const idx = [...Array(n).keys()].sort((i, j) => A[i][i] - A[j][j]);
  const values = idx.map(i => A[i][i]);
  return vectors ? { values, vectors: idx.map(i => V.map(r => r[i])) } : { values };
}

/** 2x2 symmetric eigen (fast path for surfaces). Returns [[l1, v1], [l2, v2]] with l1 <= l2. */
export function eig2(a, b, d) {
  const tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 - disc, l2 = tr / 2 + disc;
  const vec = l => Math.abs(b) > 1e-12 ? (() => { const x = l - d, y = b, n = Math.hypot(x, y); return [x / n, y / n]; })() : (Math.abs(a - l) < Math.abs(d - l) ? [1, 0] : [0, 1]);
  return [[l1, vec(l1)], [l2, vec(l2)]];
}

/** Numerical Hessian of a scalar function of 2 variables. */
export function hess2(f, x, y, h = 1e-4) {
  const fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h * h);
  const fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h * h);
  const fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h * h);
  return [fxx, fxy, fyy];
}

/** Dev check: compare an analytic 2D gradient with central differences at random points. Logs on mismatch. */
export function checkGrad2(name, L, g, dom = [-1.5, 1.5, -1.5, 1.5], n = 24) {
  const r = rng(99);
  let worst = 0;
  for (let i = 0; i < n; i++) {
    const x = r.uniform(dom[0], dom[1]), y = r.uniform(dom[2], dom[3]), h = 1e-6;
    const [gx, gy] = g(x, y);
    const fx = (L(x + h, y) - L(x - h, y)) / (2 * h), fy = (L(x, y + h) - L(x, y - h)) / (2 * h);
    worst = Math.max(worst, (Math.abs(gx - fx) + Math.abs(gy - fy)) / (1 + Math.abs(fx) + Math.abs(fy)));
  }
  if (worst > 1e-4) console.error(`[atlas] gradient check failed for ${name}: rel err ${worst.toExponential(2)}`);
  return worst;
}

export const fmt = (v, d = 3) => {
  if (!isFinite(v)) return '∞';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1).replace('e', 'e');
  return v.toFixed(d);
};
