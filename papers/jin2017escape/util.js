export { rng } from '../../src/lib/math.js';
export const median = a => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
